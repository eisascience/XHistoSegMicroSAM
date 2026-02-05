"""
ML Model Integration

Wrapper for MedSAM and other segmentation models.
"""

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, Optional
import logging
from config import is_mps_available

logger = logging.getLogger(__name__)


# Constants for image processing
NORMALIZED_IMAGE_THRESHOLD = 1.5  # Threshold to detect [0,1] vs [0,255] images
TISSUE_BACKGROUND_RATIO_THRESHOLD = 0.5  # Threshold for tissue vs background detection
MASK_FOREGROUND_VALUE = 255  # Value for foreground pixels in binary masks
MASK_BACKGROUND_VALUE = 0  # Value for background pixels in binary masks


def _safe_load_state_dict(path: str):
    """
    Safely load model state dict from checkpoint file.
    
    Handles various PyTorch versions and checkpoint formats:
    - Always loads to CPU first to avoid CUDA-deserialize errors
    - Tries weights_only=True for newer PyTorch versions
    - Falls back for older PyTorch versions without weights_only parameter
    - Handles common checkpoint wrapper formats (state_dict, model, model_state_dict)
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        State dict ready to load into model
    """
    # Always load to CPU first to avoid CUDA-deserialize errors.
    load_kwargs = {"map_location": torch.device("cpu")}

    # Try weights_only=True when supported (newer PyTorch). Fallbacks keep it compatible.
    sd = None
    try:
        sd = torch.load(path, **load_kwargs, weights_only=True)
    except TypeError:
        # Older torch doesn't have weights_only parameter
        sd = torch.load(path, **load_kwargs)
    except Exception:
        # If weights_only=True fails for some other reason (e.g., pickle error), fall back
        # First try weights_only=False, which is safer than no parameter on newer PyTorch
        try:
            sd = torch.load(path, **load_kwargs, weights_only=False)
        except TypeError:
            # Very old PyTorch without weights_only parameter at all
            sd = torch.load(path, **load_kwargs)

    # Handle common checkpoint wrappers
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break

    return sd


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in HWC RGB uint8 format for SAM.
    
    Handles:
    - Float [0,1] to uint8 [0,255] conversion
    - Grayscale to 3-channel RGB
    - Various input array shapes
    
    Args:
        image: Input image in various formats:
               - (H, W) - grayscale
               - (H, W, 1) - grayscale with channel dimension
               - (H, W, 3) - RGB
               - dtype can be uint8, float32, or float64
        
    Returns:
        Image in HWC RGB uint8 format with shape (H, W, 3)
        
    Examples:
        >>> gray = np.random.rand(100, 100)  # Float grayscale
        >>> rgb = _ensure_rgb_uint8(gray)
        >>> rgb.shape, rgb.dtype
        ((100, 100, 3), dtype('uint8'))
        
        >>> color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> result = _ensure_rgb_uint8(color)
        >>> np.array_equal(result, color)
        True
    """
    # Convert to numpy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Handle float [0,1] to uint8 [0,255]
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= NORMALIZED_IMAGE_THRESHOLD:  # Normalized [0,1]
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    else:
        # Already integer type, just ensure uint8
        image = image.astype(np.uint8)
    
    # Handle grayscale (HW) or (HW1) to RGB (HW3)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    
    # Ensure we have 3 channels
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have 3 channels, got shape {image.shape}")
    
    # Note: We assume images are already RGB. If BGR conversion is needed,
    # it should be done before calling this function.
    
    return image


def _compute_tissue_bbox(image: np.ndarray, 
                         min_area_ratio: float = 0.01,
                         morph_kernel_size: int = 5) -> np.ndarray:
    """
    Compute bounding box around tissue region using simple thresholding.
    
    Algorithm:
    1. Convert RGB image to grayscale
    2. Apply Otsu's thresholding to separate foreground from background
    3. Invert binary image if tissue was marked as background (darker regions)
    4. Apply morphological closing to fill holes
    5. Apply morphological opening to remove noise
    6. Find connected components
    7. Select largest component as tissue region
    8. Return bounding box of largest component
    
    Args:
        image: RGB uint8 image (HWC format)
        min_area_ratio: Minimum area ratio to consider valid tissue
                       (e.g., 0.01 means tissue must cover at least 1% of image)
        morph_kernel_size: Kernel size for morphological operations (should be odd)
        
    Returns:
        Bounding box as [x1, y1, x2, y2] or full image box if detection fails
        
    Edge Cases:
        - If no tissue detected: returns full image bounding box
        - If detected tissue too small: returns full image bounding box
        - If error occurs: returns full image bounding box (logged as error)
    """
    h, w = image.shape[:2]
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Otsu threshold to separate tissue from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # In pathology images, tissue is typically darker than background
        # Otsu makes darker regions black (0) and lighter regions white (255)
        # For tissue detection, we want tissue to be white (foreground)
        # If less than 50% is white after Otsu, tissue was likely marked as black
        white_ratio = np.sum(binary == 255) / (h * w)
        if white_ratio < TISSUE_BACKGROUND_RATIO_THRESHOLD:
            # Invert: tissue (currently black) becomes white
            binary = 255 - binary
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (morph_kernel_size, morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            # No components found, return full image
            logger.warning("No tissue components found, using full image box")
            return np.array([0, 0, w - 1, h - 1])
        
        # Find largest component (excluding background which is label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        # Check if area is significant
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
        if largest_area < (w * h * min_area_ratio):
            logger.warning(f"Largest component too small ({largest_area} pixels), using full image box")
            return np.array([0, 0, w - 1, h - 1])
        
        # Get bounding box
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        bbox_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        bbox_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        
        # Return as [x1, y1, x2, y2]
        bbox = np.array([x, y, x + bbox_w - 1, y + bbox_h - 1])
        
        logger.info(f"Detected tissue bbox: {bbox}, area: {largest_area} pixels ({100*largest_area/(w*h):.1f}%)")
        
        return bbox
        
    except Exception as e:
        logger.error(f"Error computing tissue bbox: {e}, using full image box")
        return np.array([0, 0, w - 1, h - 1])


class MedSAMPredictor:
    """
    Wrapper for MedSAM model inference.
    
    MedSAM is a medical image segmentation model based on SAM (Segment Anything Model).
    """
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
        """
        Initialize MedSAM predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            model_type: Model architecture (vit_b, vit_l, vit_h)
            device: Device to run inference on (cuda, mps, or cpu)
        """
        # Respect user's device choice, only fall back if requested device unavailable
        valid_devices = ["cuda", "mps", "cpu"]
        if device not in valid_devices:
            logger.warning(f"Invalid device '{device}' specified. Valid devices: {valid_devices}. Falling back to CPU")
            self.device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            # Check for MPS as fallback
            if is_mps_available():
                logger.warning("CUDA requested but not available, falling back to MPS")
                self.device = "mps"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
        elif device == "mps" and not is_mps_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        self.model_type = model_type
        
        try:
            self.model = self._load_model(checkpoint_path, model_type)
            self.model.eval()
            
            # Create SamPredictor for proper inference
            self.sam_predictor = SamPredictor(self.model)
            
            logger.info(f"Loaded MedSAM model ({model_type}) on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_model(self, checkpoint_path: str, model_type: str):
        """
        Load MedSAM model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_type: Model architecture
            
        Returns:
            Loaded model
        """
        try:
            # Use the device that was already determined and validated in __init__
            device = torch.device(self.device)

            # IMPORTANT: build WITHOUT checkpoint to avoid internal torch.load(...)
            model = sam_model_registry[model_type](checkpoint=None)

            state_dict = _safe_load_state_dict(checkpoint_path)
            model.load_state_dict(state_dict)

            model.to(device)
            
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @torch.no_grad()
    def predict(self, 
                image: np.ndarray, 
                prompt_mode: str = "auto_box",
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                multimask_output: bool = False,
                min_area_ratio: float = 0.01,
                morph_kernel_size: int = 5) -> np.ndarray:
        """
        Run inference on image using SAM with proper prompts.
        
        Args:
            image: Input image (will be converted to HWC RGB uint8)
            prompt_mode: Prompt mode - "auto_box" (detect tissue), "full_box", or "point"
            point_coords: Point prompts (N, 2) in (x, y) format (for "point" mode)
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt [x1, y1, x2, y2] (overrides prompt_mode if provided)
            multimask_output: Whether to return multiple mask predictions
            min_area_ratio: Minimum area ratio for auto tissue detection
            morph_kernel_size: Kernel size for morphological operations in auto detection
            
        Returns:
            Binary mask as uint8 numpy array (0 or 255)
        """
        # Ensure proper image format
        image_rgb_uint8 = _ensure_rgb_uint8(image)
        h, w = image_rgb_uint8.shape[:2]
        
        logger.info(f"Processing image: shape={image_rgb_uint8.shape}, dtype={image_rgb_uint8.dtype}")
        
        # Set image for SAM predictor
        self.sam_predictor.set_image(image_rgb_uint8)
        
        # Determine prompt based on mode
        if box is not None:
            # Use provided box
            prompt_box = box
            logger.info(f"Using provided box: {prompt_box}")
        elif prompt_mode == "auto_box":
            # Auto-detect tissue region
            prompt_box = _compute_tissue_bbox(image_rgb_uint8, min_area_ratio, morph_kernel_size)
            logger.info(f"Auto-detected tissue box: {prompt_box}")
        elif prompt_mode == "full_box":
            # Use full image as box
            prompt_box = np.array([0, 0, w - 1, h - 1])
            logger.info(f"Using full image box: {prompt_box}")
        elif prompt_mode == "point":
            # Use point prompts
            if point_coords is None or point_labels is None:
                # Default to center point if not provided
                center_point = np.array([[w // 2, h // 2]])
                point_coords = center_point
                point_labels = np.array([1])
                logger.info(f"Using default center point: {center_point}")
            prompt_box = None
        else:
            raise ValueError(f"Invalid prompt_mode: {prompt_mode}. Must be 'auto_box', 'full_box', or 'point'")
        
        # Run prediction using SamPredictor
        if prompt_box is not None:
            # Box-based prediction
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=prompt_box,
                multimask_output=multimask_output,
            )
        else:
            # Point-based prediction
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=multimask_output,
            )
        
        # Get best mask (highest score)
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            best_score = scores[best_idx]
        else:
            mask = masks[0]
            best_score = scores[0]
        
        # Convert to uint8 binary mask (0 or 255)
        binary_mask = (mask > 0).astype(np.uint8) * MASK_FOREGROUND_VALUE
        
        # Log statistics
        mask_area = np.sum(mask > 0)
        mask_ratio = mask_area / (h * w)
        logger.info(f"Generated mask: area={mask_area} pixels ({100*mask_ratio:.1f}% of image), score={best_score:.3f}")
        
        return binary_mask
    
    def predict_with_box(self, image: np.ndarray, box: np.ndarray, multimask_output: bool = False) -> np.ndarray:
        """
        Convenience method for box-based segmentation.
        
        Args:
            image: Input image (will be converted to HWC RGB uint8)
            box: Bounding box [x1, y1, x2, y2]
            multimask_output: Whether to return multiple mask predictions
            
        Returns:
            Binary mask as uint8 (0 or 255)
        """
        return self.predict(image, box=box, multimask_output=multimask_output)
    
    def predict_with_points(self, image: np.ndarray, 
                          points: np.ndarray, labels: np.ndarray,
                          multimask_output: bool = False) -> np.ndarray:
        """
        Convenience method for point-based segmentation.
        
        Args:
            image: Input image (will be converted to HWC RGB uint8)
            points: Point coordinates (N, 2) in (x, y) format
            labels: Point labels (N,) - 1 for foreground, 0 for background
            multimask_output: Whether to return multiple mask predictions
            
        Returns:
            Binary mask as uint8 (0 or 255)
        """
        return self.predict(image, prompt_mode="point", point_coords=points, 
                          point_labels=labels, multimask_output=multimask_output)
    
    def batch_predict(self, images: list, prompts: list) -> list:
        """
        Run batch inference on multiple images.
        
        Args:
            images: List of images (will be converted to HWC RGB uint8)
            prompts: List of prompt dictionaries with keys:
                     - 'mode': 'auto_box', 'full_box', or 'point'
                     - 'points': point coordinates (optional)
                     - 'labels': point labels (optional)
                     - 'box': bounding box (optional)
            
        Returns:
            List of binary masks as uint8 (0 or 255)
        """
        masks = []
        for image, prompt in zip(images, prompts):
            mask = self.predict(
                image,
                prompt_mode=prompt.get('mode', 'auto_box'),
                point_coords=prompt.get('points'),
                point_labels=prompt.get('labels'),
                box=prompt.get('box'),
                multimask_output=prompt.get('multimask_output', False)
            )
            masks.append(mask)
        return masks
