"""
MicroSAM Adapter for XHaloPathAnalyzer

Provides a MedSAM-compatible interface for MicroSAM to minimize changes to existing code.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Import MicroSAM predictor and elf availability check
from xhalo.ml import MicroSAMPredictor as BaseMicroSAMPredictor, is_elf_available


class MicroSAMPredictor:
    """
    Adapter class that provides a MedSAM-compatible interface for MicroSAM.
    
    This allows minimal changes to existing app.py code while using MicroSAM backend.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b_histopathology",
        device: Optional[str] = None,
        segmentation_mode: str = "interactive",  # "interactive" or "automatic"
        tile_shape: Tuple[int, int] = (1024, 1024),
        halo: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize MicroSAM predictor with MedSAM-compatible interface.
        
        Args:
            model_type: Model architecture (default: vit_b_histopathology)
            device: Device to use (cuda/mps/cpu), auto-detected if None
            segmentation_mode: "interactive" (with prompts) or "automatic" (no prompts)
            tile_shape: Tile size for tiled inference
            halo: Halo size for tiled inference
        """
        # Check if automatic mode is requested but elf is not available
        if segmentation_mode == "automatic" and not is_elf_available():
            logger.warning(
                "Automatic segmentation mode requires python-elf, which is not available. "
                "Falling back to interactive mode. Use prompt-based modes instead."
            )
            segmentation_mode = "interactive"
        
        self.segmentation_mode = segmentation_mode
        self.device = device
        
        # Initialize base MicroSAM predictor
        logger.info(f"Initializing MicroSAM (mode={segmentation_mode}, model={model_type})")
        self.predictor = BaseMicroSAMPredictor(
            model_type=model_type,
            device=device,
            tile_shape=tile_shape,
            halo=halo
        )
        
        logger.info(f"MicroSAM adapter ready (device={self.predictor.device})")
    
    def predict(
        self,
        image: np.ndarray,
        prompt_mode: str = "auto_box",
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        min_area_ratio: float = 0.01,
        morph_kernel_size: int = 5,
        threshold_mode: str = "otsu",
        threshold_value: float = 0.5,
        box_min_area: int = 100,
        box_max_area: int = 100000,
        box_dilation_radius: int = 0
    ) -> np.ndarray:
        """
        Run inference with MedSAM-compatible interface.
        
        Args:
            image: Input RGB image (H, W, 3)
            prompt_mode: Prompt mode - "auto_box", "full_box", "point", or "auto_box_from_threshold"
            point_coords: Point prompts (N, 2) in (x, y) format
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt [x1, y1, x2, y2]
            multimask_output: Whether to use multimasking
            min_area_ratio: Minimum area ratio for auto tissue detection
            morph_kernel_size: Kernel size for morphological operations
            threshold_mode: Threshold mode for auto_box_from_threshold ("manual", "otsu", "off")
            threshold_value: Threshold value (0-1) for manual mode
            box_min_area: Minimum box area for auto_box_from_threshold
            box_max_area: Maximum box area for auto_box_from_threshold
            box_dilation_radius: Dilation radius for auto_box_from_threshold
            
        Returns:
            Binary mask (H, W) with 0=background, 255=foreground
            OR instance mask (H, W) with 0=background, 1..N=instances (if automatic mode or auto_box_from_threshold)
        """
        # Ensure RGB format
        image = self.predictor.ensure_rgb(image)
        h, w = image.shape[:2]
        
        logger.info(f"MicroSAM prediction: mode={self.segmentation_mode}, prompt_mode={prompt_mode}")
        
        if self.segmentation_mode == "automatic":
            # Automatic instance segmentation (no prompts)
            instance_mask = self.predictor.predict_auto_instances(
                image=image,
                segmentation_mode="apg",
                min_size=25,
                tiled=True
            )
            
            # Convert to binary mask for compatibility
            # (or return instance mask if post-processing can handle it)
            # For now, return binary mask where any instance = 255
            binary_mask = (instance_mask > 0).astype(np.uint8) * 255
            
            logger.info(f"Automatic segmentation: {np.unique(instance_mask).size - 1} instances")
            return binary_mask
            
        else:
            # Interactive prompted segmentation
            # Prepare prompts based on mode
            boxes_prompt = None
            points_prompt = None
            labels_prompt = None
            
            if box is not None:
                # Use provided box
                boxes_prompt = np.array([box])  # Shape (1, 4)
                logger.info(f"Using provided box: {box}")
                
            elif prompt_mode == "auto_box_from_threshold":
                # Generate boxes from thresholded image
                boxes, binary_threshold_mask = compute_boxes_from_threshold(
                    image=image,
                    threshold_mode=threshold_mode,
                    threshold_value=threshold_value,
                    min_area=box_min_area,
                    max_area=box_max_area,
                    dilation_radius=box_dilation_radius,
                    normalize=True
                )
                
                if len(boxes) == 0:
                    logger.warning("No boxes found from threshold. Returning empty mask.")
                    return np.zeros((h, w), dtype=np.uint8)
                
                boxes_prompt = boxes
                logger.info(f"Generated {len(boxes)} boxes from threshold")
                
            elif prompt_mode == "auto_box":
                # Auto-detect tissue region
                from utils.medsam_models import _compute_tissue_bbox
                auto_box = _compute_tissue_bbox(image, min_area_ratio, morph_kernel_size)
                boxes_prompt = np.array([auto_box])  # Shape (1, 4)
                logger.info(f"Auto-detected tissue box: {auto_box}")
                
            elif prompt_mode == "full_box":
                # Use full image as box
                full_box = np.array([0, 0, w - 1, h - 1])
                boxes_prompt = np.array([full_box])  # Shape (1, 4)
                logger.info(f"Using full image box")
                
            elif prompt_mode == "point":
                # Use point prompts
                if point_coords is not None and point_labels is not None:
                    # MicroSAM expects (N, 1, 2) for points
                    points_prompt = point_coords[:, np.newaxis, :]  # (N, 1, 2)
                    labels_prompt = point_labels[:, np.newaxis]  # (N, 1)
                    logger.info(f"Using {len(point_coords)} point prompts")
                else:
                    # Default to center point
                    center = np.array([[[w // 2, h // 2]]])  # (1, 1, 2)
                    points_prompt = center
                    labels_prompt = np.array([[1]])  # Positive point
                    logger.info("Using default center point")
            
            # Run prompted inference
            try:
                instance_mask = self.predictor.predict_from_prompts(
                    image=image,
                    boxes=boxes_prompt,
                    points=points_prompt,
                    point_labels=labels_prompt,
                    multimasking=multimask_output,
                    tiled=True
                )
                
                # For auto_box_from_threshold, return instance mask with IDs
                # For other modes, convert to binary mask for compatibility
                if prompt_mode == "auto_box_from_threshold":
                    logger.info(f"Instance segmentation complete: {np.unique(instance_mask).size - 1} instances")
                    return instance_mask  # Return instance IDs
                else:
                    binary_mask = (instance_mask > 0).astype(np.uint8) * 255
                    logger.info(f"Interactive segmentation complete")
                    return binary_mask
                
            except Exception as e:
                logger.error(f"MicroSAM prediction failed: {e}")
                # Return empty mask on failure
                return np.zeros((h, w), dtype=np.uint8)


# For backward compatibility, also export helper functions
def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8 format."""
    from utils.medsam_models import _ensure_rgb_uint8 as original_ensure_rgb
    return original_ensure_rgb(image)


def _compute_tissue_bbox(
    image: np.ndarray,
    min_area_ratio: float = 0.01,
    morph_kernel_size: int = 5
) -> np.ndarray:
    """Compute tissue bounding box."""
    from utils.medsam_models import _compute_tissue_bbox as original_compute_bbox
    return original_compute_bbox(image, min_area_ratio, morph_kernel_size)


def compute_boxes_from_threshold(
    image: np.ndarray,
    threshold_mode: str = "otsu",
    threshold_value: float = 0.5,
    min_area: int = 100,
    max_area: int = 100000,
    dilation_radius: int = 0,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bounding boxes from thresholded image for auto_box_from_threshold mode.
    
    This is useful for cell/nucleus segmentation where you want to:
    1. Threshold a DAPI or similar channel
    2. Extract connected components
    3. Generate bounding boxes for each component
    4. Use boxes as prompts for MicroSAM instance segmentation
    
    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        threshold_mode: "manual", "otsu", or "off"
        threshold_value: Threshold value (0-1) if manual mode
        min_area: Minimum area in pixels for components
        max_area: Maximum area in pixels for components
        dilation_radius: Pixels to dilate mask before boxing (to capture full nuclei)
        normalize: Whether to normalize image before thresholding
        
    Returns:
        boxes: (N, 4) array of boxes [x1, y1, x2, y2]
        binary_mask: (H, W) binary mask of thresholded regions
    """
    from skimage import measure, morphology
    from skimage.filters import threshold_otsu
    import cv2
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            # RGB to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]  # Take first channel
    else:
        gray = image.copy()
    
    # Normalize to [0, 1] if needed
    if normalize:
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    
    # Apply threshold
    if threshold_mode == "off":
        # No thresholding, use full image
        binary_mask = np.ones_like(gray, dtype=bool)
    elif threshold_mode == "otsu":
        # Otsu's method - always needs normalized input
        gray_norm = gray if normalize else gray.astype(np.float32) / 255.0
        thresh = threshold_otsu(gray_norm)
        binary_mask = gray_norm > thresh
        logger.info(f"Otsu threshold: {thresh:.3f}")
    else:  # manual
        binary_mask = gray > threshold_value
        logger.info(f"Manual threshold: {threshold_value:.3f}")
    
    # Apply dilation if requested
    if dilation_radius > 0:
        struct = morphology.disk(dilation_radius)
        binary_mask = morphology.binary_dilation(binary_mask, struct)
        logger.info(f"Applied dilation with radius {dilation_radius}")
    
    # Label connected components
    labeled_mask = measure.label(binary_mask)
    regions = measure.regionprops(labeled_mask)
    
    logger.info(f"Found {len(regions)} connected components before filtering")
    
    # Filter by area and extract bounding boxes
    boxes = []
    for region in regions:
        area = region.area
        if min_area <= area <= max_area:
            # Get bounding box in (min_row, min_col, max_row, max_col) format
            min_row, min_col, max_row, max_col = region.bbox
            # Convert to (x1, y1, x2, y2) format for MicroSAM
            boxes.append([min_col, min_row, max_col, max_row])
    
    boxes = np.array(boxes) if boxes else np.empty((0, 4))
    
    logger.info(f"Extracted {len(boxes)} boxes after area filtering (min={min_area}, max={max_area})")
    
    return boxes, binary_mask.astype(np.uint8)
