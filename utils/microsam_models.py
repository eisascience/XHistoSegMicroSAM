"""
MicroSAM Model Integration

Wrapper for micro-sam segmentation models with support for:
- Interactive prompted instance segmentation (box + point prompts)
- Automatic instance segmentation (no prompts)
- Tiled inference for large images
- Embeddings caching for faster inference
"""

import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MicroSAMPredictor:
    """
    Wrapper for micro-sam model inference.
    
    Provides two modes:
    - Interactive: prompted instance segmentation using boxes or points
    - Automatic: automatic instance segmentation without prompts
    """
    
    def __init__(
        self,
        model_type: str = "vit_b_histopathology",
        device: Optional[str] = None,
        tile_shape: Tuple[int, int] = (1024, 1024),
        halo: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize MicroSAM predictor.
        
        Args:
            model_type: Model architecture to use (default: vit_b_histopathology)
                       Histopathology models: vit_b_histopathology, vit_l_histopathology
                       Standard SAM models: vit_b, vit_l, vit_h
            device: Device to run on (cuda/mps/cpu). Auto-detected if None.
            tile_shape: Tile size for tiled inference (height, width)
            halo: Halo/overlap size for tiled inference (height, width)
        """
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model_type = model_type
        self.tile_shape = tile_shape
        self.halo = halo
        
        logger.info(f"Initializing MicroSAM with model={model_type}, device={self.device}")
        
        # Import and load micro_sam model
        try:
            from micro_sam.util import get_sam_model
            self.predictor = get_sam_model(
                model_type=model_type,
                device=self.device
            )
            logger.info("MicroSAM model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import micro_sam: {e}")
            logger.error("Please install micro-sam and segment-anything")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MicroSAM: {e}")
            raise
    
    def ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is RGB format (H, W, 3) with uint8 dtype.
        
        Handles various input formats:
        - Grayscale (H, W) or (H, W, 1): replicate to 3 channels
        - RGB (H, W, 3): pass through
        - RGBA (H, W, 4): drop alpha channel
        - Multi-channel (H, W, C) where C > 4: select first 3 channels with warning
        
        Args:
            image: Input image array
            
        Returns:
            RGB image (H, W, 3) with uint8 dtype
        """
        # Handle grayscale
        if image.ndim == 2:
            # HxW -> HxWx3
            image = np.stack([image, image, image], axis=-1)
            logger.info("Converted grayscale (H,W) to RGB by replicating channel")
        elif image.ndim == 3:
            if image.shape[2] == 1:
                # HxWx1 -> HxWx3
                image = np.repeat(image, 3, axis=2)
                logger.info("Converted grayscale (H,W,1) to RGB by replicating channel")
            elif image.shape[2] == 3:
                # Already RGB
                pass
            elif image.shape[2] == 4:
                # RGBA -> RGB
                image = image[:, :, :3]
                logger.info("Converted RGBA to RGB by dropping alpha channel")
            elif image.shape[2] > 4:
                # Multi-channel -> select first 3
                logger.warning(
                    f"Image has {image.shape[2]} channels. "
                    f"Selecting first 3 channels as RGB. If your channels represent non-RGB data "
                    f"(e.g., fluorescence), results may be incorrect. Consider converting to RGB "
                    f"before calling this method."
                )
                image = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                # Normalized [0, 1] -> [0, 255]
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    def precompute_embeddings(
        self,
        image: np.ndarray,
        embeddings_path: Union[str, Path],
        tiled: bool = True
    ) -> None:
        """
        Precompute and cache image embeddings for faster inference.
        
        Args:
            image: RGB image array (H, W, 3)
            embeddings_path: Path to save embeddings (zarr format)
            tiled: Whether to use tiled embedding computation
        """
        from micro_sam.util import precompute_image_embeddings
        
        embeddings_path = Path(embeddings_path)
        
        # Check if embeddings already exist
        if embeddings_path.exists():
            logger.info(f"Embeddings already exist at {embeddings_path}, skipping precomputation")
            return
        
        # Ensure RGB format
        image = self.ensure_rgb(image)
        
        logger.info(f"Precomputing embeddings (tiled={tiled}) to {embeddings_path}")
        
        # Prepare kwargs
        kwargs = {
            "predictor": self.predictor,
            "input_": image,
            "save_path": str(embeddings_path),
            "verbose": True
        }
        
        if tiled:
            kwargs.update({
                "tile_shape": self.tile_shape,
                "halo": self.halo
            })
        
        try:
            precompute_image_embeddings(**kwargs)
            logger.info("Embeddings precomputed successfully")
        except Exception as e:
            logger.error(f"Failed to precompute embeddings: {e}")
            raise
    
    @torch.no_grad()
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
        tiled: bool = True,
        embeddings_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Run inference on image using prompts (MedSAM-compatible interface).
        
        Args:
            image: Input image (will be converted to HWC RGB uint8)
            prompt_mode: Prompt mode - "auto_box", "full_box", or "point"
            point_coords: Point prompts (N, 2) in (x, y) format
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt [x1, y1, x2, y2]
            multimask_output: Whether to use multimasking
            min_area_ratio: Minimum area ratio for auto tissue detection
            morph_kernel_size: Kernel size for morphological operations
            tiled: Whether to use tiled inference
            embeddings_path: Optional path to precomputed embeddings
            
        Returns:
            Binary mask as uint8 numpy array (0 or 255)
        """
        from micro_sam.inference import batched_inference, batched_tiled_inference
        
        # Ensure proper image format
        image_rgb = self.ensure_rgb(image)
        h, w = image_rgb.shape[:2]
        
        logger.info(f"Processing image: shape={image_rgb.shape}, dtype={image_rgb.dtype}")
        
        # Prepare prompts based on mode
        boxes_prompt = None
        points_prompt = None
        labels_prompt = None
        
        if box is not None:
            # Use provided box
            boxes_prompt = np.array([box])  # Shape (1, 4)
            logger.info(f"Using provided box: {box}")
            
        elif prompt_mode == "auto_box":
            # Auto-detect tissue region
            auto_box = self._compute_tissue_bbox(image_rgb, min_area_ratio, morph_kernel_size)
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
        else:
            raise ValueError(f"Invalid prompt_mode: {prompt_mode}")
        
        # Select inference function
        inference_fn = batched_tiled_inference if tiled else batched_inference
        
        # Prepare kwargs
        kwargs = {
            "predictor": self.predictor,
            "image": image_rgb,
            "batch_size": 1,
            "return_instance_segmentation": True
        }
        
        if embeddings_path is not None:
            kwargs["embedding_path"] = str(embeddings_path)
        
        if tiled:
            kwargs.update({
                "tile_shape": self.tile_shape,
                "halo": self.halo
            })
        
        if boxes_prompt is not None:
            kwargs["boxes"] = boxes_prompt
        
        if points_prompt is not None:
            kwargs["points"] = points_prompt
        
        if labels_prompt is not None:
            kwargs["point_labels"] = labels_prompt
        
        if multimask_output:
            kwargs["multimasking"] = multimask_output
        
        logger.info(f"Running prompted instance segmentation (tiled={tiled})")
        
        try:
            instance_mask = inference_fn(**kwargs)
            
            # Convert to binary mask (0 or 255)
            binary_mask = (instance_mask > 0).astype(np.uint8) * 255
            
            # Log statistics
            mask_area = np.sum(binary_mask > 0)
            mask_ratio = mask_area / (h * w)
            logger.info(f"Generated mask: area={mask_area} pixels ({100*mask_ratio:.1f}% of image)")
            
            return binary_mask
            
        except Exception as e:
            logger.error(f"MicroSAM prediction failed: {e}")
            # Return empty mask on failure
            return np.zeros((h, w), dtype=np.uint8)
    
    def predict_auto_instances(
        self,
        image: np.ndarray,
        embeddings_path: Optional[Union[str, Path]] = None,
        segmentation_mode: str = "apg",
        min_size: int = 25,
        foreground_threshold: float = 0.5,
        center_distance_threshold: float = 0.5,
        boundary_distance_threshold: float = 0.5,
        nms_threshold: float = 0.7,
        multimasking: bool = False,
        tiled: bool = True
    ) -> np.ndarray:
        """
        Perform automatic instance segmentation without prompts.
        
        Args:
            image: RGB image array (H, W, 3)
            embeddings_path: Optional path to precomputed embeddings
            segmentation_mode: Mode ("apg", "ais", or "amg")
            min_size: Minimum object size in pixels
            foreground_threshold: Threshold for foreground detection
            center_distance_threshold: Distance threshold for center detection
            boundary_distance_threshold: Distance threshold for boundary detection
            nms_threshold: Non-maximum suppression threshold
            multimasking: Whether to use multi-mask prediction
            tiled: Whether to use tiled inference
            
        Returns:
            Instance segmentation mask (H, W) with 0=background, 1..N=instance IDs
        """
        from micro_sam.instance_segmentation import get_instance_segmentation_generator
        
        # Ensure RGB format
        image = self.ensure_rgb(image)
        
        # Precompute embeddings if path provided
        if embeddings_path is not None:
            self.precompute_embeddings(image, embeddings_path, tiled=tiled)
        
        logger.info(f"Running automatic instance segmentation (mode={segmentation_mode}, tiled={tiled})")
        
        # Get instance segmentation generator
        try:
            generator = get_instance_segmentation_generator(
                predictor=self.predictor,
                segmentation_mode=segmentation_mode,
                min_size=min_size,
                output_mode="binary_mask"
            )
        except Exception as e:
            logger.error(f"Failed to create instance segmentation generator: {e}")
            raise
        
        # Prepare inference kwargs
        kwargs = {
            "image": image,
        }
        
        if embeddings_path is not None:
            kwargs["embedding_path"] = str(embeddings_path)
        
        if tiled:
            kwargs.update({
                "tile_shape": self.tile_shape,
                "halo": self.halo
            })
        
        # Add mode-specific parameters
        if segmentation_mode == "apg":
            kwargs.update({
                "foreground_threshold": foreground_threshold,
                "center_distance_threshold": center_distance_threshold,
                "boundary_distance_threshold": boundary_distance_threshold,
            })
        
        if nms_threshold is not None:
            kwargs["nms_threshold"] = nms_threshold
        
        if multimasking:
            kwargs["multimasking"] = multimasking
        
        try:
            # Generate instance segmentation
            instance_mask = generator(**kwargs)
            
            # Ensure output is instance-labeled array
            if isinstance(instance_mask, dict):
                instance_mask = instance_mask.get("segmentation", instance_mask)
            
            unique_vals = np.unique(instance_mask)
            num_instances = unique_vals.size - 1 if 0 in unique_vals else unique_vals.size
            logger.info(f"Automatic segmentation complete: {num_instances} instances found")
            
            return instance_mask
            
        except Exception as e:
            logger.error(f"Failed to run automatic segmentation: {e}")
            raise
    
    def _compute_tissue_bbox(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.01,
        morph_kernel_size: int = 5
    ) -> np.ndarray:
        """
        Compute bounding box around tissue region using simple thresholding.
        
        Args:
            image: RGB uint8 image (HWC format)
            min_area_ratio: Minimum area ratio to consider valid tissue
            morph_kernel_size: Kernel size for morphological operations
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Otsu threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Check if tissue is darker than background
            white_ratio = np.sum(binary == 255) / (h * w)
            if white_ratio < 0.5:
                binary = 255 - binary
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels <= 1:
                logger.warning("No tissue components found, using full image box")
                return np.array([0, 0, w - 1, h - 1])
            
            # Find largest component
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            
            # Check if area is significant
            largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
            if largest_area < (w * h * min_area_ratio):
                logger.warning(f"Largest component too small, using full image box")
                return np.array([0, 0, w - 1, h - 1])
            
            # Get bounding box
            x = stats[largest_idx, cv2.CC_STAT_LEFT]
            y = stats[largest_idx, cv2.CC_STAT_TOP]
            bbox_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
            bbox_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
            
            bbox = np.array([x, y, x + bbox_w - 1, y + bbox_h - 1])
            logger.info(f"Detected tissue bbox: {bbox}")
            
            return bbox
            
        except Exception as e:
            logger.error(f"Error computing tissue bbox: {e}, using full image box")
            return np.array([0, 0, w - 1, h - 1])
    
    def predict_with_box(self, image: np.ndarray, box: np.ndarray, multimask_output: bool = False) -> np.ndarray:
        """
        Convenience method for box-based segmentation.
        
        Args:
            image: Input image
            box: Bounding box [x1, y1, x2, y2]
            multimask_output: Whether to use multimasking
            
        Returns:
            Binary mask as uint8 (0 or 255)
        """
        return self.predict(image, box=box, multimask_output=multimask_output)
    
    def predict_with_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = False
    ) -> np.ndarray:
        """
        Convenience method for point-based segmentation.
        
        Args:
            image: Input image
            points: Point coordinates (N, 2) in (x, y) format
            labels: Point labels (N,) - 1 for foreground, 0 for background
            multimask_output: Whether to use multimasking
            
        Returns:
            Binary mask as uint8 (0 or 255)
        """
        return self.predict(
            image,
            prompt_mode="point",
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output
        )
