"""
MicroSAM Adapter for XHaloPathAnalyzer

Provides a MedSAM-compatible interface for MicroSAM to minimize changes to existing code.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Import MicroSAM predictor
from xhalo.ml import MicroSAMPredictor as BaseMicroSAMPredictor


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
        morph_kernel_size: int = 5
    ) -> np.ndarray:
        """
        Run inference with MedSAM-compatible interface.
        
        Args:
            image: Input RGB image (H, W, 3)
            prompt_mode: Prompt mode - "auto_box", "full_box", or "point" (ignored in automatic mode)
            point_coords: Point prompts (N, 2) in (x, y) format
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt [x1, y1, x2, y2]
            multimask_output: Whether to use multimasking
            min_area_ratio: Minimum area ratio for auto tissue detection
            morph_kernel_size: Kernel size for morphological operations
            
        Returns:
            Binary mask (H, W) with 0=background, 255=foreground
            OR instance mask (H, W) with 0=background, 1..N=instances (if automatic mode)
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
                
            elif prompt_mode == "auto_box":
                # Auto-detect tissue region
                from utils.ml_models import _compute_tissue_bbox
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
                
                # Convert to binary mask for compatibility
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
    from utils.ml_models import _ensure_rgb_uint8 as original_ensure_rgb
    return original_ensure_rgb(image)


def _compute_tissue_bbox(
    image: np.ndarray,
    min_area_ratio: float = 0.01,
    morph_kernel_size: int = 5
) -> np.ndarray:
    """Compute tissue bounding box."""
    from utils.ml_models import _compute_tissue_bbox as original_compute_bbox
    return original_compute_bbox(image, min_area_ratio, morph_kernel_size)
