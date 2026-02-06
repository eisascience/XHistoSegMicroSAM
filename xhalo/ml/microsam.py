"""
MicroSAM Segmentation Module

Provides integration with micro-sam for histopathology image segmentation.
Supports both interactive prompted segmentation and automatic instance segmentation.

Memory Management Notes:
- For images with many objects (500+), memory can be exhausted
- Reduce TILE_SHAPE in .env (e.g., 512,512 or 256,256) 
- Reduce MICROSAM_BATCH_SIZE in .env (e.g., 4 or 8)
- Set MICROSAM_DEVICE=cpu in .env to avoid GPU OOM
- MPS cache is automatically cleared before each prediction
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union
import logging
from pathlib import Path
import warnings
import importlib.util
import os

logger = logging.getLogger(__name__)

# Check if elf is available at runtime
# elf (python-elf) is required for automatic instance segmentation (APG/AIS modes)
# but cannot be installed in Python 3.11 due to numba/llvmlite constraints
_ELF_AVAILABLE = importlib.util.find_spec("elf") is not None

if not _ELF_AVAILABLE:
    # Log at debug level on import - user will see info message when they try to use it
    logger.debug(
        "python-elf is not available. Automatic instance segmentation modes (APG/AIS) will not work. "
        "Prompt-based modes (point, auto_box, auto_box_from_threshold) work without elf."
    )


class MicroSAMPredictor:
    """
    Wrapper for micro-sam model inference with support for:
    - Interactive prompted instance segmentation (box + point prompts)
    - Automatic instance segmentation (apg, ais, amg modes)
    - Tiled inference for large images
    - Embeddings caching for faster inference
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
                       Options: vit_b_histopathology, vit_l_histopathology, vit_b, vit_l, vit_h
            device: Device to run inference on (cuda/mps/cpu). Auto-detected if None.
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
        
        # Get batch size from environment or use default
        self.batch_size = int(os.getenv("MICROSAM_BATCH_SIZE", "8"))
        
        logger.info(f"Initializing MicroSAM with model={model_type}, device={self.device}, batch_size={self.batch_size}")
        
        # Import micro_sam modules
        try:
            from micro_sam.util import get_sam_model
            self.predictor = get_sam_model(
                model_type=model_type,
                device=self.device
            )
            logger.info("MicroSAM model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import micro_sam: {e}")
            logger.error("Please install micro-sam: pip install micro-sam")
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
                    f"Selecting first 3 channels. This may change behavior - please verify."
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
    
    def predict_from_prompts(
        self,
        image: np.ndarray,
        embeddings_path: Optional[Union[str, Path]] = None,
        boxes: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimasking: bool = False,
        mask_threshold: Optional[float] = None,
        tiled: bool = True,
        optimize_memory: bool = False
    ) -> np.ndarray:
        """
        Perform prompted instance segmentation.
        
        Args:
            image: RGB image array (H, W, 3)
            embeddings_path: Optional path to precomputed embeddings
            boxes: Bounding box prompts (N, 4) in format [min_x, min_y, max_x, max_y]
            points: Point prompts (N, 1, 2) in format [[[x, y]]]
            point_labels: Point labels (N, 1) where 1=positive, 0=negative
            multimasking: Whether to return multiple mask predictions
            mask_threshold: Threshold for mask binarization
            tiled: Whether to use tiled inference
            optimize_memory: Optimize memory usage (may be slower)
            
        Returns:
            Instance segmentation mask (H, W) with 0=background, 1..N=instance IDs
        """
        from micro_sam.inference import batched_inference
        
        # Clear MPS cache before prediction to prevent OOM on Apple Silicon
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache before prediction")
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")
        
        # Ensure RGB format
        image = self.ensure_rgb(image)
        
        # Note: batched_tiled_inference doesn't exist in v1.3.0
        # Always use batched_inference regardless of tiled parameter
        if tiled:
            logger.warning(
                "Tiled inference requested but not available in micro-sam v1.3.0. "
                "Using batched_inference instead (may be slower for large images)."
            )
        inference_fn = batched_inference
        
        # Prepare kwargs
        # Note: batched_inference in v1.3.0 does not support tile_shape and halo parameters
        kwargs = {
            "predictor": self.predictor,
            "image": image,
            "batch_size": self.batch_size,
            "return_instance_segmentation": True
        }
        
        if embeddings_path is not None:
            kwargs["embedding_path"] = str(embeddings_path)
        
        if boxes is not None:
            kwargs["boxes"] = boxes
        
        if points is not None:
            kwargs["points"] = points
        
        if point_labels is not None:
            kwargs["point_labels"] = point_labels
        
        if multimasking:
            kwargs["multimasking"] = multimasking
        
        if mask_threshold is not None:
            kwargs["mask_threshold"] = mask_threshold
        
        logger.info(f"Running prompted instance segmentation (tiled={tiled})")
        
        try:
            instance_mask = inference_fn(**kwargs)
            logger.info(f"Segmentation complete: {np.unique(instance_mask).size - 1} instances found")
            return instance_mask
        except Exception as e:
            logger.error(f"Failed to run prompted segmentation: {e}")
            raise
    
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
        
        IMPORTANT: This method requires python-elf to be installed, which is not available
        in Python 3.11 environments. Use prompt-based modes instead.
        
        Args:
            image: RGB image array (H, W, 3)
            embeddings_path: Optional path to precomputed embeddings
            segmentation_mode: Segmentation mode ("apg", "ais", or "amg")
                             - "apg": Automatic Instance Segmentation (Polygons) - recommended
                             - "ais": Automatic Instance Segmentation
                             - "amg": Automatic Mask Generation (SAM's original AMG)
            min_size: Minimum object size in pixels
            foreground_threshold: Threshold for foreground detection
            center_distance_threshold: Distance threshold for center detection
            boundary_distance_threshold: Distance threshold for boundary detection
            nms_threshold: Non-maximum suppression threshold
            multimasking: Whether to use multi-mask prediction
            tiled: Whether to use tiled inference
            
        Returns:
            Instance segmentation mask (H, W) with 0=background, 1..N=instance IDs
            
        Raises:
            RuntimeError: If elf is not available
        """
        # Check if elf is available
        if not _ELF_AVAILABLE:
            raise RuntimeError(
                "Automatic instance segmentation (APG/AIS) requires python-elf, which is not installed. "
                "python-elf cannot be installed in Python 3.11 due to numba/llvmlite constraints. "
                "Please use one of these alternatives:\n"
                "1. Use prompt-based modes: point, auto_box, or auto_box_from_threshold\n"
                "2. Create a conda environment with Python <3.10 and install python-elf from conda-forge\n"
                "Prompt-based modes provide excellent cell/nucleus segmentation without requiring elf."
            )
        
        from micro_sam.instance_segmentation import get_instance_segmentation_generator
        
        # Clear MPS cache before prediction to prevent OOM on Apple Silicon
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache before prediction")
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")
        
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
                output_mode="binary_mask"  # Return binary masks for each instance
            )
        except Exception as e:
            logger.error(f"Failed to create instance segmentation generator: {e}")
            raise
        
        # Prepare inference kwargs
        # Note: tile_shape and halo are not supported by the generator in v1.3.0
        kwargs = {
            "image": image,
        }
        
        if embeddings_path is not None:
            kwargs["embedding_path"] = str(embeddings_path)
        
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
                # Handle dict output if generator returns additional info
                instance_mask = instance_mask.get("segmentation", instance_mask)
            
            num_instances = np.unique(instance_mask).size - 1 if 0 in np.unique(instance_mask) else np.unique(instance_mask).size
            logger.info(f"Automatic segmentation complete: {num_instances} instances found")
            
            return instance_mask
        except Exception as e:
            logger.error(f"Failed to run automatic segmentation: {e}")
            raise


def segment_tissue(
    image: np.ndarray,
    method: str = "threshold",
    **kwargs
) -> np.ndarray:
    """
    Simple tissue segmentation utility for detecting tissue regions.
    
    Args:
        image: RGB image array (H, W, 3)
        method: Segmentation method ("threshold" or "otsu")
        **kwargs: Additional method-specific parameters
        
    Returns:
        Binary mask (H, W) with True for tissue regions
    """
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    
    # Convert to grayscale
    gray = rgb2gray(image)
    
    if method == "otsu":
        threshold = threshold_otsu(gray)
        mask = gray < threshold  # Dark regions are tissue
    else:  # threshold
        threshold = kwargs.get("threshold", 0.8)
        mask = gray < threshold
    
    return mask.astype(np.uint8) * 255


def is_elf_available() -> bool:
    """
    Check if python-elf is available at runtime.
    
    Returns:
        True if elf is available, False otherwise
    """
    return _ELF_AVAILABLE


def get_elf_info_message() -> str:
    """
    Get informational message about elf availability and alternatives.
    
    Returns:
        Formatted message string
    """
    if _ELF_AVAILABLE:
        return "✓ python-elf is available. All segmentation modes are supported."
    else:
        return (
            "⚠️ python-elf is not available. Automatic instance segmentation (APG/AIS) is disabled.\n\n"
            "**Available modes:**\n"
            "- point: Interactive point prompts\n"
            "- auto_box: Auto-detect tissue bounding box\n"
            "- auto_box_from_threshold: Generate boxes from thresholded channel (recommended for nuclei)\n"
            "- full_box: Use entire image\n\n"
            "**To enable automatic modes:**\n"
            "Create a conda environment with Python <3.10 and install:\n"
            "```\n"
            "conda create -n microsam-auto python=3.9\n"
            "conda activate microsam-auto\n"
            "conda install -c conda-forge python-elf\n"
            "```"
        )
