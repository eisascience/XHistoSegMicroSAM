"""
Image Processing Utilities

Functions for loading, preprocessing, and postprocessing images
for ML model inference and visualization.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Dict
import io
import logging

logger = logging.getLogger(__name__)


def load_image_from_bytes(image_data: bytes) -> np.ndarray:
    """
    Load image data from bytes into numpy array.
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary (handles BGR, grayscale, etc.)
        # PIL.Image.open always returns RGB, not BGR like cv2.imread
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        logger.info(f"Loaded image: shape={img_array.shape}, dtype={img_array.dtype}, "
                   f"range=[{img_array.min()}, {img_array.max()}]")
        return img_array
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise


def resize_image(image: np.ndarray, target_size: int, 
                 maintain_aspect: bool = True) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image to target size.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target dimension (max side length)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized image, original shape)
    """
    original_shape = image.shape[:2]
    h, w = original_shape
    
    if maintain_aspect:
        scale = target_size / max(h, w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_resized = image
    else:
        image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return image_resized, original_shape


def pad_to_square(image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image to square dimensions.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target square size
        
    Returns:
        Tuple of (padded image, padding amounts (top, bottom, left, right))
    """
    h, w = image.shape[:2]
    
    # Handle case where image is already larger than target
    if h > target_size or w > target_size:
        logger.warning(f"Image {h}x{w} larger than target {target_size}. Consider resizing first.")
        return image, (0, 0, 0, 0)
    
    pad_h = target_size - h
    pad_w = target_size - w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
        padded[pad_top:pad_top+h, pad_left:pad_left+w, :] = image
    else:
        padded = np.zeros((target_size, target_size), dtype=image.dtype)
        padded[pad_top:pad_top+h, pad_left:pad_left+w] = image
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def preprocess_for_medsam(image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess image for MedSAM inference.
    
    Steps:
    1. Resize to target size (maintaining aspect ratio)
    2. Pad to square
    3. Normalize (if needed)
    
    Args:
        image: Input RGB image (H, W, 3)
        target_size: Target size for model input
        
    Returns:
        Tuple of (preprocessed image, preprocessing metadata)
    """
    original_shape = image.shape[:2]
    
    # Resize
    resized, _ = resize_image(image, target_size, maintain_aspect=True)
    
    # Pad to square
    padded, padding = pad_to_square(resized, target_size)
    
    # Store preprocessing metadata
    metadata = {
        'original_shape': original_shape,
        'resized_shape': resized.shape[:2],
        'padding': padding,
        'target_size': target_size
    }
    
    logger.info(f"Preprocessed image: {original_shape} -> {padded.shape}")
    
    return padded, metadata


def postprocess_mask(mask: np.ndarray, metadata: Dict) -> np.ndarray:
    """
    Postprocess segmentation mask back to original dimensions.
    
    Args:
        mask: Binary mask from model (H, W)
        metadata: Preprocessing metadata from preprocess_for_medsam
        
    Returns:
        Mask resized to original image dimensions
    """
    # Remove padding
    pad_top, pad_bottom, pad_left, pad_right = metadata['padding']
    target_size = metadata['target_size']
    resized_shape = metadata['resized_shape']
    
    # Crop out padding
    h, w = resized_shape
    mask_cropped = mask[pad_top:pad_top+h, pad_left:pad_left+w]
    
    # Resize back to original
    original_shape = metadata['original_shape']
    mask_resized = cv2.resize(
        mask_cropped.astype(np.uint8),
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    logger.info(f"Postprocessed mask: {mask.shape} -> {mask_resized.shape}")
    
    return mask_resized > 0


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255].
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image
    """
    return (image * 255).astype(np.uint8)


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, 
                         color: Tuple[int, int, int] = (255, 0, 0),
                         alpha: float = 0.5) -> np.ndarray:
    """
    Overlay binary mask on image with transparency.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        color: Overlay color (R, G, B)
        alpha: Transparency (0 = transparent, 1 = opaque)
        
    Returns:
        Image with mask overlay
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result


def compute_mask_statistics(mask: np.ndarray, mpp: Optional[float] = None) -> Dict:
    """
    Compute statistics for segmentation mask.
    
    Args:
        mask: Binary mask (H, W)
        mpp: Microns per pixel (optional, for area calculation)
        
    Returns:
        Dictionary with statistics
    """
    num_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    coverage = num_pixels / total_pixels if total_pixels > 0 else 0
    
    stats = {
        'num_positive_pixels': int(num_pixels),
        'total_pixels': int(total_pixels),
        'coverage_percent': float(coverage * 100)
    }
    
    if mpp is not None:
        area_um2 = num_pixels * (mpp ** 2)
        stats['area_um2'] = float(area_um2)
        stats['area_mm2'] = float(area_um2 / 1e6)
    
    return stats
