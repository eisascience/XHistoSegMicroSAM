"""
Image processing utilities for handling large pathology images
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str, max_size: Optional[int] = None) -> np.ndarray:
    """
    Load image from file in RGB format
    
    Args:
        image_path: Path to image file
        max_size: Optional maximum dimension size
        
    Returns:
        RGB image as numpy array (H, W, C) in [0, 255] range
    """
    try:
        image = Image.open(image_path)
        # Convert to RGB - PIL returns RGB, not BGR like cv2.imread
        image = np.array(image.convert('RGB'))
        
        if max_size and max(image.shape[:2]) > max_size:
            image = resize_image(image, max_size)
        
        logger.info(f"Loaded image from {image_path}: shape={image.shape}, dtype={image.dtype}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def resize_image(
    image: np.ndarray, 
    max_dimension: int,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image while optionally maintaining aspect ratio
    
    Args:
        image: Input image
        max_dimension: Maximum size for longest dimension
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if keep_aspect_ratio:
        scale = max_dimension / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h = new_w = max_dimension
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 0
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Extract tiles from large image
    
    Args:
        image: Input image
        tile_size: Size of tiles
        overlap: Overlap between tiles
        
    Returns:
        List of (tile, (x, y)) tuples
    """
    tiles = []
    h, w = image.shape[:2]
    stride = tile_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile = image[y:y_end, x:x_end]
            tiles.append((tile, (x, y)))
    
    return tiles


def merge_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Merge tiles back into full image
    
    Args:
        tiles: List of (tile, (x, y)) tuples
        output_shape: Shape of output image (H, W)
        
    Returns:
        Merged image
    """
    h, w = output_shape
    merged = np.zeros((h, w), dtype=np.uint8)
    
    for tile, (x, y) in tiles:
        tile_h, tile_w = tile.shape[:2]
        merged[y:y+tile_h, x:x+tile_w] = tile
    
    return merged


def apply_colormap(mask: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply colormap to grayscale mask for visualization
    
    Args:
        mask: Binary or grayscale mask
        colormap: OpenCV colormap constant
        
    Returns:
        Colored image (H, W, 3)
    """
    # Normalize to 0-255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    colored = cv2.applyColorMap(mask, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Overlay segmentation mask on image
    
    Args:
        image: Original image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Transparency of overlay
        color: RGB color for mask
        
    Returns:
        Image with overlay
    """
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Resize mask if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = image.copy()
    overlay[mask > 0] = color
    
    # Blend images
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255]
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    return image.astype(np.uint8)
