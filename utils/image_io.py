"""
Image I/O Utilities for Multi-Channel Support

Provides robust loading of various image formats including multi-channel TIFFs.
"""

import numpy as np
from PIL import Image
import tifffile
import io
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_image_any(uploaded_file) -> Dict:
    """
    Load image from uploaded file with robust multi-channel detection.
    
    Args:
        uploaded_file: Streamlit UploadedFile or bytes
        
    Returns:
        Dictionary with keys:
        - kind: "rgb" | "grayscale" | "multichannel"
        - data: np.ndarray (original data)
        - channels: np.ndarray shaped (C,H,W) for multichannel, None otherwise
        - rgb: np.ndarray shaped (H,W,3) for rgb images, None otherwise
        - grayscale: np.ndarray shaped (H,W) for grayscale images, None otherwise
        - channel_names: list[str] like ["Ch 0", "Ch 1", ...]
        - dtype: str (original dtype)
        - vmin: float (minimum value)
        - vmax: float (maximum value)
        - shape_original: tuple (original shape before processing)
    """
    # Get bytes from uploaded file
    if hasattr(uploaded_file, 'read'):
        image_bytes = uploaded_file.read()
        filename = getattr(uploaded_file, 'name', 'unknown')
        uploaded_file.seek(0)  # Reset file pointer
    else:
        image_bytes = uploaded_file
        filename = 'unknown'
    
    # Check if it's a TIFF file
    is_tiff = filename.lower().endswith(('.tif', '.tiff'))
    
    if is_tiff:
        return _load_tiff(image_bytes, filename)
    else:
        return _load_standard_image(image_bytes, filename)


def _load_tiff(image_bytes: bytes, filename: str) -> Dict:
    """Load TIFF file with multi-channel support using tifffile."""
    try:
        # Use tifffile to read TIFF
        data = tifffile.imread(io.BytesIO(image_bytes))
        logger.info(f"TIFF loaded: {filename}, shape={data.shape}, dtype={data.dtype}")
        
        # Squeeze singleton dimensions
        data = np.squeeze(data)
        shape_original = data.shape
        dtype_str = str(data.dtype)
        vmin, vmax = float(data.min()), float(data.max())
        
        # Determine image kind based on shape
        if data.ndim == 2:
            # Grayscale image (H, W)
            return {
                'kind': 'grayscale',
                'data': data,
                'channels': None,
                'rgb': None,
                'grayscale': data,
                'channel_names': ['Gray'],
                'dtype': dtype_str,
                'vmin': vmin,
                'vmax': vmax,
                'shape_original': shape_original
            }
        
        elif data.ndim == 3:
            h, w = data.shape[0], data.shape[1]
            third_dim = data.shape[2]
            
            # Check if it's RGB (H, W, 3)
            if third_dim == 3 and h > 16 and w > 16:
                logger.info(f"Detected as RGB image (H,W,3): {data.shape}")
                return {
                    'kind': 'rgb',
                    'data': data,
                    'channels': None,
                    'rgb': data,
                    'grayscale': None,
                    'channel_names': ['R', 'G', 'B'],
                    'dtype': dtype_str,
                    'vmin': vmin,
                    'vmax': vmax,
                    'shape_original': shape_original
                }
            
            # Check if it's channels-first (C, H, W)
            elif data.shape[0] <= 16 and data.shape[1] > 16 and data.shape[2] > 16:
                # Channels-first: (C, H, W)
                n_channels = data.shape[0]
                logger.info(f"Detected as channels-first (C,H,W): {data.shape}")
                channel_names = [f"Ch {i}" for i in range(n_channels)]
                return {
                    'kind': 'multichannel',
                    'data': data,
                    'channels': data,  # Already in (C, H, W) format
                    'rgb': None,
                    'grayscale': None,
                    'channel_names': channel_names,
                    'dtype': dtype_str,
                    'vmin': vmin,
                    'vmax': vmax,
                    'shape_original': shape_original
                }
            
            # Check if it's channels-last (H, W, C)
            elif data.shape[2] <= 16 and data.shape[0] > 16 and data.shape[1] > 16:
                # Channels-last: (H, W, C) -> transpose to (C, H, W)
                n_channels = data.shape[2]
                logger.info(f"Detected as channels-last (H,W,C): {data.shape}, transposing to (C,H,W)")
                data_chw = np.transpose(data, (2, 0, 1))
                channel_names = [f"Ch {i}" for i in range(n_channels)]
                return {
                    'kind': 'multichannel',
                    'data': data,
                    'channels': data_chw,  # Transposed to (C, H, W)
                    'rgb': None,
                    'grayscale': None,
                    'channel_names': channel_names,
                    'dtype': dtype_str,
                    'vmin': vmin,
                    'vmax': vmax,
                    'shape_original': shape_original
                }
            
            else:
                # Ambiguous case - assume RGB if third_dim == 3, otherwise channels-first
                if third_dim == 3:
                    logger.warning(f"Ambiguous shape {data.shape}, assuming RGB (H,W,3)")
                    return {
                        'kind': 'rgb',
                        'data': data,
                        'channels': None,
                        'rgb': data,
                        'grayscale': None,
                        'channel_names': ['R', 'G', 'B'],
                        'dtype': dtype_str,
                        'vmin': vmin,
                        'vmax': vmax,
                        'shape_original': shape_original
                    }
                else:
                    logger.warning(f"Ambiguous shape {data.shape}, assuming channels-first (C,H,W)")
                    n_channels = data.shape[0]
                    channel_names = [f"Ch {i}" for i in range(n_channels)]
                    return {
                        'kind': 'multichannel',
                        'data': data,
                        'channels': data,
                        'rgb': None,
                        'grayscale': None,
                        'channel_names': channel_names,
                        'dtype': dtype_str,
                        'vmin': vmin,
                        'vmax': vmax,
                        'shape_original': shape_original
                    }
        
        elif data.ndim > 3:
            # Handle higher-dimensional data (e.g., Z-stacks)
            # For now, try to reduce to 2D channels by taking a middle slice
            logger.warning(f"Higher-dimensional data detected: {data.shape}")
            
            # Try common patterns: (Z, C, H, W) or (C, Z, H, W)
            if data.ndim == 4:
                # Assume (Z, C, H, W) and take middle Z slice
                z_mid = data.shape[0] // 2
                data_2d = data[z_mid, :, :, :]
                logger.info(f"Taking Z-slice {z_mid} from (Z,C,H,W), resulting shape: {data_2d.shape}")
                # Recursively call with 3D data
                # Create a temporary bytes object
                return _process_3d_as_multichannel(data_2d, dtype_str, shape_original)
            
            raise ValueError(f"Unsupported TIFF shape: {data.shape}. Expected 2D, 3D, or 4D (with Z-stack).")
        
        else:
            raise ValueError(f"Unsupported TIFF dimensionality: {data.ndim}D")
    
    except Exception as e:
        logger.error(f"Failed to load TIFF {filename}: {str(e)}")
        raise


def _process_3d_as_multichannel(data: np.ndarray, dtype_str: str, shape_original: tuple) -> Dict:
    """Process 3D array as multichannel data."""
    vmin, vmax = float(data.min()), float(data.max())
    
    # Apply same logic as 3D case in _load_tiff
    if data.shape[2] == 3 and data.shape[0] > 16 and data.shape[1] > 16:
        # RGB
        return {
            'kind': 'rgb',
            'data': data,
            'channels': None,
            'rgb': data,
            'grayscale': None,
            'channel_names': ['R', 'G', 'B'],
            'dtype': dtype_str,
            'vmin': vmin,
            'vmax': vmax,
            'shape_original': shape_original
        }
    elif data.shape[0] <= 16:
        # Channels-first (C, H, W)
        n_channels = data.shape[0]
        channel_names = [f"Ch {i}" for i in range(n_channels)]
        return {
            'kind': 'multichannel',
            'data': data,
            'channels': data,
            'rgb': None,
            'grayscale': None,
            'channel_names': channel_names,
            'dtype': dtype_str,
            'vmin': vmin,
            'vmax': vmax,
            'shape_original': shape_original
        }
    else:
        # Channels-last (H, W, C)
        n_channels = data.shape[2]
        data_chw = np.transpose(data, (2, 0, 1))
        channel_names = [f"Ch {i}" for i in range(n_channels)]
        return {
            'kind': 'multichannel',
            'data': data,
            'channels': data_chw,
            'rgb': None,
            'grayscale': None,
            'channel_names': channel_names,
            'dtype': dtype_str,
            'vmin': vmin,
            'vmax': vmax,
            'shape_original': shape_original
        }


def _load_standard_image(image_bytes: bytes, filename: str) -> Dict:
    """Load standard image formats (JPG, PNG) using PIL."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        if img.mode == 'RGB':
            data = np.array(img)
            logger.info(f"Loaded RGB image: {filename}, shape={data.shape}")
            return {
                'kind': 'rgb',
                'data': data,
                'channels': None,
                'rgb': data,
                'grayscale': None,
                'channel_names': ['R', 'G', 'B'],
                'dtype': str(data.dtype),
                'vmin': float(data.min()),
                'vmax': float(data.max()),
                'shape_original': data.shape
            }
        
        elif img.mode in ['L', 'I', 'F']:
            # Grayscale
            data = np.array(img)
            logger.info(f"Loaded grayscale image: {filename}, shape={data.shape}")
            return {
                'kind': 'grayscale',
                'data': data,
                'channels': None,
                'rgb': None,
                'grayscale': data,
                'channel_names': ['Gray'],
                'dtype': str(data.dtype),
                'vmin': float(data.min()),
                'vmax': float(data.max()),
                'shape_original': data.shape
            }
        
        else:
            # Convert other modes to RGB
            img_rgb = img.convert('RGB')
            data = np.array(img_rgb)
            logger.info(f"Loaded image (converted to RGB): {filename}, shape={data.shape}")
            return {
                'kind': 'rgb',
                'data': data,
                'channels': None,
                'rgb': data,
                'grayscale': None,
                'channel_names': ['R', 'G', 'B'],
                'dtype': str(data.dtype),
                'vmin': float(data.min()),
                'vmax': float(data.max()),
                'shape_original': data.shape
            }
    
    except Exception as e:
        logger.error(f"Failed to load image {filename}: {str(e)}")
        raise


def normalize_to_uint8(data: np.ndarray, percentile_clip: tuple = (1, 99)) -> np.ndarray:
    """
    Normalize array to uint8 [0, 255] range with percentile clipping.
    
    Args:
        data: Input array (any dtype)
        percentile_clip: Tuple of (low, high) percentiles for clipping
        
    Returns:
        Normalized uint8 array
    """
    # Compute percentiles
    p_low, p_high = np.percentile(data, percentile_clip)
    
    # Clip
    data_clipped = np.clip(data, p_low, p_high)
    
    # Scale to [0, 255]
    if p_high > p_low:
        data_scaled = (data_clipped - p_low) / (p_high - p_low) * 255.0
    else:
        data_scaled = np.zeros_like(data_clipped)
    
    return data_scaled.astype(np.uint8)


def apply_threshold(data: np.ndarray, threshold: float, mode: str = 'binary') -> np.ndarray:
    """
    Apply threshold to data.
    
    Args:
        data: Input array
        threshold: Threshold value
        mode: 'binary' (return mask) or 'masked' (return masked intensity)
        
    Returns:
        Thresholded array
    """
    if mode == 'binary':
        return (data >= threshold).astype(np.uint8) * 255
    elif mode == 'masked':
        result = data.copy()
        result[data < threshold] = 0
        return result
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")


def apply_gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to data.
    
    Args:
        data: Input array
        sigma: Gaussian sigma parameter
        
    Returns:
        Smoothed array
    """
    import cv2
    if sigma <= 0:
        return data
    
    # Convert to float for smoothing
    data_float = data.astype(np.float32)
    
    # Gaussian blur
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # 6*sigma + 1, always odd
    smoothed = cv2.GaussianBlur(data_float, (kernel_size, kernel_size), sigma)
    
    return smoothed.astype(data.dtype)
