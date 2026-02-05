#!/usr/bin/env python
"""
Standalone test for SAM helper functions
"""
import numpy as np
import cv2


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in HWC RGB uint8 format for SAM.
    """
    # Convert to numpy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Handle float [0,1] to uint8 [0,255]
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.5:  # Normalized [0,1]
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
    
    return image


def _compute_tissue_bbox(image: np.ndarray, 
                         min_area_ratio: float = 0.01,
                         morph_kernel_size: int = 5) -> np.ndarray:
    """
    Compute bounding box around tissue region using simple thresholding.
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
        if white_ratio < 0.5:
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
            return np.array([0, 0, w - 1, h - 1])
        
        # Find largest component (excluding background which is label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        # Check if area is significant
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
        if largest_area < (w * h * min_area_ratio):
            return np.array([0, 0, w - 1, h - 1])
        
        # Get bounding box
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        bbox_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        bbox_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        
        # Return as [x1, y1, x2, y2]
        bbox = np.array([x, y, x + bbox_w - 1, y + bbox_h - 1])
        
        return bbox
        
    except Exception as e:
        return np.array([0, 0, w - 1, h - 1])


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing SAM Helper Functions")
    print("=" * 60)
    
    # Test 1: _ensure_rgb_uint8 with float [0,1]
    print("\nTest 1: Float [0,1] to uint8")
    image = np.random.rand(100, 100, 3).astype(np.float32)
    result = _ensure_rgb_uint8(image)
    print(f"  Input shape: {image.shape}, dtype: {image.dtype}, range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}, range: [{result.min()}, {result.max()}]")
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    assert result.shape == (100, 100, 3), f"Expected (100, 100, 3), got {result.shape}"
    print("   PASSED")
    
    # Test 2: _ensure_rgb_uint8 with uint8
    print("\nTest 2: uint8 passthrough")
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result = _ensure_rgb_uint8(image)
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
    assert result.dtype == np.uint8
    assert np.array_equal(result, image)
    print("   PASSED")
    
    # Test 3: Grayscale to RGB
    print("\nTest 3: Grayscale to RGB")
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    result = _ensure_rgb_uint8(gray)
    print(f"  Input shape: {gray.shape}, Output shape: {result.shape}")
    assert result.shape == (100, 100, 3)
    assert np.array_equal(result[:,:,0], result[:,:,1])
    print("   PASSED")
    
    # Test 4: _compute_tissue_bbox
    print("\nTest 4: Tissue bbox detection")
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    image[50:150, 50:150] = 100  # Add tissue region
    bbox = _compute_tissue_bbox(image)
    print(f"  Detected bbox: {bbox}")
    assert bbox.shape == (4,), f"Expected shape (4,), got {bbox.shape}"
    assert bbox[0] < bbox[2], f"x1 should be < x2: {bbox}"
    assert bbox[1] < bbox[3], f"y1 should be < y2: {bbox}"
    # Check that detected box is reasonable
    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    print(f"  Box area: {box_area} pixels")
    assert box_area > 5000, f"Box area too small: {box_area}"
    print("   PASSED")
    
    # Test 5: Empty image (fallback)
    print("\nTest 5: Empty image fallback")
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = _compute_tissue_bbox(image)
    print(f"  Detected bbox: {bbox}")
    assert np.array_equal(bbox, [0, 0, 99, 99]), f"Expected [0, 0, 99, 99], got {bbox}"
    print("   PASSED")
    
    # Test 6: Large tissue region
    print("\nTest 6: Large tissue region")
    image = np.ones((300, 300, 3), dtype=np.uint8) * 100  # Most of image is tissue
    image[:20, :] = 255  # White border on top
    image[-20:, :] = 255  # White border on bottom
    bbox = _compute_tissue_bbox(image)
    print(f"  Detected bbox: {bbox}")
    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    img_area = 300 * 300
    print(f"  Box covers {100*box_area/img_area:.1f}% of image")
    assert box_area > 0.5 * img_area, "Should detect large tissue region"
    print("   PASSED")
    
    print("\n" + "=" * 60)
    print(" All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
