#!/usr/bin/env python3
"""
Test script to verify tile_shape parameter fix.
Tests that the batched_inference calls work without tile_shape and halo parameters.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.microsam_adapter import MicroSAMPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_prediction():
    """Test basic prediction with auto_box mode."""
    logger.info("=" * 60)
    logger.info("Testing basic prediction with auto_box mode")
    logger.info("=" * 60)
    
    # Create a simple test image with a white circle on black background
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    mask = (x - 256)**2 + (y - 256)**2 <= 100**2
    test_img[mask] = 255
    
    logger.info(f"Created test image: shape={test_img.shape}, dtype={test_img.dtype}")
    logger.info(f"Test image stats: min={test_img.min()}, max={test_img.max()}, mean={test_img.mean():.1f}")
    
    try:
        # Initialize predictor
        logger.info("Initializing MicroSAMPredictor...")
        predictor = MicroSAMPredictor(model_type="vit_b_histopathology")
        
        # Run prediction with auto_box mode
        logger.info("Running prediction with prompt_mode='auto_box'...")
        result = predictor.predict(test_img, prompt_mode="auto_box")
        
        # Check results
        unique_vals = np.unique(result)
        result_sum = result.sum()
        
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Result shape: {result.shape}")
        logger.info(f"Result dtype: {result.dtype}")
        logger.info(f"Unique values: {unique_vals}")
        logger.info(f"Sum: {result_sum}")
        logger.info(f"Non-zero pixels: {np.sum(result > 0)}")
        
        # Verify we got a non-empty result
        if result_sum > 0:
            logger.info("✓ SUCCESS: Got non-zero segmentation mask")
            return True
        else:
            logger.error("✗ FAILED: Segmentation mask is all zeros")
            return False
            
    except TypeError as e:
        if "unexpected keyword argument 'tile_shape'" in str(e):
            logger.error("✗ FAILED: Still getting tile_shape error!")
            logger.error(f"Error: {e}")
            return False
        else:
            raise
    except Exception as e:
        logger.error(f"✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_point_mode():
    """Test prediction with point mode."""
    logger.info("=" * 60)
    logger.info("Testing prediction with point mode")
    logger.info("=" * 60)
    
    # Create a simple test image
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    mask = (x - 256)**2 + (y - 256)**2 <= 100**2
    test_img[mask] = 255
    
    try:
        predictor = MicroSAMPredictor(model_type="vit_b_histopathology")
        
        # Use center point as prompt
        point_coords = np.array([[256, 256]])
        point_labels = np.array([1])
        
        logger.info("Running prediction with prompt_mode='point'...")
        result = predictor.predict(
            test_img, 
            prompt_mode="point",
            point_coords=point_coords,
            point_labels=point_labels
        )
        
        result_sum = result.sum()
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Sum: {result_sum}")
        logger.info(f"Non-zero pixels: {np.sum(result > 0)}")
        
        if result_sum > 0:
            logger.info("✓ SUCCESS: Got non-zero segmentation mask with point mode")
            return True
        else:
            logger.error("✗ FAILED: Segmentation mask is all zeros")
            return False
            
    except TypeError as e:
        if "unexpected keyword argument 'tile_shape'" in str(e):
            logger.error("✗ FAILED: Still getting tile_shape error!")
            logger.error(f"Error: {e}")
            return False
        else:
            raise
    except Exception as e:
        logger.error(f"✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_box_mode():
    """Test prediction with full_box mode."""
    logger.info("=" * 60)
    logger.info("Testing prediction with full_box mode")
    logger.info("=" * 60)
    
    # Create a simple test image
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    mask = (x - 256)**2 + (y - 256)**2 <= 100**2
    test_img[mask] = 255
    
    try:
        predictor = MicroSAMPredictor(model_type="vit_b_histopathology")
        
        logger.info("Running prediction with prompt_mode='full_box'...")
        result = predictor.predict(test_img, prompt_mode="full_box")
        
        result_sum = result.sum()
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Sum: {result_sum}")
        logger.info(f"Non-zero pixels: {np.sum(result > 0)}")
        
        if result_sum > 0:
            logger.info("✓ SUCCESS: Got non-zero segmentation mask with full_box mode")
            return True
        else:
            logger.error("✗ FAILED: Segmentation mask is all zeros")
            return False
            
    except TypeError as e:
        if "unexpected keyword argument 'tile_shape'" in str(e):
            logger.error("✗ FAILED: Still getting tile_shape error!")
            logger.error(f"Error: {e}")
            return False
        else:
            raise
    except Exception as e:
        logger.error(f"✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("Starting tile_shape parameter fix tests...")
    logger.info("")
    
    results = []
    
    # Test 1: auto_box mode
    results.append(("auto_box mode", test_basic_prediction()))
    logger.info("")
    
    # Test 2: point mode
    results.append(("point mode", test_point_mode()))
    logger.info("")
    
    # Test 3: full_box mode
    results.append(("full_box mode", test_full_box_mode()))
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        logger.info("")
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SOME TESTS FAILED! ✗")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
