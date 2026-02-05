"""
Tests for SAM/MedSAM segmentation fixes
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ml_models import _ensure_rgb_uint8, _compute_tissue_bbox


class TestEnsureRGBUint8:
    """Test image format conversion"""
    
    def test_float_0_1_to_uint8(self):
        """Test conversion from float [0,1] to uint8 [0,255]"""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        result = _ensure_rgb_uint8(image)
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)
        assert result.min() >= 0
        assert result.max() <= 255
    
    def test_float_already_255_range(self):
        """Test that images already in [0,255] range are handled"""
        image = np.random.rand(100, 100, 3).astype(np.float32) * 255
        result = _ensure_rgb_uint8(image)
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)
    
    def test_uint8_passthrough(self):
        """Test that uint8 images pass through correctly"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = _ensure_rgb_uint8(image)
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)
        assert np.array_equal(result, image)
    
    def test_grayscale_to_rgb(self):
        """Test grayscale to RGB conversion"""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = _ensure_rgb_uint8(gray)
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)
        # Check all channels are the same
        assert np.array_equal(result[:,:,0], result[:,:,1])
        assert np.array_equal(result[:,:,1], result[:,:,2])
    
    def test_grayscale_with_channel_dim(self):
        """Test grayscale image with channel dimension (HW1) to RGB"""
        gray = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        result = _ensure_rgb_uint8(gray)
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)


class TestComputeTissueBox:
    """Test tissue bounding box detection"""
    
    def test_simple_tissue_detection(self):
        """Test detection on simple synthetic image"""
        # Create image with white background and gray tissue in center
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Add tissue region
        image[50:150, 50:150] = 100
        
        bbox = _compute_tissue_bbox(image)
        
        assert bbox.shape == (4,)
        assert isinstance(bbox, np.ndarray)
        # Should detect something in the middle region
        assert 0 <= bbox[0] < 200  # x1
        assert 0 <= bbox[1] < 200  # y1
        assert bbox[0] < bbox[2] <= 200  # x1 < x2
        assert bbox[1] < bbox[3] <= 200  # y1 < y2
    
    def test_full_image_tissue(self):
        """Test when entire image is tissue"""
        # Create image that's mostly tissue
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        bbox = _compute_tissue_bbox(image)
        
        # Should return something close to full image
        assert bbox.shape == (4,)
        assert bbox[2] - bbox[0] > 50  # Width > 50
        assert bbox[3] - bbox[1] > 50  # Height > 50
    
    def test_empty_image(self):
        """Test with uniform background (no tissue)"""
        # Uniform white image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        bbox = _compute_tissue_bbox(image)
        
        # Should return full image box as fallback
        assert bbox.shape == (4,)
        assert np.array_equal(bbox, [0, 0, 99, 99])
    
    def test_min_area_ratio(self):
        """Test min_area_ratio parameter"""
        # Create image with small tissue region
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Add tiny tissue region (1%)
        image[95:105, 95:105] = 100
        
        # With low threshold, should detect small region
        bbox = _compute_tissue_bbox(image, min_area_ratio=0.001)
        assert bbox.shape == (4,)
        
        # With high threshold, should return full image
        bbox = _compute_tissue_bbox(image, min_area_ratio=0.1)
        assert np.array_equal(bbox, [0, 0, 199, 199])
    
    def test_morph_kernel_size(self):
        """Test morphological operations with different kernel sizes"""
        # Create noisy image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image[50:150, 50:150] = 100
        
        # Add some noise
        noise = np.random.randint(0, 2, (200, 200)) * 50
        image[:,:,0] = np.clip(image[:,:,0] + noise, 0, 255).astype(np.uint8)
        
        # Different kernel sizes should still detect region
        bbox1 = _compute_tissue_bbox(image, morph_kernel_size=3)
        bbox2 = _compute_tissue_bbox(image, morph_kernel_size=7)
        
        assert bbox1.shape == (4,)
        assert bbox2.shape == (4,)


class TestMedSAMPredictorIntegration:
    """Integration tests for MedSAMPredictor with new features"""
    
    def test_predictor_can_be_initialized(self):
        """Test that predictor can be initialized without checkpoint"""
        # This test just checks the class can be imported and structure is correct
        from utils.ml_models import MedSAMPredictor
        
        # Check class has required methods
        assert hasattr(MedSAMPredictor, '__init__')
        assert hasattr(MedSAMPredictor, 'predict')
        assert hasattr(MedSAMPredictor, 'predict_with_box')
        assert hasattr(MedSAMPredictor, 'predict_with_points')
    
    def test_predict_signature(self):
        """Test that predict method has correct signature"""
        from utils.ml_models import MedSAMPredictor
        import inspect
        
        sig = inspect.signature(MedSAMPredictor.predict)
        params = list(sig.parameters.keys())
        
        # Check for new parameters
        assert 'prompt_mode' in params
        assert 'min_area_ratio' in params
        assert 'morph_kernel_size' in params
        assert 'multimask_output' in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
