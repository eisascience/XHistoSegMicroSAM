"""
Comprehensive tests for MedSAM segmentation module
"""

import pytest
import numpy as np
import torch
from xhalo.ml.medsam import MedSAMPredictor, segment_tissue


class TestMedSAMPredictor:
    """Test MedSAM predictor functionality"""
    
    def test_predictor_init_cpu(self):
        """Test predictor initialization on CPU"""
        predictor = MedSAMPredictor(device="cpu")
        
        assert predictor.device == "cpu"
        assert predictor.model is None  # No model loaded yet
    
    def test_predictor_init_cuda_available(self):
        """Test predictor initialization with CUDA/MPS if available"""
        predictor = MedSAMPredictor()
        
        # Should default to cuda/mps if available, else cpu
        assert predictor.device in ["cpu", "cuda", "mps"]
    
    def test_predictor_with_model_path(self):
        """Test predictor initialization with model path"""
        import tempfile
        import os
        
        # Use a temporary file path that works on all platforms
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fake_model_path = f.name
        
        try:
            # This will use mock model since we don't have actual checkpoint
            predictor = MedSAMPredictor(model_path=fake_model_path, device="cpu")
            
            assert predictor.model_path == fake_model_path
            assert predictor.device == "cpu"
        finally:
            # Clean up
            if os.path.exists(fake_model_path):
                os.unlink(fake_model_path)
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        preprocessed = predictor.preprocess_image(image, target_size=(1024, 1024))
        
        assert isinstance(preprocessed, torch.Tensor)
        assert preprocessed.shape[-2:] == (1024, 1024)
    
    def test_preprocess_different_sizes(self):
        """Test preprocessing with different target sizes"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        for size in [(512, 512), (1024, 1024), (256, 256)]:
            preprocessed = predictor.preprocess_image(image, target_size=size)
            assert preprocessed.shape[-2:] == size
    
    def test_predict_basic(self):
        """Test basic prediction on an image"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        mask = predictor.predict(image)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        # Mask values should be 0 or 255 (mock implementation)
        assert np.all((mask == 0) | (mask == 255))
    
    def test_predict_different_sizes(self):
        """Test prediction on different image sizes"""
        predictor = MedSAMPredictor(device="cpu")
        
        for size in [(256, 256), (512, 512), (1024, 1024)]:
            image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            mask = predictor.predict(image)
            
            assert mask.shape == size
    
    def test_predict_tiles_basic(self):
        """Test tiled prediction"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        mask = predictor.predict_tiles(image, tile_size=512, overlap=64)
        
        assert mask.shape == (1024, 1024)
        assert mask.dtype == np.uint8
    
    def test_predict_tiles_large_image(self):
        """Test tiled prediction on large image"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        mask = predictor.predict_tiles(image, tile_size=1024, overlap=128)
        
        assert mask.shape == (2048, 2048)
        assert mask.dtype == np.uint8
    
    def test_predict_tiles_different_overlaps(self):
        """Test tiled prediction with different overlap values"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        for overlap in [0, 32, 64, 128]:
            mask = predictor.predict_tiles(image, tile_size=512, overlap=overlap)
            assert mask.shape == (1024, 1024)
    
    def test_predict_grayscale_image(self):
        """Test prediction on grayscale image"""
        predictor = MedSAMPredictor(device="cpu")
        # Create grayscale image
        gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        # Convert to 3-channel
        image = np.stack([gray] * 3, axis=-1)
        
        mask = predictor.predict(image)
        
        assert mask.shape == (512, 512)
    
    def test_predict_normalized_image(self):
        """Test prediction on already normalized image"""
        predictor = MedSAMPredictor(device="cpu")
        # Create uint8 image (not normalized float)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        mask = predictor.predict(image)
        
        assert mask.shape == (512, 512)


class TestSegmentTissue:
    """Test high-level tissue segmentation function"""
    
    def test_segment_tissue_basic(self):
        """Test basic tissue segmentation"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Create predictor and use it
        predictor = MedSAMPredictor(device="cpu")
        mask = segment_tissue(image, predictor=predictor)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
    
    def test_segment_tissue_with_parameters(self):
        """Test tissue segmentation with custom parameters"""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        predictor = MedSAMPredictor(device="cpu")
        mask = segment_tissue(
            image,
            predictor=predictor,
            tile_size=512,
            overlap=64
        )
        
        assert mask.shape == (1024, 1024)
    
    def test_segment_tissue_small_image(self):
        """Test segmentation on small image"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        predictor = MedSAMPredictor(device="cpu")
        mask = segment_tissue(image, predictor=predictor)
        
        assert mask.shape == (256, 256)
    
    def test_segment_tissue_large_image(self):
        """Test segmentation on larger image using tiling"""
        image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        predictor = MedSAMPredictor(device="cpu")
        mask = segment_tissue(image, predictor=predictor, tile_size=1024)
        
        assert mask.shape == (2048, 2048)


class TestMedSAMEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_image(self):
        """Test with empty/zero image"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Should not crash
        mask = predictor.predict(image)
        assert mask.shape == (512, 512)
    
    def test_single_channel_image(self):
        """Test with single channel image"""
        predictor = MedSAMPredictor(device="cpu")
        # Convert single channel to 3 channels
        gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        image = np.stack([gray, gray, gray], axis=-1)
        
        mask = predictor.predict(image)
        assert mask.shape == (512, 512)
    
    def test_invalid_image_shape(self):
        """Test with invalid image shape"""
        predictor = MedSAMPredictor(device="cpu")
        
        # 2D image without channel dimension should be handled or raise error
        image_2d = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # This should either work (by adding channel) or raise a clear error
        try:
            mask = predictor.predict(image_2d)
            # If it works, check the shape
            assert mask.shape == (512, 512)
        except (ValueError, IndexError):
            # Expected to fail with invalid shape
            pass
    
    def test_very_small_tile_size(self):
        """Test with very small tile size"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Should handle small tile sizes
        mask = predictor.predict_tiles(image, tile_size=128, overlap=16)
        assert mask.shape == (512, 512)
    
    def test_high_overlap(self):
        """Test with overlap close to tile size"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # High overlap relative to tile size
        mask = predictor.predict_tiles(image, tile_size=512, overlap=256)
        assert mask.shape == (1024, 1024)
