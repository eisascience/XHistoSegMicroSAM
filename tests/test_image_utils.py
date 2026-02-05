"""
Comprehensive tests for image utility functions
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from xhalo.utils.image_utils import (
    load_image,
    resize_image,
    extract_tiles,
    merge_tiles,
    apply_colormap,
    overlay_mask,
    normalize_image,
    denormalize_image
)


class TestLoadImage:
    """Test image loading functionality"""
    
    def test_load_valid_image(self):
        """Test loading a valid image file"""
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(f.name)
            temp_path = f.name
        
        try:
            loaded = load_image(temp_path)
            assert loaded.shape == (100, 100, 3)
            assert loaded.dtype == np.uint8
        finally:
            os.unlink(temp_path)
    
    def test_load_image_with_max_size(self):
        """Test loading image with size limit"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8))
            img.save(f.name)
            temp_path = f.name
        
        try:
            loaded = load_image(temp_path, max_size=500)
            assert max(loaded.shape[:2]) == 500
            assert loaded.shape[2] == 3
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_image(self):
        """Test loading non-existent image raises error"""
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.png")


class TestResizeImage:
    """Test image resizing functionality"""
    
    def test_resize_with_aspect_ratio(self):
        """Test resizing with aspect ratio maintained"""
        image = np.random.randint(0, 255, (1000, 500, 3), dtype=np.uint8)
        resized = resize_image(image, 400, keep_aspect_ratio=True)
        
        assert max(resized.shape[:2]) == 400
        assert resized.shape[2] == 3
        # Aspect ratio should be maintained (2:1)
        assert abs(resized.shape[0] / resized.shape[1] - 2.0) < 0.1
    
    def test_resize_without_aspect_ratio(self):
        """Test resizing to square without maintaining aspect ratio"""
        image = np.random.randint(0, 255, (1000, 500, 3), dtype=np.uint8)
        resized = resize_image(image, 400, keep_aspect_ratio=False)
        
        assert resized.shape == (400, 400, 3)
    
    def test_resize_small_dimension(self):
        """Test resizing with small max dimension"""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        resized = resize_image(image, 100)
        
        assert max(resized.shape[:2]) == 100


class TestExtractTiles:
    """Test tile extraction functionality"""
    
    def test_extract_tiles_no_overlap(self):
        """Test extracting tiles without overlap"""
        image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        tiles = extract_tiles(image, tile_size=512, overlap=0)
        
        assert len(tiles) == 16  # 4x4 tiles
        for tile, (x, y) in tiles:
            assert tile.shape[0] == 512
            assert tile.shape[1] == 512
            assert tile.shape[2] == 3
    
    def test_extract_tiles_with_overlap(self):
        """Test extracting tiles with overlap"""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        tiles = extract_tiles(image, tile_size=512, overlap=64)
        
        assert len(tiles) > 0
        for tile, (x, y) in tiles:
            assert tile.shape[0] <= 512
            assert tile.shape[1] <= 512
    
    def test_extract_tiles_edge_cases(self):
        """Test tile extraction with non-divisible dimensions"""
        image = np.random.randint(0, 255, (1500, 1200, 3), dtype=np.uint8)
        tiles = extract_tiles(image, tile_size=512, overlap=0)
        
        assert len(tiles) > 0
        # Check that all tiles have correct dimensions
        for tile, (x, y) in tiles:
            assert tile.ndim == 3
            assert tile.shape[2] == 3


class TestMergeTiles:
    """Test tile merging functionality"""
    
    def test_merge_tiles_basic(self):
        """Test basic tile merging"""
        original_shape = (1024, 1024)
        tiles = []
        
        # Create some tiles
        tile_size = 512
        for y in range(0, original_shape[0], tile_size):
            for x in range(0, original_shape[1], tile_size):
                tile = np.random.randint(0, 2, (tile_size, tile_size), dtype=np.uint8)
                tiles.append((tile, (x, y)))
        
        merged = merge_tiles(tiles, original_shape)
        
        assert merged.shape == original_shape
        assert merged.dtype == np.uint8


class TestNormalizeDenormalize:
    """Test normalization and denormalization functions"""
    
    def test_normalize_image(self):
        """Test image normalization"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = normalize_image(image)
        
        assert normalized.dtype in [np.float32, np.float64]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_denormalize_image(self):
        """Test image denormalization"""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        denormalized = denormalize_image(image)
        
        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255
    
    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize and denormalize are inverse operations"""
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = normalize_image(original)
        restored = denormalize_image(normalized)
        
        # Should be approximately equal (within 1 due to rounding)
        assert np.allclose(original, restored, atol=1)


class TestOverlayMask:
    """Test mask overlay functionality"""
    
    def test_overlay_mask_basic(self):
        """Test basic mask overlay"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        
        overlay = overlay_mask(image, mask, alpha=0.5)
        
        assert overlay.shape == image.shape
        assert overlay.dtype == np.uint8
    
    def test_overlay_mask_different_alphas(self):
        """Test mask overlay with different alpha values"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        
        # Test with different alpha values
        for alpha in [0.0, 0.3, 0.7, 1.0]:
            overlay = overlay_mask(image, mask, alpha=alpha)
            assert overlay.shape == image.shape
    
    def test_overlay_mask_empty_mask(self):
        """Test overlay with empty mask"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        overlay = overlay_mask(image, mask, alpha=0.5)
        
        # With empty mask, output should be similar to input
        assert overlay.shape == image.shape


class TestApplyColormap:
    """Test colormap application"""
    
    def test_apply_colormap(self):
        """Test applying colormap to grayscale image"""
        grayscale = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        colored = apply_colormap(grayscale)
        
        assert colored.ndim == 3
        assert colored.shape[:2] == grayscale.shape
        assert colored.shape[2] == 3
