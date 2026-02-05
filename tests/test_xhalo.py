"""
Unit tests for XHalo Path Analyzer
"""

import pytest
import numpy as np
from xhalo.ml import MedSAMPredictor
from xhalo.utils import (
    resize_image,
    extract_tiles,
    mask_to_geojson,
    geojson_to_mask,
    overlay_mask
)


class TestImageUtils:
    """Test image utility functions"""
    
    def test_resize_image(self):
        """Test image resizing"""
        image = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        resized = resize_image(image, 500)
        
        assert max(resized.shape[:2]) == 500
        assert resized.shape[2] == 3
    
    def test_extract_tiles(self):
        """Test tile extraction"""
        image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        tiles = extract_tiles(image, tile_size=512, overlap=64)
        
        assert len(tiles) > 0
        for tile, (x, y) in tiles:
            assert tile.shape[0] <= 512
            assert tile.shape[1] <= 512
            assert tile.shape[2] == 3
    
    def test_overlay_mask(self):
        """Test mask overlay on image"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        
        overlay = overlay_mask(image, mask, alpha=0.5)
        
        assert overlay.shape == image.shape
        assert overlay.dtype == np.uint8


class TestGeoJSONUtils:
    """Test GeoJSON utility functions"""
    
    def test_mask_to_geojson(self):
        """Test mask to GeoJSON conversion"""
        # Create a simple mask with a square
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        geojson_data = mask_to_geojson(mask, min_area=10)
        
        assert "features" in geojson_data
        assert len(geojson_data["features"]) > 0
        
        feature = geojson_data["features"][0]
        assert "geometry" in feature
        assert "properties" in feature
    
    def test_geojson_to_mask(self):
        """Test GeoJSON to mask conversion"""
        # Create simple GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]]
                    },
                    "properties": {}
                }
            ]
        }
        
        mask = geojson_to_mask(geojson_data, (100, 100))
        
        assert mask.shape == (100, 100)
        assert np.sum(mask > 0) > 0


class TestMedSAM:
    """Test MedSAM predictor"""
    
    def test_predictor_init(self):
        """Test predictor initialization"""
        predictor = MedSAMPredictor(device="cpu")
        assert predictor.device == "cpu"
    
    def test_predict(self):
        """Test prediction on sample image"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        mask = predictor.predict(image)
        
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
    
    def test_predict_tiles(self):
        """Test tiled prediction"""
        predictor = MedSAMPredictor(device="cpu")
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        mask = predictor.predict_tiles(image, tile_size=512, overlap=64)
        
        assert mask.shape == (1024, 1024)
        assert mask.dtype == np.uint8


class TestHaloAPI:
    """Test Halo API client"""
    
    @pytest.mark.asyncio
    async def test_mock_client(self):
        """Test mock API client"""
        from xhalo.api import MockHaloAPIClient
        
        client = MockHaloAPIClient()
        
        # Test list slides
        slides = await client.list_slides()
        assert len(slides) > 0
        assert "id" in slides[0]
        assert "name" in slides[0]
        
        # Test get slide info
        slide_info = await client.get_slide_info(slides[0]["id"])
        assert slide_info is not None
        assert "width" in slide_info
        assert "height" in slide_info
        
        # Test list ROIs
        rois = await client.list_rois(slides[0]["id"])
        assert isinstance(rois, list)
        
        # Test import annotations
        annotations = [{"type": "test", "geometry": {}}]
        success = await client.import_annotations(
            slides[0]["id"],
            annotations,
            "Test Layer"
        )
        assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
