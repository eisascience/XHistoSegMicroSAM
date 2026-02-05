"""
Comprehensive tests for GeoJSON utility functions
"""

import pytest
import numpy as np
import json
import tempfile
import os

from xhalo.utils.geojson_utils import (
    mask_to_contours,
    contour_to_polygon,
    mask_to_geojson,
    geojson_to_mask,
    convert_to_halo_annotations,
    save_geojson,
    load_geojson
)


class TestMaskToContours:
    """Test mask to contours conversion"""
    
    def test_simple_mask_to_contours(self):
        """Test converting a simple mask to contours"""
        # Create a mask with a square
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        contours = mask_to_contours(mask, min_area=10)
        
        assert len(contours) > 0
        # Check that we got contours
        assert all(isinstance(c, np.ndarray) for c in contours)
    
    def test_mask_with_multiple_regions(self):
        """Test mask with multiple disconnected regions"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:60, 20:60] = 1  # First square
        mask[120:160, 120:160] = 1  # Second square
        
        contours = mask_to_contours(mask, min_area=10)
        
        # Should have 2 contours
        assert len(contours) == 2
    
    def test_min_area_filter(self):
        """Test that small regions are filtered out"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:80, 20:80] = 1  # Large region (3600 pixels)
        mask[120:125, 120:125] = 1  # Small region (25 pixels)
        
        # With low threshold, should get both
        contours_low = mask_to_contours(mask, min_area=10)
        assert len(contours_low) == 2
        
        # With high threshold, should only get large region
        contours_high = mask_to_contours(mask, min_area=100)
        assert len(contours_high) == 1
    
    def test_empty_mask(self):
        """Test with empty mask"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        contours = mask_to_contours(mask, min_area=10)
        
        assert len(contours) == 0


class TestContourToPolygon:
    """Test contour to polygon conversion"""
    
    def test_valid_contour_to_polygon(self):
        """Test converting a valid contour to polygon"""
        # Create a simple triangular contour
        contour = np.array([[[10, 10]], [[50, 10]], [[30, 50]]], dtype=np.int32)
        
        polygon = contour_to_polygon(contour)
        
        assert polygon is not None
        assert polygon.is_valid
    
    def test_invalid_contour(self):
        """Test with too few points"""
        # Only 2 points - not enough for a polygon
        contour = np.array([[[10, 10]], [[50, 10]]], dtype=np.int32)
        
        polygon = contour_to_polygon(contour)
        
        assert polygon is None


class TestMaskToGeoJSON:
    """Test mask to GeoJSON conversion"""
    
    def test_mask_to_geojson_simple(self):
        """Test converting a simple mask to GeoJSON"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        geojson_data = mask_to_geojson(mask, min_area=10)
        
        assert "type" in geojson_data
        assert geojson_data["type"] == "FeatureCollection"
        assert "features" in geojson_data
        assert len(geojson_data["features"]) > 0
        
        # Check feature structure
        feature = geojson_data["features"][0]
        assert "type" in feature
        assert "geometry" in feature
        assert "properties" in feature
    
    def test_mask_to_geojson_with_properties(self):
        """Test adding custom properties to GeoJSON features"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        props = {"label": "tissue", "confidence": 0.95}
        geojson_data = mask_to_geojson(mask, min_area=10, properties=props)
        
        # Check that properties are included
        feature = geojson_data["features"][0]
        assert "label" in feature["properties"]
        assert feature["properties"]["label"] == "tissue"
    
    def test_mask_to_geojson_empty_mask(self):
        """Test with empty mask"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        geojson_data = mask_to_geojson(mask, min_area=10)
        
        assert geojson_data["type"] == "FeatureCollection"
        assert len(geojson_data["features"]) == 0


class TestGeoJSONToMask:
    """Test GeoJSON to mask conversion"""
    
    def test_geojson_to_mask_simple(self):
        """Test converting GeoJSON to mask"""
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
        assert mask.dtype == np.uint8
        assert np.sum(mask > 0) > 0  # Should have some non-zero pixels
    
    def test_geojson_to_mask_multiple_features(self):
        """Test with multiple features"""
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]]]
                    },
                    "properties": {}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[60, 60], [90, 60], [90, 90], [60, 90], [60, 60]]]
                    },
                    "properties": {}
                }
            ]
        }
        
        mask = geojson_to_mask(geojson_data, (100, 100))
        
        assert mask.shape == (100, 100)
        assert np.sum(mask > 0) > 0
    
    def test_geojson_to_mask_empty_features(self):
        """Test with no features"""
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        mask = geojson_to_mask(geojson_data, (100, 100))
        
        assert mask.shape == (100, 100)
        assert np.sum(mask) == 0  # Should be all zeros


class TestSaveLoadGeoJSON:
    """Test saving and loading GeoJSON files"""
    
    def test_save_geojson(self):
        """Test saving GeoJSON to file"""
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]]
                    },
                    "properties": {"id": 1}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            temp_path = f.name
        
        try:
            save_geojson(geojson_data, temp_path)
            
            # Verify file exists and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["type"] == "FeatureCollection"
            assert len(loaded["features"]) == 1
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_geojson(self):
        """Test loading GeoJSON from file"""
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]]
                    },
                    "properties": {"id": 1}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            json.dump(geojson_data, f)
            temp_path = f.name
        
        try:
            loaded = load_geojson(temp_path)
            
            assert loaded["type"] == "FeatureCollection"
            assert len(loaded["features"]) == 1
            assert loaded["features"][0]["properties"]["id"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_save_load_roundtrip(self):
        """Test that save and load are inverse operations"""
        original = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [50, 10], [30, 50], [10, 10]]]
                    },
                    "properties": {"label": "test", "value": 42}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as f:
            temp_path = f.name
        
        try:
            save_geojson(original, temp_path)
            loaded = load_geojson(temp_path)
            
            assert loaded["type"] == original["type"]
            assert len(loaded["features"]) == len(original["features"])
            assert loaded["features"][0]["properties"] == original["features"][0]["properties"]
        finally:
            os.unlink(temp_path)


class TestConvertToHaloAnnotations:
    """Test conversion to Halo annotation format"""
    
    def test_convert_to_halo_annotations(self):
        """Test converting mask to Halo-compatible annotations"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        annotations = convert_to_halo_annotations(mask, layer_name="Test Layer")
        
        assert isinstance(annotations, list)
        if len(annotations) > 0:
            # Check annotation structure
            assert isinstance(annotations[0], dict)
