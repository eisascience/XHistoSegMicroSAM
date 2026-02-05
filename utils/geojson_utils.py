"""
GeoJSON Utilities

Functions for converting segmentation masks to GeoJSON format
for import into Halo digital pathology platform.
"""

import numpy as np
from skimage import measure
from skimage.measure import approximate_polygon
import cv2
from typing import List, Dict, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def mask_to_polygons(mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """
    Convert binary mask to list of polygon contours.
    
    Uses skimage.measure.find_contours to extract polygon boundaries
    from the mask, filtering out small objects.
    
    Args:
        mask: Binary mask (H, W) with True/1 for object pixels
        min_area: Minimum polygon area in pixels (smaller polygons filtered out)
        
    Returns:
        List of polygon contours as numpy arrays (N, 2) with (row, col) coordinates
    """
    try:
        # Find contours at 0.5 threshold
        contours = measure.find_contours(mask.astype(float), 0.5)
        
        # Filter by area
        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Need at least 3 points for polygon
                area = polygon_area(contour)
                if area >= min_area:
                    polygons.append(contour)
        
        logger.info(f"Extracted {len(polygons)} polygons from mask")
        return polygons
        
    except Exception as e:
        logger.error(f"Failed to extract polygons: {str(e)}")
        raise


def polygon_area(polygon: np.ndarray) -> float:
    """
    Calculate polygon area using shoelace formula.
    
    Args:
        polygon: Polygon vertices as (N, 2) array with (row, col) coordinates
        
    Returns:
        Polygon area in pixels
    """
    x = polygon[:, 1]
    y = polygon[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def simplify_polygon(polygon: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """
    Simplify polygon using Douglas-Peucker algorithm.
    
    Reduces the number of vertices while maintaining shape.
    
    Args:
        polygon: Polygon vertices (N, 2)
        tolerance: Simplification tolerance (larger = more simplified)
        
    Returns:
        Simplified polygon
    """
    try:
        simplified = approximate_polygon(polygon, tolerance=tolerance)
        return simplified
    except ImportError:
        logger.warning("skimage not available for polygon simplification")
        return polygon


def polygons_to_geojson(polygons: List[np.ndarray], 
                        properties: Dict = None,
                        simplify: bool = True,
                        tolerance: float = 1.0) -> Dict:
    """
    Convert polygons to GeoJSON FeatureCollection.
    
    Creates a GeoJSON object compatible with Halo's annotation format.
    
    Args:
        polygons: List of polygon contours (N, 2) arrays with (row, col) coords
        properties: Optional properties to add to each feature
        simplify: Whether to simplify polygons
        tolerance: Simplification tolerance
        
    Returns:
        GeoJSON FeatureCollection dictionary
    """
    features = []
    
    for idx, polygon in enumerate(polygons):
        # Simplify polygon if requested
        if simplify:
            polygon = simplify_polygon(polygon, tolerance=tolerance)
        
        # Convert from (row, col) to (x, y) and close polygon
        coords = [[float(point[1]), float(point[0])] for point in polygon]
        coords.append(coords[0])  # Close polygon
        
        # Create feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "id": idx,
                "object_type": "annotation"
            }
        }
        
        # Add custom properties
        if properties:
            feature["properties"].update(properties)
        
        features.append(feature)
    
    # Create FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    logger.info(f"Created GeoJSON with {len(features)} features")
    
    return geojson


def save_geojson(geojson: Dict, filepath: str):
    """
    Save GeoJSON to file.
    
    Args:
        geojson: GeoJSON dictionary
        filepath: Output file path
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info(f"Saved GeoJSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save GeoJSON: {str(e)}")
        raise


def load_geojson(filepath: str) -> Dict:
    """
    Load GeoJSON from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        GeoJSON dictionary
    """
    try:
        with open(filepath, 'r') as f:
            geojson = json.load(f)
        logger.info(f"Loaded GeoJSON from {filepath}")
        return geojson
    except Exception as e:
        logger.error(f"Failed to load GeoJSON: {str(e)}")
        raise


def geojson_to_mask(geojson: Dict, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert GeoJSON polygons to binary mask.
    
    Useful for visualizing or editing annotations.
    
    Args:
        geojson: GeoJSON FeatureCollection
        shape: Output mask shape (height, width)
        
    Returns:
        Binary mask (H, W)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    
    for feature in geojson.get('features', []):
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            # Convert from (x, y) to (col, row)
            points = np.array([[int(x), int(y)] for x, y in coords])
            cv2.fillPoly(mask, [points], 1)
    
    return mask > 0


def merge_nearby_polygons(polygons: List[np.ndarray], distance_threshold: float = 10.0) -> List[np.ndarray]:
    """
    Merge polygons that are close together.
    
    Useful for combining fragmented segmentations.
    
    Args:
        polygons: List of polygon contours
        distance_threshold: Maximum distance for merging
        
    Returns:
        List of merged polygons
    """
    # This is a placeholder for a more sophisticated merging algorithm
    # In practice, you might use morphological operations or spatial clustering
    logger.warning("Polygon merging not fully implemented")
    return polygons


def filter_polygons_by_area(polygons: List[np.ndarray], 
                            min_area: float = 100,
                            max_area: float = float('inf')) -> List[np.ndarray]:
    """
    Filter polygons by area range.
    
    Args:
        polygons: List of polygon contours
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels
        
    Returns:
        Filtered list of polygons
    """
    filtered = []
    for polygon in polygons:
        area = polygon_area(polygon)
        if min_area <= area <= max_area:
            filtered.append(polygon)
    
    logger.info(f"Filtered {len(polygons)} -> {len(filtered)} polygons by area")
    return filtered
