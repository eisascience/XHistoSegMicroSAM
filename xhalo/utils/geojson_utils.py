"""
GeoJSON utilities for converting segmentation masks to/from GeoJSON format
Compatible with Halo's annotation format
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import geojson
import json
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)


def mask_to_contours(
    mask: np.ndarray,
    min_area: int = 100
) -> List[np.ndarray]:
    """
    Convert binary mask to contours
    
    Args:
        mask: Binary segmentation mask (H, W)
        min_area: Minimum contour area to keep
        
    Returns:
        List of contours as numpy arrays
    """
    # Ensure binary
    mask = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter by area
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)
    
    return filtered_contours


def contour_to_polygon(contour: np.ndarray) -> Polygon:
    """
    Convert OpenCV contour to Shapely polygon
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Shapely Polygon object
    """
    # Reshape contour to list of coordinates
    points = contour.reshape(-1, 2).tolist()
    
    # Create polygon (need at least 3 points)
    if len(points) >= 3:
        return Polygon(points)
    
    return None


def mask_to_geojson(
    mask: np.ndarray,
    min_area: int = 100,
    simplify_tolerance: float = 1.0,
    properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convert segmentation mask to GeoJSON format
    
    Args:
        mask: Binary segmentation mask (H, W)
        min_area: Minimum polygon area to include
        simplify_tolerance: Tolerance for polygon simplification
        properties: Optional properties to add to each feature
        
    Returns:
        GeoJSON FeatureCollection dictionary
    """
    # Get contours from mask
    contours = mask_to_contours(mask, min_area)
    
    features = []
    
    for i, contour in enumerate(contours):
        polygon = contour_to_polygon(contour)
        
        if polygon is None or not polygon.is_valid:
            continue
        
        # Simplify polygon to reduce complexity
        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
        
        # Create feature
        feature_properties = properties.copy() if properties else {}
        feature_properties.update({
            "id": i,
            "area": polygon.area,
            "perimeter": polygon.length
        })
        
        feature = geojson.Feature(
            geometry=geojson.Polygon([list(polygon.exterior.coords)]),
            properties=feature_properties
        )
        
        features.append(feature)
    
    # Create FeatureCollection
    feature_collection = geojson.FeatureCollection(features)
    
    return feature_collection


def geojson_to_mask(
    geojson_data: Dict[str, Any],
    mask_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert GeoJSON annotations to binary mask
    
    Args:
        geojson_data: GeoJSON FeatureCollection dictionary
        mask_shape: Shape of output mask (H, W)
        
    Returns:
        Binary segmentation mask
    """
    mask = np.zeros(mask_shape, dtype=np.uint8)
    
    if "features" not in geojson_data:
        logger.warning("No features found in GeoJSON")
        return mask
    
    for feature in geojson_data["features"]:
        try:
            # Parse geometry
            geometry = shape(feature["geometry"])
            
            # Handle different geometry types
            if isinstance(geometry, Polygon):
                polygons = [geometry]
            elif isinstance(geometry, MultiPolygon):
                polygons = list(geometry.geoms)
            else:
                continue
            
            # Draw each polygon on mask
            for polygon in polygons:
                coords = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.fillPoly(mask, [coords], 1)
        
        except Exception as e:
            logger.warning(f"Error processing feature: {e}")
            continue
    
    return mask


def convert_to_halo_annotations(
    mask: np.ndarray,
    annotation_type: str = "tissue",
    layer_name: str = "AI Annotations",
    min_area: int = 100
) -> List[Dict[str, Any]]:
    """
    Convert segmentation mask to Halo-compatible annotation format
    
    Args:
        mask: Binary segmentation mask
        annotation_type: Type of annotation (tissue, cell, etc.)
        layer_name: Name of annotation layer
        min_area: Minimum polygon area
        
    Returns:
        List of Halo annotation dictionaries
    """
    contours = mask_to_contours(mask, min_area)
    
    annotations = []
    
    for i, contour in enumerate(contours):
        polygon = contour_to_polygon(contour)
        
        if polygon is None or not polygon.is_valid:
            continue
        
        # Format for Halo
        annotation = {
            "type": annotation_type,
            "layer": layer_name,
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(polygon.exterior.coords)]
            },
            "properties": {
                "id": f"ai_{annotation_type}_{i}",
                "area": float(polygon.area),
                "perimeter": float(polygon.length),
                "confidence": 1.0
            }
        }
        
        annotations.append(annotation)
    
    return annotations


def save_geojson(geojson_data: Dict[str, Any], output_path: str):
    """
    Save GeoJSON data to file
    
    Args:
        geojson_data: GeoJSON dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    logger.info(f"Saved GeoJSON to {output_path}")


def load_geojson(input_path: str) -> Dict[str, Any]:
    """
    Load GeoJSON data from file
    
    Args:
        input_path: Input file path
        
    Returns:
        GeoJSON dictionary
    """
    with open(input_path, 'r') as f:
        geojson_data = json.load(f)
    
    logger.info(f"Loaded GeoJSON from {input_path}")
    return geojson_data
