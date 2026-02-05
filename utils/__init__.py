"""
XHaloPathAnalyzer Utilities Package

This package contains utility modules for:
- Halo API integration
- Image processing
- ML model inference
- GeoJSON conversion
"""

from .halo_api import HaloAPI
from .image_proc import (
    load_image_from_bytes,
    preprocess_for_medsam,
    postprocess_mask,
    overlay_mask_on_image,
    compute_mask_statistics
)
from .ml_models import MedSAMPredictor
from .geojson_utils import (
    mask_to_polygons,
    polygons_to_geojson,
    save_geojson
)

__all__ = [
    'HaloAPI',
    'load_image_from_bytes',
    'preprocess_for_medsam',
    'postprocess_mask',
    'overlay_mask_on_image',
    'compute_mask_statistics',
    'MedSAMPredictor',
    'mask_to_polygons',
    'polygons_to_geojson',
    'save_geojson'
]

__version__ = '1.0.0'
