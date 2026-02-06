"""
XHaloPathAnalyzer Utilities Package

This package contains utility modules for:
- Halo API integration (utils.halo_api)
- Image processing (utils.image_proc)
- ML model inference (utils.microsam_models)
- GeoJSON conversion (utils.geojson_utils)

Import from submodules directly:
    from utils.halo_api import HaloAPI
    from utils.microsam_models import MicroSAMPredictor
    from utils.image_proc import load_image_from_bytes
    from utils.geojson_utils import mask_to_polygons

Note: MedSAM has been quarantined to utils.medsam_models and is not imported by default.
"""

__version__ = '1.0.0'

# Do not import submodules at package level to avoid import errors
# when dependencies are missing. Import from submodules directly instead.
