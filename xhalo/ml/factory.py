"""
ML Model Factory

Factory functions for creating segmentation predictors.
"""

from typing import Optional, Dict, Any
import logging
from .microsam import MicroSAMPredictor

logger = logging.getLogger(__name__)


def get_predictor(
    model_type: str = "vit_b_histopathology",
    device: Optional[str] = None,
    tile_shape: tuple = (1024, 1024),
    halo: tuple = (256, 256),
    **kwargs
) -> MicroSAMPredictor:
    """
    Factory function to create a segmentation predictor.
    
    Args:
        model_type: Model architecture (default: vit_b_histopathology)
        device: Device to use (cuda/mps/cpu), auto-detected if None
        tile_shape: Tile size for tiled inference
        halo: Halo size for tiled inference
        **kwargs: Additional predictor-specific arguments
        
    Returns:
        Initialized MicroSAMPredictor instance
    """
    logger.info(f"Creating MicroSAM predictor with model={model_type}")
    
    return MicroSAMPredictor(
        model_type=model_type,
        device=device,
        tile_shape=tile_shape,
        halo=halo
    )


def get_predictor_from_config(config: Any) -> MicroSAMPredictor:
    """
    Create predictor from configuration object.
    
    Args:
        config: Configuration object with attributes:
            - MICROSAM_MODEL_TYPE
            - DEVICE
            - TILE_SHAPE
            - HALO_SIZE
            
    Returns:
        Initialized MicroSAMPredictor instance
    """
    return get_predictor(
        model_type=getattr(config, 'MICROSAM_MODEL_TYPE', 'vit_b_histopathology'),
        device=getattr(config, 'DEVICE', None),
        tile_shape=getattr(config, 'TILE_SHAPE', (1024, 1024)),
        halo=getattr(config, 'HALO_SIZE', (256, 256))
    )
