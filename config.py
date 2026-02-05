"""
Configuration Management for XHaloPathAnalyzer

Centralized configuration with environment variable support.
"""

import os
import torch
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    Returns:
        bool: True if MPS is available, False otherwise
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


class Config:
    """
    Centralized configuration management for XHaloPathAnalyzer.
    
    Loads settings from environment variables and provides defaults.
    Validates configuration on initialization.
    """
    
    # Halo API Settings
    HALO_API_ENDPOINT = os.getenv("HALO_API_ENDPOINT", "")
    HALO_API_TOKEN = os.getenv("HALO_API_TOKEN", "")
    
    # Halo Link Integration Settings
    HALOLINK_BASE_URL = os.getenv("HALOLINK_BASE_URL", "")
    HALOLINK_GRAPHQL_URL = os.getenv("HALOLINK_GRAPHQL_URL", "")
    HALOLINK_GRAPHQL_PATH = os.getenv("HALOLINK_GRAPHQL_PATH", "")
    HALOLINK_CLIENT_ID = os.getenv("HALOLINK_CLIENT_ID", "")
    HALOLINK_CLIENT_SECRET = os.getenv("HALOLINK_CLIENT_SECRET", "")
    HALOLINK_SCOPE = os.getenv("HALOLINK_SCOPE", "")
    
    # Local Mode (bypass Halo API requirement)
    LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
    
    # MicroSAM Model Settings
    MICROSAM_MODEL_TYPE = os.getenv("MICROSAM_MODEL_TYPE", "vit_b_histopathology")
    MICROSAM_CACHEDIR = os.getenv("MICROSAM_CACHEDIR", None)  # Model cache directory (optional)
    
    # Tiling Settings for Large Images
    try:
        TILE_SHAPE = tuple(map(int, os.getenv("TILE_SHAPE", "1024,1024").split(",")))
        if len(TILE_SHAPE) != 2:
            raise ValueError("TILE_SHAPE must have exactly 2 values")
    except (ValueError, TypeError):
        logger.error("TILE_SHAPE must be two comma-separated integers, using default: 1024,1024")
        TILE_SHAPE = (1024, 1024)
    
    try:
        HALO_SIZE = tuple(map(int, os.getenv("HALO_SIZE", "256,256").split(",")))
        if len(HALO_SIZE) != 2:
            raise ValueError("HALO_SIZE must have exactly 2 values")
    except (ValueError, TypeError):
        logger.error("HALO_SIZE must be two comma-separated integers, using default: 256,256")
        HALO_SIZE = (256, 256)
    
    ENABLE_TILING = os.getenv("ENABLE_TILING", "true").lower() == "true"
    
    # Embeddings Cache Settings
    ENABLE_EMBEDDINGS_CACHE = os.getenv("ENABLE_EMBEDDINGS_CACHE", "false").lower() == "true"
    EMBEDDINGS_CACHE_DIR = os.getenv("EMBEDDINGS_CACHE_DIR", "./cache/embeddings")
    
    # Application Settings
    try:
        MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "500"))
    except ValueError:
        logger.error("MAX_IMAGE_SIZE_MB must be a number, using default: 500")
        MAX_IMAGE_SIZE_MB = 500
    
    TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Image Processing Settings
    try:
        DEFAULT_TARGET_SIZE = int(os.getenv("DEFAULT_TARGET_SIZE", "1024"))
    except ValueError:
        logger.error("DEFAULT_TARGET_SIZE must be a number, using default: 1024")
        DEFAULT_TARGET_SIZE = 1024
    
    try:
        JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "95"))
    except ValueError:
        logger.error("JPEG_QUALITY must be a number, using default: 95")
        JPEG_QUALITY = 95
    
    # Device Configuration (automatically detect CUDA/MPS)
    # Priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif is_mps_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    
    # GeoJSON Settings
    try:
        MIN_POLYGON_AREA = int(os.getenv("MIN_POLYGON_AREA", "100"))
    except ValueError:
        logger.error("MIN_POLYGON_AREA must be a number, using default: 100")
        MIN_POLYGON_AREA = 100
    
    try:
        SIMPLIFY_TOLERANCE = float(os.getenv("SIMPLIFY_TOLERANCE", "1.0"))
    except ValueError:
        logger.error("SIMPLIFY_TOLERANCE must be a number, using default: 1.0")
        SIMPLIFY_TOLERANCE = 1.0
    
    @classmethod
    def validate(cls, require_halo_api=True):
        """
        Validate required configuration settings.
        
        Args:
            require_halo_api: If False, skip Halo API validation (for local mode)
        
        Raises:
            ValueError: If required settings are missing or invalid
        """
        errors = []
        
        # Check required settings (only if not in local mode)
        if require_halo_api and not cls.LOCAL_MODE:
            if not cls.HALO_API_ENDPOINT:
                errors.append("HALO_API_ENDPOINT is required")
            if not cls.HALO_API_TOKEN:
                errors.append("HALO_API_TOKEN is required")
            
        # Validate numeric settings
        if cls.MAX_IMAGE_SIZE_MB <= 0:
            errors.append("MAX_IMAGE_SIZE_MB must be positive")
        
        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))
        
        # Create directories if they don't exist
        Path(cls.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        if cls.ENABLE_EMBEDDINGS_CACHE:
            Path(cls.EMBEDDINGS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Configuration validated successfully")
        logger.info(f"Using device: {cls.DEVICE}")
        
    @classmethod
    def get_temp_path(cls, filename: str) -> Path:
        """Generate path for temporary file"""
        return Path(cls.TEMP_DIR) / filename
    
    @classmethod
    def log_config(cls):
        """Log current configuration (excluding sensitive data)"""
        logger.info("=== Configuration ===")
        logger.info(f"API Endpoint: {cls.HALO_API_ENDPOINT}")
        logger.info(f"MicroSAM Model: {cls.MICROSAM_MODEL_TYPE}")
        logger.info(f"Device: {cls.DEVICE}")
        logger.info(f"Temp Directory: {cls.TEMP_DIR}")
        logger.info(f"Tiling: {cls.ENABLE_TILING} (tile_shape={cls.TILE_SHAPE}, halo={cls.HALO_SIZE})")
        logger.info("====================")
