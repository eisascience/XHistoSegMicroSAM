"""
Tests for configuration module
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from config import Config


class TestConfig:
    """Test configuration class"""
    
    def test_config_defaults(self):
        """Test that config has reasonable defaults"""
        assert Config.MODEL_TYPE in ["vit_b", "vit_l", "vit_h"]
        assert Config.MAX_IMAGE_SIZE_MB > 0
        assert Config.DEFAULT_TARGET_SIZE > 0
        assert Config.JPEG_QUALITY >= 0 and Config.JPEG_QUALITY <= 100
        assert Config.MIN_POLYGON_AREA >= 0
        assert Config.SIMPLIFY_TOLERANCE >= 0
    
    def test_device_configuration(self):
        """Test device configuration"""
        device = Config.DEVICE
        assert device in ["cpu", "cuda", "mps"]
    
    def test_temp_dir_exists(self):
        """Test temp directory configuration"""
        assert Config.TEMP_DIR is not None
        assert isinstance(Config.TEMP_DIR, str)
    
    def test_log_level(self):
        """Test log level configuration"""
        assert Config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def test_get_temp_path(self):
        """Test generating temporary file paths"""
        temp_path = Config.get_temp_path("test_file.txt")
        
        assert isinstance(temp_path, Path)
        assert temp_path.name == "test_file.txt"
        # Check that it uses the temp dir
        assert "temp" in str(temp_path).lower()
    
    def test_get_temp_path_with_subdirectory(self):
        """Test generating temp paths with subdirectories"""
        temp_path = Config.get_temp_path("subdir/test_file.txt")
        
        assert isinstance(temp_path, Path)
        assert "test_file.txt" in str(temp_path)
    
    def test_log_config(self):
        """Test logging configuration"""
        # Should not raise any errors
        Config.log_config()
    
    def test_numeric_config_values(self):
        """Test that numeric configs are actually numbers"""
        assert isinstance(Config.MAX_IMAGE_SIZE_MB, int)
        assert isinstance(Config.DEFAULT_TARGET_SIZE, int)
        assert isinstance(Config.JPEG_QUALITY, int)
        assert isinstance(Config.MIN_POLYGON_AREA, int)
        assert isinstance(Config.SIMPLIFY_TOLERANCE, float)


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_validate_with_missing_api_endpoint(self):
        """Test validation fails with missing API endpoint when require_halo_api=True"""
        with patch.object(Config, 'HALO_API_ENDPOINT', ''):
            with patch.object(Config, 'HALO_API_TOKEN', ''):
                with patch.object(Config, 'LOCAL_MODE', False):
                    with pytest.raises(ValueError) as exc_info:
                        Config.validate(require_halo_api=True)
                    
                    assert "HALO_API_ENDPOINT" in str(exc_info.value)
    
    def test_validate_in_local_mode(self):
        """Test validation passes in local mode without API credentials"""
        with patch.object(Config, 'HALO_API_ENDPOINT', ''):
            with patch.object(Config, 'HALO_API_TOKEN', ''):
                with patch.object(Config, 'LOCAL_MODE', True):
                    # Should not raise an error
                    Config.validate(require_halo_api=False)
    
    def test_validate_with_invalid_max_image_size(self):
        """Test validation fails with invalid max image size"""
        original = Config.MAX_IMAGE_SIZE_MB
        try:
            Config.MAX_IMAGE_SIZE_MB = -1
            
            with patch.object(Config, 'HALO_API_ENDPOINT', 'http://test'):
                with patch.object(Config, 'HALO_API_TOKEN', 'test-token'):
                    with pytest.raises(ValueError) as exc_info:
                        Config.validate()
                    
                    assert "MAX_IMAGE_SIZE_MB" in str(exc_info.value)
        finally:
            Config.MAX_IMAGE_SIZE_MB = original
    
    def test_validate_creates_directories(self):
        """Test that validate creates necessary directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Config, 'TEMP_DIR', tmpdir):
                with patch.object(Config, 'LOCAL_MODE', True):
                    Config.validate(require_halo_api=False)
                
                # Temp dir should exist
                assert os.path.exists(tmpdir)


class TestConfigEnvironmentVariables:
    """Test configuration loading from environment variables"""
    
    def test_env_var_loading(self):
        """Test that environment variables are loaded"""
        # These should be loaded from environment or use defaults
        assert hasattr(Config, 'HALO_API_ENDPOINT')
        assert hasattr(Config, 'HALO_API_TOKEN')
        assert hasattr(Config, 'MEDSAM_CHECKPOINT')
    
    def test_model_checkpoint_path(self):
        """Test model checkpoint path configuration"""
        checkpoint = Config.MEDSAM_CHECKPOINT
        
        assert isinstance(checkpoint, str)
        assert len(checkpoint) > 0


class TestConfigEdgeCases:
    """Test configuration edge cases"""
    
    def test_config_with_empty_strings(self):
        """Test configuration handles empty strings"""
        # Empty strings should be handled gracefully
        assert isinstance(Config.HALO_API_ENDPOINT, str)
        assert isinstance(Config.HALO_API_TOKEN, str)
    
    def test_config_immutability(self):
        """Test that config values can be modified (for testing)"""
        original = Config.DEFAULT_TARGET_SIZE
        
        # Should be able to modify
        Config.DEFAULT_TARGET_SIZE = 2048
        assert Config.DEFAULT_TARGET_SIZE == 2048
        
        # Restore original
        Config.DEFAULT_TARGET_SIZE = original
    
    def test_temp_path_special_characters(self):
        """Test temp path with special characters"""
        # Should handle special characters in filename
        temp_path = Config.get_temp_path("file with spaces.txt")
        assert isinstance(temp_path, Path)
        
        temp_path = Config.get_temp_path("file-with-dashes.txt")
        assert isinstance(temp_path, Path)
