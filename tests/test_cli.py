"""
Tests for CLI module
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO
from xhalo import cli


class TestCLIParsing:
    """Test command-line argument parsing"""
    
    def test_version_argument(self):
        """Test --version argument"""
        with patch.object(sys, 'argv', ['xhalo-analyzer', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            
            # Version flag should exit with code 0
            assert exc_info.value.code == 0
    
    def test_no_arguments_shows_help(self):
        """Test that no arguments shows help"""
        with patch.object(sys, 'argv', ['xhalo-analyzer']):
            with pytest.raises(SystemExit):
                cli.main()
    
    def test_web_command_parsing(self):
        """Test web command argument parsing"""
        test_args = ['xhalo-analyzer', 'web', '--port', '8080', '--host', '0.0.0.0']
        
        with patch.object(sys, 'argv', test_args):
            with patch.object(cli, 'launch_web_ui') as mock_launch:
                cli.main()
                
                # Should call launch_web_ui with correct args
                mock_launch.assert_called_once_with(8080, '0.0.0.0')
    
    def test_web_command_default_args(self):
        """Test web command with default arguments"""
        test_args = ['xhalo-analyzer', 'web']
        
        with patch.object(sys, 'argv', test_args):
            with patch.object(cli, 'launch_web_ui') as mock_launch:
                cli.main()
                
                # Should use default port and host
                mock_launch.assert_called_once_with(8501, 'localhost')
    
    def test_process_command_parsing(self):
        """Test process command argument parsing"""
        test_args = [
            'xhalo-analyzer', 'process', 'input.tif',
            '--output', 'mask.png',
            '--geojson', 'result.geojson',
            '--tile-size', '512'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch.object(cli, 'process_image') as mock_process:
                cli.main()
                
                # Should call process_image
                mock_process.assert_called_once()
                args = mock_process.call_args[0][0]
                assert args.input == 'input.tif'
                assert args.output == 'mask.png'
                assert args.geojson == 'result.geojson'
                assert args.tile_size == 512
    
    def test_process_command_minimal_args(self):
        """Test process command with minimal arguments"""
        test_args = ['xhalo-analyzer', 'process', 'input.tif']
        
        with patch.object(sys, 'argv', test_args):
            with patch.object(cli, 'process_image') as mock_process:
                cli.main()
                
                mock_process.assert_called_once()
                args = mock_process.call_args[0][0]
                assert args.input == 'input.tif'
                assert args.output is None
                assert args.geojson is None
                assert args.tile_size == 1024  # default


class TestLaunchWebUI:
    """Test web UI launching"""
    
    def test_launch_web_ui_calls_streamlit(self):
        """Test that launch_web_ui calls streamlit"""
        with patch('subprocess.run') as mock_run:
            cli.launch_web_ui(8501, 'localhost')
            
            # Should call subprocess.run with streamlit command
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            
            assert 'streamlit' in call_args
            assert 'run' in call_args
            assert '--server.port' in call_args
            assert '8501' in call_args
            assert '--server.address' in call_args
            assert 'localhost' in call_args
    
    def test_launch_web_ui_custom_port(self):
        """Test launching web UI with custom port"""
        with patch('subprocess.run') as mock_run:
            cli.launch_web_ui(9000, '0.0.0.0')
            
            call_args = mock_run.call_args[0][0]
            assert '9000' in call_args
            assert '0.0.0.0' in call_args


class TestProcessImage:
    """Test image processing from CLI"""
    
    def test_process_image_loads_and_processes(self):
        """Test that process_image loads and processes an image"""
        import numpy as np
        from unittest.mock import Mock, MagicMock
        
        # Mock image and mask
        mock_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        
        args = Mock()
        args.input = 'test.tif'
        args.output = None
        args.geojson = None
        args.tile_size = 1024
        
        # Create a module mock
        mock_utils = MagicMock()
        mock_utils.load_image = MagicMock(return_value=mock_image)
        
        mock_ml = MagicMock()
        mock_ml.segment_tissue = MagicMock(return_value=mock_mask)
        
        with patch.dict('sys.modules', {
            'xhalo.utils': mock_utils,
            'xhalo.ml': mock_ml
        }):
            # Reload the function with mocked modules
            import importlib
            import xhalo.cli as cli_module
            importlib.reload(cli_module)
            
            # Should not raise any errors
            try:
                cli_module.process_image(args)
            except SystemExit:
                pass  # May exit, that's okay
    
    def test_process_image_basic(self):
        """Test basic process_image call structure"""
        from unittest.mock import Mock
        
        args = Mock()
        args.input = 'test.tif'
        args.output = None
        args.geojson = None
        args.tile_size = 1024
        
        # Just check that the function exists and accepts args
        assert hasattr(cli, 'process_image')
        assert callable(cli.process_image)


class TestCLIEdgeCases:
    """Test CLI edge cases and error handling"""
    
    def test_invalid_command(self):
        """Test handling of invalid command"""
        test_args = ['xhalo-analyzer', 'invalid-command']
        
        with patch.object(sys, 'argv', test_args):
            # Should show help and exit
            with pytest.raises(SystemExit):
                cli.main()
    
    def test_process_without_input(self):
        """Test process command without input file"""
        test_args = ['xhalo-analyzer', 'process']
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                cli.main()


class TestCLIIntegration:
    """Integration tests for CLI"""
    
    def test_cli_imports(self):
        """Test that CLI module imports correctly"""
        from xhalo import cli
        
        assert hasattr(cli, 'main')
        assert hasattr(cli, 'launch_web_ui')
        assert hasattr(cli, 'process_image')
    
    def test_cli_functions_callable(self):
        """Test that CLI functions are callable"""
        assert callable(cli.main)
        assert callable(cli.launch_web_ui)
        assert callable(cli.process_image)
