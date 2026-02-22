"""
Base Pipeline Class for XHistoSegMicroSAM

Abstract base class for analysis pipelines that defines the interface
for creating custom analysis workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class BasePipeline(ABC):
    """
    Abstract base class for analysis pipelines.
    
    A pipeline defines:
    - Required/optional channels
    - Channel roles (nucleus, cell marker, signal, etc.)
    - Processing workflow
    - Visualization
    - Data export format
    """
    
    # Metadata
    name: str = "Unnamed Pipeline"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    
    # Channel requirements
    required_channels: List[str] = []
    optional_channels: List[str] = []
    
    def __init__(self):
        self.config = {}
    
    @abstractmethod
    def configure_ui(self, st_module, available_channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Render pipeline-specific UI controls in Streamlit.
        
        Args:
            st_module: Streamlit module for rendering UI
            available_channels: List of channel names available in the selected image.
                When provided, UI selectors should offer these names instead of
                generic placeholders so that user-renamed channels appear correctly.
            
        Returns:
            Dictionary of user-configured parameters
        """
        pass
    
    @abstractmethod
    def validate_channels(self, available_channels: List[str]) -> bool:
        """
        Check if available channels meet pipeline requirements.
        
        Args:
            available_channels: List of channel names from the image
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, image: np.ndarray, channels: Dict[str, np.ndarray], 
                predictor, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the analysis pipeline.
        
        Args:
            image: Full multi-channel image (H, W, C)
            channels: Dictionary mapping channel names to 2D arrays
            predictor: MicroSAM predictor instance
            config: User configuration from configure_ui()
            
        Returns:
            Dictionary of results (masks, measurements, classifications, etc.)
        """
        pass
    
    @abstractmethod
    def visualize(self, results: Dict[str, Any], st_module):
        """
        Display pipeline-specific visualizations.
        
        Args:
            results: Output from process()
            st_module: Streamlit module for rendering
        """
        pass
    
    @abstractmethod
    def export_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export pipeline results in standard formats.
        
        Args:
            results: Output from process()
            
        Returns:
            Dictionary with keys like 'csv', 'masks', 'metadata'
        """
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get pipeline metadata."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'required_channels': self.required_channels,
            'optional_channels': self.optional_channels
        }
