"""
Pipeline Framework for XHistoSegMicroSAM

Pipelines define complete analysis workflows including:
- Channel requirements
- Segmentation strategy
- Measurements
- Visualization
- Data export
"""

from .base import BasePipeline
from .basic_single_channel import BasicSingleChannelPipeline
from .multi_channel_hierarchical import MultiChannelHierarchicalPipeline

# Registry of available pipelines
AVAILABLE_PIPELINES = {
    'basic': BasicSingleChannelPipeline,
    'multi_channel': MultiChannelHierarchicalPipeline
}


def get_pipeline(name: str) -> BasePipeline:
    """Get pipeline instance by name"""
    if name not in AVAILABLE_PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(AVAILABLE_PIPELINES.keys())}")
    return AVAILABLE_PIPELINES[name]()


def list_pipelines():
    """List all available pipelines"""
    return {name: cls().get_info() for name, cls in AVAILABLE_PIPELINES.items()}
