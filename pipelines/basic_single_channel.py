"""
Basic Single-Channel Pipeline

Simple single-channel segmentation pipeline that replicates the current app behavior.
Works with any image type (single-channel, RGB, or multi-channel as RGB composite).
"""

from .base import BasePipeline
import numpy as np
from typing import Dict, List, Any

# Visualization constants
OVERLAY_ALPHA = 0.5  # Blending factor for overlay (0.0 = original, 1.0 = mask color only)
OVERLAY_COLOR = [255, 0, 0]  # Red color for mask overlay (RGB)


class BasicSingleChannelPipeline(BasePipeline):
    """
    Simple single-channel segmentation pipeline.
    This replicates the current app behavior.
    """
    
    name = "Basic Single Channel"
    description = "Standard single-channel or RGB segmentation with MicroSAM"
    version = "1.0.0"
    author = "XHistoSeg Team"
    
    required_channels = []  # Works with any image
    optional_channels = []
    
    def configure_ui(self, st):
        """UI for basic segmentation parameters"""
        
        st.write("### Segmentation Settings")
        
        config = {}
        
        # Prompt mode
        config['prompt_mode'] = st.selectbox(
            "Prompt Mode",
            ['auto_box', 'auto_box_from_threshold', 'full_box', 'point'],
            help="How to generate prompts for SAM"
        )
        
        # Mode-specific parameters
        if config['prompt_mode'] == 'auto_box':
            config['min_area_ratio'] = st.slider(
                "Min tissue area ratio", 0.0, 1.0, 0.01
            )
            config['morph_kernel_size'] = st.slider(
                "Morphology kernel size", 0, 20, 5
            )
        
        elif config['prompt_mode'] == 'auto_box_from_threshold':
            config['threshold_mode'] = st.selectbox(
                "Threshold Mode", ['otsu', 'manual']
            )
            if config['threshold_mode'] == 'manual':
                config['threshold_value'] = st.slider(
                    "Threshold Value", 0.0, 1.0, 0.5
                )
            config['min_area'] = st.slider("Min box area", 10, 1000, 100)
            config['max_area'] = st.slider("Max box area", 100, 100000, 10000)
        
        config['multimask_output'] = st.checkbox("Multi-mask output", value=False)
        
        return config
    
    def validate_channels(self, available_channels):
        """Basic pipeline works with any image"""
        return True
    
    def process(self, image, channels, predictor, config):
        """Run basic segmentation"""
        
        # Use full image (RGB or converted to RGB)
        if image.ndim == 2:
            # Grayscale -> RGB
            image_rgb = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 1:
            image_rgb = np.repeat(image, 3, axis=2)
        else:
            image_rgb = image[:, :, :3]  # Take first 3 channels as RGB
        
        # Prepare kwargs based on mode
        kwargs = {}
        if config['prompt_mode'] == 'auto_box':
            kwargs['min_area_ratio'] = config['min_area_ratio']
            kwargs['morph_kernel_size'] = config['morph_kernel_size']
        elif config['prompt_mode'] == 'auto_box_from_threshold':
            kwargs['threshold_mode'] = config['threshold_mode']
            if 'threshold_value' in config:
                kwargs['threshold_value'] = config['threshold_value']
            kwargs['box_min_area'] = config['min_area']
            kwargs['box_max_area'] = config['max_area']
        
        # Run prediction
        mask = predictor.predict(
            image_rgb,
            prompt_mode=config['prompt_mode'],
            multimask_output=config['multimask_output'],
            **kwargs
        )
        
        # Compute statistics
        num_positive = np.sum(mask > 0)
        coverage = (num_positive / mask.size) * 100
        
        results = {
            'mask': mask,
            'image': image_rgb,
            'statistics': {
                'num_positive_pixels': int(num_positive),
                'coverage_percent': float(coverage),
                'total_pixels': mask.size
            },
            'config': config
        }
        
        return results
    
    def visualize(self, results, st):
        """Display results"""
        
        st.subheader("Segmentation Results")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        stats = results['statistics']
        
        with col1:
            st.metric("Positive Pixels", f"{stats['num_positive_pixels']:,}")
        with col2:
            st.metric("Coverage", f"{stats['coverage_percent']:.2f}%")
        with col3:
            st.metric("Total Pixels", f"{stats['total_pixels']:,}")
        
        # Visualizations
        st.write("### Visualizations")
        
        mask = results['mask']
        image = results['image']
        
        # Create overlay
        mask_binary = (mask > 0).astype(np.uint8) * 255
        overlay = image.copy()
        overlay[mask > 0] = overlay[mask > 0] * (1 - OVERLAY_ALPHA) + np.array(OVERLAY_COLOR) * OVERLAY_ALPHA
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(mask_binary, caption="Mask", use_container_width=True)
        with col3:
            st.image(overlay.astype(np.uint8), caption="Overlay", use_container_width=True)
    
    def export_data(self, results):
        """Export results"""
        
        import pandas as pd
        
        # Summary statistics as CSV
        stats_df = pd.DataFrame([results['statistics']])
        
        return {
            'statistics_csv': stats_df,
            'mask': results['mask'],
            'config': results['config']
        }
