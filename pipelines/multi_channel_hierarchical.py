"""
Multi-Channel Hierarchical Pipeline

Generic multi-channel pipeline with hierarchical object detection.

Workflow:
1. Segment nuclei (nucleus channel)
2. Use nucleus centroids as seeds for cell segmentation
3. Assign relationships (nucleus -> cell)
4. Measure per-compartment (nuclear vs cytoplasmic)
"""

from .base import BasePipeline
import numpy as np
from typing import Dict, List, Any
from scipy import ndimage

# Constants
EPSILON = 1e-6  # Small value to prevent division by zero


class MultiChannelHierarchicalPipeline(BasePipeline):
    """
    Generic multi-channel pipeline with hierarchical object detection.
    
    Workflow:
    1. Segment nuclei (nucleus channel)
    2. Use nucleus centroids as seeds for cell segmentation
    3. Assign relationships (nucleus -> cell)
    4. Measure per-compartment (nuclear vs cytoplasmic)
    """
    
    name = "Multi-Channel Hierarchical"
    description = "Nucleus-guided cell segmentation with compartmental analysis"
    version = "1.0.0"
    author = "XHistoSeg Team"
    
    required_channels = ['nucleus']
    optional_channels = ['cell_marker', 'signal']
    
    def configure_ui(self, st):
        """Configure channel assignments and parameters"""
        
        config = {}
        
        st.write("### Channel Assignment")
        st.write("Assign roles to your image channels")
        
        # This would be populated from actual image channels
        # For now, placeholder
        config['nucleus_channel'] = st.selectbox(
            "Nucleus Channel (e.g., DAPI)",
            ['Channel_0', 'Channel_1', 'Channel_2', 'Channel_3', 'Channel_4']
        )
        
        config['cell_channels'] = st.multiselect(
            "Cell Marker Channels (e.g., CD5, CD68)",
            ['Channel_0', 'Channel_1', 'Channel_2', 'Channel_3', 'Channel_4']
        )
        
        config['signal_channels'] = st.multiselect(
            "Signal Channels (e.g., vRNA, vDNA)",
            ['Channel_0', 'Channel_1', 'Channel_2', 'Channel_3', 'Channel_4']
        )
        
        st.write("### Segmentation Parameters")
        
        config['nucleus_mode'] = st.selectbox(
            "Nucleus Segmentation Mode",
            ['auto_box_from_threshold', 'auto_box']
        )
        
        config['cell_mode'] = st.selectbox(
            "Cell Segmentation Mode",
            ['point', 'auto_box']
        )
        
        config['compartmental_analysis'] = st.checkbox(
            "Enable compartmental analysis (nuclear vs cytoplasmic)",
            value=True
        )
        
        return config
    
    def validate_channels(self, available_channels):
        """Check if at least nucleus channel is available"""
        return len(available_channels) >= 1
    
    def process(self, image, channels, predictor, config):
        """Run hierarchical segmentation"""
        
        results = {}
        
        # Phase 1: Segment nuclei
        nucleus_channel_name = config['nucleus_channel']
        nucleus_img = channels[nucleus_channel_name]
        
        # Convert to RGB for MicroSAM
        nucleus_rgb = np.stack([nucleus_img]*3, axis=-1)
        
        nuclear_mask = predictor.predict(
            nucleus_rgb,
            prompt_mode=config['nucleus_mode']
        )
        
        results['nuclear_mask'] = nuclear_mask
        results['num_nuclei'] = len(np.unique(nuclear_mask)) - 1  # Exclude background
        
        # Phase 2: Get nucleus centroids for guidance
        nucleus_ids = np.unique(nuclear_mask)[1:]  # Exclude 0
        centroids = []
        for nuc_id in nucleus_ids:
            nuc_region = (nuclear_mask == nuc_id)
            centroid = ndimage.center_of_mass(nuc_region)
            centroids.append(centroid)
        
        results['centroids'] = centroids
        
        # Phase 3: Segment cells (if cell channels specified)
        cell_masks = {}
        for cell_channel_name in config['cell_channels']:
            cell_img = channels[cell_channel_name]
            cell_rgb = np.stack([cell_img]*3, axis=-1)
            
            # Use nucleus centroids as point prompts
            if config['cell_mode'] == 'point' and len(centroids) > 0:
                # TODO: Implement point-guided segmentation
                # For now, use auto mode
                cell_mask = predictor.predict(cell_rgb, prompt_mode='auto_box')
            else:
                cell_mask = predictor.predict(cell_rgb, prompt_mode=config['cell_mode'])
            
            cell_masks[cell_channel_name] = cell_mask
        
        results['cell_masks'] = cell_masks
        
        # Phase 4: Compartmental analysis
        if config['compartmental_analysis'] and config['signal_channels']:
            measurements = self._measure_compartments(
                nuclear_mask, cell_masks, channels, config['signal_channels']
            )
            results['measurements'] = measurements
        
        return results
    
    def _measure_compartments(self, nuclear_mask, cell_masks, channels, signal_channels):
        """Measure signal intensity in nuclear vs cytoplasmic compartments"""
        
        measurements = []
        
        nucleus_ids = np.unique(nuclear_mask)[1:]
        
        for nuc_id in nucleus_ids:
            nuc_region = (nuclear_mask == nuc_id)
            
            # Find parent cell (first cell mask that contains this nucleus)
            cell_region = None
            for cell_mask in cell_masks.values():
                centroid = ndimage.center_of_mass(nuc_region)
                # Add boundary checks to prevent index out of bounds
                centroid_y = int(centroid[0])
                centroid_x = int(centroid[1])
                
                # Ensure centroid is within mask bounds
                if 0 <= centroid_y < cell_mask.shape[0] and 0 <= centroid_x < cell_mask.shape[1]:
                    cell_id = cell_mask[centroid_y, centroid_x]
                    if cell_id > 0:
                        cell_region = (cell_mask == cell_id)
                        break
            
            # Measure each signal channel
            measurement = {'nucleus_id': int(nuc_id)}
            
            for sig_channel in signal_channels:
                sig_img = channels[sig_channel]
                
                # Nuclear measurement
                measurement[f'{sig_channel}_nuclear'] = float(np.mean(sig_img[nuc_region]))
                
                # Cytoplasmic measurement (if cell found)
                if cell_region is not None:
                    cyto_region = cell_region & ~nuc_region
                    if cyto_region.sum() > 0:
                        measurement[f'{sig_channel}_cytoplasmic'] = float(np.mean(sig_img[cyto_region]))
                        measurement[f'{sig_channel}_ratio'] = (
                            measurement[f'{sig_channel}_nuclear'] / 
                            (measurement[f'{sig_channel}_cytoplasmic'] + EPSILON)
                        )
            
            measurements.append(measurement)
        
        return measurements
    
    def visualize(self, results, st):
        """Display multi-channel results"""
        
        st.subheader("Multi-Channel Segmentation Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nuclei Detected", results['num_nuclei'])
        with col2:
            st.metric("Cell Channels", len(results['cell_masks']))
        with col3:
            if 'measurements' in results:
                st.metric("Objects Measured", len(results['measurements']))
        
        # Show masks
        st.write("### Segmentation Masks")
        
        cols = st.columns(len(results['cell_masks']) + 1)
        
        with cols[0]:
            nuclear_vis = (results['nuclear_mask'] > 0).astype(np.uint8) * 255
            st.image(nuclear_vis, caption="Nuclei", use_container_width=True)
        
        for i, (name, mask) in enumerate(results['cell_masks'].items()):
            with cols[i+1]:
                mask_vis = (mask > 0).astype(np.uint8) * 255
                st.image(mask_vis, caption=name, use_container_width=True)
        
        # Show measurements table
        if 'measurements' in results:
            st.write("### Compartmental Measurements")
            import pandas as pd
            df = pd.DataFrame(results['measurements'])
            st.dataframe(df)
    
    def export_data(self, results):
        """Export multi-channel results"""
        
        import pandas as pd
        
        exports = {
            'nuclear_mask': results['nuclear_mask'],
            'cell_masks': results['cell_masks']
        }
        
        if 'measurements' in results:
            exports['measurements_csv'] = pd.DataFrame(results['measurements'])
        
        return exports
