"""XHaloPathAnalyzer - Main Streamlit Application

Web-based GUI for custom image analysis on Halo digital pathology slides.
Provides interface for:
- Authentication with Halo API
- Slide selection and metadata viewing
- ROI export and image processing
- MedSAM segmentation analysis
- GeoJSON export for Halo import
Halo AI Workflow - Main Streamlit Application
Web-based GUI for digital pathology analysis with Halo API integration
"""

import streamlit as st
import asyncio
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

from config import Config
from utils.halo_api import HaloAPI
from utils.image_proc import (
    load_image_from_bytes,
    overlay_mask_on_image,
    compute_mask_statistics
)
from utils.ml_models import MedSAMPredictor as UtilsMedSAMPredictor, _ensure_rgb_uint8, _compute_tissue_bbox
from utils.geojson_utils import (
    mask_to_polygons,
    polygons_to_geojson
)
from PIL import Image
import io
import json
import logging
from typing import Optional, List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for visualization
PROMPT_BOX_COLOR = (0, 255, 0)  # Green in RGB
PROMPT_BOX_THICKNESS = 3
PROMPT_BOX_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
PROMPT_BOX_LABEL_SCALE = 1.0
PROMPT_BOX_LABEL_THICKNESS = 2
PROMPT_BOX_LABEL_Y_OFFSET = 10
PROMPT_BOX_LABEL_MIN_Y = 20

# Channel mapping constant
CHANNEL_INDEX_MAP = {'R': 0, 'G': 1, 'B': 2}

# Import local modules
from xhalo.api import HaloAPIClient, MockHaloAPIClient
from xhalo.ml import MedSAMPredictor as XHaloMedSAMPredictor, segment_tissue
from xhalo.utils import (
    load_image,
    resize_image,
    overlay_mask,
    mask_to_geojson,
    convert_to_halo_annotations,
    save_geojson
)

# Page configuration
st.set_page_config(
    page_title="XHaloPathAnalyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'local_mode' not in st.session_state:
        st.session_state.local_mode = False
    if 'api' not in st.session_state:
        st.session_state.api = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'selected_slide' not in st.session_state:
        st.session_state.selected_slide = None
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_mask' not in st.session_state:
        st.session_state.current_mask = None
    if 'current_image_name' not in st.session_state:
        st.session_state.current_image_name = None
    # Multi-image queue support
    if 'images' not in st.session_state:
        st.session_state.images = []  # List of dicts with id, name, bytes, np_rgb_uint8, status, error, result
    if 'batch_running' not in st.session_state:
        st.session_state.batch_running = False
    if 'batch_index' not in st.session_state:
        st.session_state.batch_index = 0

init_session_state()


def authentication_page():
    """Authentication and configuration page"""
    st.markdown('<h1 class="main-header">XHaloPathAnalyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Web-Based GUI for Halo Digital Pathology Analysis")
    
    st.markdown("---")
    
    # Add mode selection at the top
    st.subheader("Select Analysis Mode")
    mode = st.radio(
        "Choose how you want to work:",
        ["Halo API Mode", "Local Image Upload Mode"],
        help="Halo API Mode connects to your Halo instance. Local Mode allows direct upload of images."
    )
    
    if mode == "Local Image Upload Mode":
        st.info("**Local Mode**: Upload images (JPG, PNG, TIFF) directly for analysis without Halo API connection")
        
        if st.button("Start Local Mode", type="primary", ):
            st.session_state.authenticated = True
            st.session_state.local_mode = True
            st.success("Local mode activated!")
            st.rerun()
            
        st.markdown("---")
        st.markdown("### Features in Local Mode")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Image Upload**")
            st.write("Upload single or multiple images for analysis")
        with col2:
            st.markdown("**AI Analysis**")
            st.write("Run MedSAM segmentation on uploaded images")
        with col3:
            st.markdown("**Export Results**")
            st.write("Download segmentation masks and GeoJSON")
            
    else:
        st.subheader("Halo API Authentication")
        st.write("Connect to your Halo digital pathology instance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            endpoint = st.text_input(
                "Halo API Endpoint",
                value=Config.HALO_API_ENDPOINT,
                placeholder="https://your-halo-instance.com/graphql",
                help="URL of your Halo GraphQL API endpoint"
            )
            
            token = st.text_input(
                "API Token",
                value=Config.HALO_API_TOKEN,
                type="password",
                placeholder="Enter your API token",
                help="API authentication token from Halo settings"
            )
            
            if st.button("Connect", type="primary", ):
                if not endpoint or not token:
                    st.error("Please provide both endpoint and token")
                else:
                    with st.spinner("Testing connection..."):
                        try:
                            # Create API instance
                            api = HaloAPI(endpoint, token)
                            
                            # Test connection
                            success = asyncio.run(api.test_connection())
                            
                            if success:
                                st.session_state.api = api
                                st.session_state.authenticated = True
                                st.session_state.local_mode = False
                                st.success("Connected successfully!")
                                st.rerun()
                            else:
                                st.error("Connection test failed")
                                
                        except Exception as e:
                            st.error(f"Connection failed: {str(e)}")
        
        with col2:
            st.info("""
            **How to get API token:**
            1. Log into Halo
            2. Go to Settings → API
            3. Create new token
            4. Copy and paste here
            """)
        
        st.markdown("---")
        st.markdown("### Features in Halo Mode")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Slide Selection**")
            st.write("Browse and select slides from your Halo instance")
        with col2:
            st.markdown("**AI Analysis**")
            st.write("Run MedSAM segmentation on regions of interest")
        with col3:
            st.markdown("**Export Results**")
            st.write("Generate GeoJSON annotations for Halo import")


def slide_selection_page():
    """Slide selection and browsing interface"""
    st.title("Slide Selection")
    
    if st.session_state.api is None:
        st.warning("Please authenticate first")
        return
    
    # Fetch slides
    with st.spinner("Loading slides from Halo..."):
        try:
            slides = asyncio.run(st.session_state.api.get_slides(limit=100))
        except Exception as e:
            st.error(f"Failed to fetch slides: {str(e)}")
            return
    
    if not slides:
        st.warning("No slides found in your Halo instance")
        return
    
    st.success(f"Found {len(slides)} slides")
    
    # Convert to DataFrame for display
    df = pd.DataFrame(slides)
    
    # Add filters
    st.subheader("Filter Slides")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name_filter = st.text_input("Filter by name", "")
    with col2:
        study_filter = st.text_input("Filter by study ID", "")
    
    # Apply filters
    if name_filter:
        df = df[df['name'].str.contains(name_filter, case=False, na=False)]
    if study_filter:
        df = df[df['studyId'].str.contains(study_filter, case=False, na=False)]
    
    st.markdown("---")
    
    # Display slides table
    st.subheader("Available Slides")
    
    if len(df) == 0:
        st.info("No slides match the filter criteria")
        return
    
    # Select slide from dropdown
    slide_names = df['name'].tolist()
    selected_name = st.selectbox(
        "Select a slide",
        slide_names,
        help="Choose a slide to analyze"
    )
    
    # Get selected slide data
    selected_idx = slide_names.index(selected_name)
    selected_slide = slides[selected_idx]
    
    # Display slide details
    st.markdown("---")
    st.subheader("Slide Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Name", selected_slide['name'])
        st.metric("Width", f"{selected_slide['width']:,} px")
    with col2:
        st.metric("ID", selected_slide['id'][:16] + "...")
        st.metric("Height", f"{selected_slide['height']:,} px")
    with col3:
        mpp = selected_slide.get('mpp', 'N/A')
        st.metric("MPP", f"{mpp}" if mpp != 'N/A' else "N/A")
        st.metric("Format", selected_slide.get('format', 'Unknown'))
    
    # Save to session state
    if st.button("Select This Slide", type="primary", ):
        st.session_state.selected_slide = selected_slide
        st.success(f"Selected: {selected_slide['name']}")


def prepare_channel_input(image: np.ndarray, channel_config: Dict[str, Any]) -> List[Tuple[np.ndarray, str]]:
    """
    Prepare channel input(s) for MedSAM based on channel configuration.
    
    Args:
        image: RGB uint8 image (H, W, 3)
        channel_config: Dict with 'mode' and 'channels' keys
            mode: 'rgb', 'single', or 'multi'
            channels: list of channel names ['R', 'G', 'B'] for single/multi mode
    
    Returns:
        List of tuples (processed_image, channel_name) where processed_image is (H, W, 3) uint8
    """
    mode = channel_config.get('mode', 'rgb')
    selected_channels = channel_config.get('channels', ['R', 'G', 'B'])
    
    if mode == 'rgb':
        # RGB composite - return as-is
        return [(image, 'RGB')]
    
    elif mode == 'single':
        # Single channel - replicate into 3 channels
        if not selected_channels:
            selected_channels = ['R']
        
        channel_name = selected_channels[0]
        channel_idx = CHANNEL_INDEX_MAP[channel_name]
        single_channel = image[:, :, channel_idx]
        
        # Replicate to 3 channels
        three_channel = np.stack([single_channel, single_channel, single_channel], axis=-1)
        return [(three_channel, channel_name)]
    
    elif mode == 'multi':
        # Multi-channel - generate one image per selected channel
        results = []
        for channel_name in selected_channels:
            channel_idx = CHANNEL_INDEX_MAP[channel_name]
            single_channel = image[:, :, channel_idx]
            three_channel = np.stack([single_channel, single_channel, single_channel], axis=-1)
            results.append((three_channel, channel_name))
        return results
    
    # Default fallback
    return [(image, 'RGB')]


def merge_channel_masks(channel_masks: Dict[str, np.ndarray], merge_mode: str, k_value: int = 1) -> np.ndarray:
    """
    Merge masks from multiple channels.
    
    Args:
        channel_masks: Dict mapping channel name to binary mask (H, W)
        merge_mode: 'union', 'intersection', or 'voting'
        k_value: For voting mode, require k out of n channels to be positive
    
    Returns:
        Merged binary mask (H, W)
    """
    if not channel_masks:
        return None
    
    masks_list = list(channel_masks.values())
    
    if merge_mode == 'union':
        # Any channel positive
        merged = np.zeros_like(masks_list[0], dtype=bool)
        for mask in masks_list:
            merged |= mask > 0
        return merged.astype(np.uint8) * 255
    
    elif merge_mode == 'intersection':
        # All channels positive
        merged = np.ones_like(masks_list[0], dtype=bool)
        for mask in masks_list:
            merged &= mask > 0
        return merged.astype(np.uint8) * 255
    
    elif merge_mode == 'voting':
        # k out of n channels positive
        vote_sum = np.zeros_like(masks_list[0], dtype=int)
        for mask in masks_list:
            vote_sum += (mask > 0).astype(int)
        merged = vote_sum >= k_value
        return merged.astype(np.uint8) * 255
    
    # Default: union
    merged = np.zeros_like(masks_list[0], dtype=bool)
    for mask in masks_list:
        merged |= mask > 0
    return merged.astype(np.uint8) * 255


def post_process_mask(mask: np.ndarray, 
                      min_area_px: int = 0,
                      fill_holes: bool = False,
                      morph_open_radius: int = 0,
                      morph_close_radius: int = 0,
                      watershed_split: bool = False,
                      min_distance: int = 10) -> Dict[str, Any]:
    """
    Post-process binary mask with cleaning and optional instance segmentation.
    
    Args:
        mask: Binary mask (H, W) uint8
        min_area_px: Remove objects smaller than this (in pixels)
        fill_holes: Fill holes in objects
        morph_open_radius: Morphological opening radius (0 = skip)
        morph_close_radius: Morphological closing radius (0 = skip)
        watershed_split: Apply watershed for instance segmentation
        min_distance: Minimum distance for watershed seed detection
    
    Returns:
        Dict with:
            - binary_mask: Cleaned binary mask
            - instance_mask: Instance label image (int32) if watershed enabled, else None
            - measurements: List of per-object measurements
    """
    from skimage import morphology, measure
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    
    # Convert to binary if needed
    binary = mask > 0
    
    # Morphological operations
    if morph_open_radius > 0:
        footprint = morphology.disk(morph_open_radius)
        binary = morphology.opening(binary, footprint)
    
    if morph_close_radius > 0:
        footprint = morphology.disk(morph_close_radius)
        binary = morphology.closing(binary, footprint)
    
    # Fill holes
    if fill_holes:
        binary = ndi.binary_fill_holes(binary)
    
    # Remove small objects
    if min_area_px > 0:
        # Note: max_size parameter removes objects with area <= threshold (new API in scikit-image 0.26+)
        binary = morphology.remove_small_objects(binary, max_size=min_area_px)
    
    # Instance segmentation
    instance_mask = None
    measurements = []
    
    if watershed_split and np.any(binary):
        # Compute distance transform
        distance = ndi.distance_transform_edt(binary)
        
        # Find peaks (local maxima)
        from skimage.feature import peak_local_max
        coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
        
        # Create markers
        markers = np.zeros_like(binary, dtype=int)
        for i, coord in enumerate(coords):
            markers[coord[0], coord[1]] = i + 1
        
        # Apply watershed
        if np.max(markers) > 0:
            instance_mask = watershed(-distance, markers, mask=binary)
        else:
            # No peaks found, use connected components
            instance_mask = measure.label(binary)
    else:
        # Just use connected components for measurements
        instance_mask = measure.label(binary)
    
    # Compute measurements
    if instance_mask is not None and np.max(instance_mask) > 0:
        props = measure.regionprops(instance_mask)
        for prop in props:
            measurements.append({
                'label': prop.label,
                'area': prop.area,
                'centroid': prop.centroid,
                'bbox': prop.bbox
            })
    
    return {
        'binary_mask': binary.astype(np.uint8) * 255,
        'instance_mask': instance_mask,
        'measurements': measurements
    }


def run_analysis_on_item(item: Dict[str, Any], prompt_mode: str = "auto_box", 
                         multimask_output: bool = False, 
                         min_area_ratio: float = 0.01,
                         morph_kernel_size: int = 5,
                         post_processing: Optional[Dict[str, Any]] = None,
                         merge_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run analysis on a single image item with channel and post-processing support.
    
    Args:
        item: Image item dict with 'bytes' field containing raw image data
        prompt_mode: Segmentation prompt mode (auto_box, full_box, point)
        multimask_output: Whether to generate multiple mask predictions
        min_area_ratio: Minimum area ratio for tissue detection (0-1)
        morph_kernel_size: Kernel size for morphological operations (odd integer)
        post_processing: Dict with post-processing params (min_area_px, fill_holes, etc.)
        merge_config: Dict with merge mode and k_value for multi-channel merging
        
    Returns:
        Dict with the following keys:
            - image: numpy.ndarray - Original RGB image (H, W, 3)
            - mask: numpy.ndarray - Binary segmentation mask (H, W)
            - channel_masks: dict - Per-channel masks if multi-channel mode
            - mask_merged: numpy.ndarray - Merged mask if multi-channel mode
            - binary_mask: numpy.ndarray - Post-processed binary mask
            - instance_mask: numpy.ndarray - Instance label image if watershed enabled
            - measurements: list - Per-object measurements
            - statistics: dict - Mask statistics (num_positive_pixels, coverage_percent, etc.)
            - overlay: numpy.ndarray - Image with mask overlay
            - prompt_box: numpy.ndarray or None - Bounding box used for prompt [x1, y1, x2, y2]
            - img_with_box: numpy.ndarray or None - Image with prompt box visualization
            - prompt_mode: str - Prompt mode used
            - timestamp: str - ISO format timestamp
    
    Raises:
        Exception: If image loading, segmentation, or processing fails
        
    Example:
        >>> item = {'bytes': image_bytes, 'name': 'test.png', ...}
        >>> result = run_analysis_on_item(item, prompt_mode='auto_box')
        >>> print(result['statistics']['coverage_percent'])
    """
    # Decode bytes to RGB uint8 numpy
    image = load_image_from_bytes(item['bytes'])
    item['np_rgb_uint8'] = image  # Store for future use
    
    # Initialize predictor if needed
    if st.session_state.predictor is None:
        st.session_state.predictor = UtilsMedSAMPredictor(
            Config.MEDSAM_CHECKPOINT,
            model_type=Config.MODEL_TYPE,
            device=Config.DEVICE
        )
    
    # Get channel configuration
    channel_config = item.get('channel_config', {'mode': 'rgb', 'channels': ['R', 'G', 'B']})
    
    # Prepare channel inputs
    channel_inputs = prepare_channel_input(image, channel_config)
    
    # Run segmentation on each channel
    channel_masks = {}
    prompt_box = None
    img_with_box = None
    
    for channel_image, channel_name in channel_inputs:
        # Compute prompt box for visualization (only for first channel)
        if prompt_box is None:
            if prompt_mode == "auto_box":
                image_rgb = _ensure_rgb_uint8(channel_image)
                prompt_box = _compute_tissue_bbox(image_rgb, min_area_ratio, morph_kernel_size)
            elif prompt_mode == "full_box":
                h, w = channel_image.shape[:2]
                prompt_box = np.array([0, 0, w - 1, h - 1])
        
        # Run segmentation
        channel_mask = st.session_state.predictor.predict(
            channel_image,
            prompt_mode=prompt_mode,
            multimask_output=multimask_output,
            min_area_ratio=min_area_ratio,
            morph_kernel_size=morph_kernel_size
        )
        
        channel_masks[channel_name] = channel_mask
    
    # Merge masks if multi-channel
    if len(channel_masks) > 1:
        merge_mode = merge_config.get('mode', 'union') if merge_config else 'union'
        k_value = merge_config.get('k_value', 1) if merge_config else 1
        mask = merge_channel_masks(channel_masks, merge_mode, k_value)
        mask_merged = mask
    else:
        # Single channel or RGB mode
        mask = list(channel_masks.values())[0]
        mask_merged = None
    
    # Apply post-processing if requested
    binary_mask = mask
    instance_mask = None
    measurements = []
    
    if post_processing:
        post_result = post_process_mask(
            mask,
            min_area_px=post_processing.get('min_area_px', 0),
            fill_holes=post_processing.get('fill_holes', False),
            morph_open_radius=post_processing.get('morph_open_radius', 0),
            morph_close_radius=post_processing.get('morph_close_radius', 0),
            watershed_split=post_processing.get('watershed_split', False),
            min_distance=post_processing.get('min_distance', 10)
        )
        binary_mask = post_result['binary_mask']
        instance_mask = post_result['instance_mask']
        measurements = post_result['measurements']
    
    # Compute statistics on final mask
    stats = compute_mask_statistics(binary_mask, mpp=None)
    
    # Create overlay visualization using final mask
    overlay = overlay_mask_on_image(
        image,
        binary_mask,
        color=(255, 0, 0),
        alpha=0.5
    )
    
    # Create prompt box visualization if available
    if prompt_box is not None:
        img_with_box = image.copy()
        x1, y1, x2, y2 = prompt_box.astype(int)
        img_with_box = cv2.rectangle(img_with_box, (x1, y1), (x2, y2), 
                                    PROMPT_BOX_COLOR, PROMPT_BOX_THICKNESS)
        label_y = max(y1 - PROMPT_BOX_LABEL_Y_OFFSET, PROMPT_BOX_LABEL_MIN_Y)
        cv2.putText(img_with_box, f"Prompt: {prompt_mode}", 
                   (x1, label_y), PROMPT_BOX_LABEL_FONT, PROMPT_BOX_LABEL_SCALE,
                   PROMPT_BOX_COLOR, PROMPT_BOX_LABEL_THICKNESS)
    
    # Build and return result dict
    result = {
        'image': image,
        'mask': mask,  # Original mask before post-processing
        'channel_masks': channel_masks if len(channel_masks) > 1 else None,
        'mask_merged': mask_merged,
        'binary_mask': binary_mask,
        'instance_mask': instance_mask,
        'measurements': measurements,
        'statistics': stats,
        'overlay': overlay,
        'prompt_box': prompt_box,
        'img_with_box': img_with_box,
        'prompt_mode': prompt_mode,
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def image_upload_page():
    """Image upload interface for local mode"""
    st.title("Image Upload")
    
    st.markdown("""Upload one or more images (JPG, PNG, TIFF) for analysis. 
    Batch processing allows you to analyze multiple images sequentially.
    """)
    
    st.markdown("---")
    
    # File uploader
    st.subheader("Select Images")
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "tiff", "tif"],
        accept_multiple_files=True,
        help="Supported formats: JPG, PNG, TIFF"
    )
    
    if uploaded_files:
        st.success(f" {len(uploaded_files)} file(s) uploaded")
        
        # Automatically populate session_state.images from uploaded files
        # Create unique IDs based on filename and file size
        uploaded_ids = set()
        existing_ids = {img['id'] for img in st.session_state.images}  # O(1) lookup
        
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            uploaded_ids.add(file_id)
            
            # Check if this image is already in the list (O(1) lookup)
            if file_id not in existing_ids:
                # Read bytes
                image_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Add to images list
                st.session_state.images.append({
                    'id': file_id,
                    'name': uploaded_file.name,
                    'bytes': image_bytes,
                    'np_rgb_uint8': None,
                    'status': 'ready',  # ready, processing, done, failed, skipped
                    'error': None,
                    'result': None,
                    'include': True  # Whether to include in batch processing
                })
        
        # Remove images that are no longer in uploaded_files
        st.session_state.images = [
            img for img in st.session_state.images 
            if img['id'] in uploaded_ids
        ]
        
        # Display uploaded images with status
        st.subheader("Uploaded Images")
        
        if st.session_state.images:
            # Create table data
            table_data = []
            for i, img in enumerate(st.session_state.images):
                # Preview thumbnail
                try:
                    if img['np_rgb_uint8'] is None and img['status'] != 'processing':
                        preview_img = load_image_from_bytes(img['bytes'])
                        # Create small thumbnail for preview (100px height)
                        h, w = preview_img.shape[:2]
                        thumb_h = 100
                        thumb_w = int(w * thumb_h / h)
                        thumbnail = cv2.resize(preview_img, (thumb_w, thumb_h))
                    else:
                        thumbnail = img.get('np_rgb_uint8')
                        if thumbnail is not None:
                            h, w = thumbnail.shape[:2]
                            thumb_h = 100
                            thumb_w = int(w * thumb_h / h)
                            thumbnail = cv2.resize(thumbnail, (thumb_w, thumb_h))
                except Exception as e:
                    # Handle image loading/processing errors gracefully
                    thumbnail = None
                
                # Get dimensions
                try:
                    if img['np_rgb_uint8'] is not None:
                        dims = f"{img['np_rgb_uint8'].shape[1]} × {img['np_rgb_uint8'].shape[0]}"
                    else:
                        temp_img = load_image_from_bytes(img['bytes'])
                        dims = f"{temp_img.shape[1]} × {temp_img.shape[0]}"
                except Exception as e:
                    # Handle image loading errors gracefully
                    dims = "Unknown"
                
                table_data.append({
                    'index': i,
                    'name': img['name'],
                    'dimensions': dims,
                    'status': img['status'],
                    'thumbnail': thumbnail
                })
            
            # Display table
            for row in table_data:
                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 1])
                
                with col1:
                    # Include checkbox
                    include = st.checkbox(
                        "Include",
                        value=st.session_state.images[row['index']]['include'],
                        key=f"include_{row['index']}",
                        label_visibility="collapsed"
                    )
                    st.session_state.images[row['index']]['include'] = include
                
                with col2:
                    st.write(f"**{row['name']}**")
                
                with col3:
                    st.write(row['dimensions'])
                
                with col4:
                    # Status badge
                    status = row['status']
                    if status == 'ready':
                        st.write("Ready")
                    elif status == 'processing':
                        st.write("Processing...")
                    elif status == 'done':
                        st.write("Done")
                    elif status == 'failed':
                        st.write("Failed")
                    elif status == 'skipped':
                        st.write("Skipped")
                
                with col5:
                    # Thumbnail preview
                    if row['thumbnail'] is not None:
                        st.image(row['thumbnail'], width=50)
            
            st.markdown("---")
            
            # Clear uploads button
            if st.button("Clear Uploads", type="secondary"):
                st.session_state.images = []
                st.rerun()
        
        st.info("Go to **MedSAM Analysis** tab to process your images")
        
    else:
        st.info("Please upload one or more images to get started")
        
        # If there are no uploaded files but images exist, clear them
        if st.session_state.images:
            st.session_state.images = []


def channels_page():
    """Channel inspection and configuration page"""
    st.title("Channels")
    
    st.markdown("""
    Preview individual color channels and configure which channels to use for MedSAM analysis.
    """)
    
    # Check if we have images
    if not st.session_state.images:
        st.warning("Please upload images first in the Image Upload tab")
        return
    
    st.markdown("---")
    
    # Display channel info for each image
    for i, item in enumerate(st.session_state.images):
        with st.expander(f"{i+1}. {item['name']}", expanded=(i == 0)):
            # Load image if not already loaded
            if 'np_rgb_uint8' not in item or item['np_rgb_uint8'] is None:
                image = load_image_from_bytes(item['bytes'])
                item['np_rgb_uint8'] = image
            else:
                image = item['np_rgb_uint8']
            
            # Initialize channel config if not present
            if 'channel_config' not in item:
                item['channel_config'] = {
                    'mode': 'rgb',
                    'channels': ['R', 'G', 'B']
                }
            
            # Channel previews
            st.write("**Channel Previews**")
            col1, col2, col3, col4 = st.columns(4)
            
            # Extract channels
            r_channel = image[:, :, 0]
            g_channel = image[:, :, 1]
            b_channel = image[:, :, 2]
            
            with col1:
                st.image(image, caption="RGB Composite", use_container_width=True)
            with col2:
                st.image(r_channel, caption="R Channel", use_container_width=True, clamp=True)
            with col3:
                st.image(g_channel, caption="G Channel", use_container_width=True, clamp=True)
            with col4:
                st.image(b_channel, caption="B Channel", use_container_width=True, clamp=True)
            
            st.markdown("---")
            
            # Channel mode selection
            st.write("**Channel Configuration**")
            
            mode = st.radio(
                "Channel Mode",
                options=['rgb', 'single', 'multi'],
                format_func=lambda x: {
                    'rgb': 'RGB Composite',
                    'single': 'Single Channel',
                    'multi': 'Multi-Channel'
                }[x],
                key=f"channel_mode_{i}",
                index=['rgb', 'single', 'multi'].index(item['channel_config']['mode']),
                horizontal=True
            )
            
            item['channel_config']['mode'] = mode
            
            # Channel selection for single/multi mode
            if mode in ['single', 'multi']:
                if mode == 'single':
                    st.write("Select a single channel:")
                    selected = st.radio(
                        "Channel",
                        options=['R', 'G', 'B'],
                        key=f"single_channel_{i}",
                        index=['R', 'G', 'B'].index(item['channel_config']['channels'][0]) if item['channel_config']['channels'] else 0,
                        horizontal=True
                    )
                    item['channel_config']['channels'] = [selected]
                else:  # multi
                    st.write("Select channels to process separately (will be merged):")
                    channels = []
                    col_r, col_g, col_b = st.columns(3)
                    with col_r:
                        if st.checkbox("R", value='R' in item['channel_config']['channels'], key=f"multi_r_{i}"):
                            channels.append('R')
                    with col_g:
                        if st.checkbox("G", value='G' in item['channel_config']['channels'], key=f"multi_g_{i}"):
                            channels.append('G')
                    with col_b:
                        if st.checkbox("B", value='B' in item['channel_config']['channels'], key=f"multi_b_{i}"):
                            channels.append('B')
                    
                    if channels:
                        item['channel_config']['channels'] = channels
                    else:
                        st.warning("Please select at least one channel")
                        item['channel_config']['channels'] = ['R']
            
            # Show preview of what will be fed to MedSAM
            st.markdown("---")
            st.write("**MedSAM Input Preview**")
            
            channel_inputs = prepare_channel_input(image, item['channel_config'])
            
            if len(channel_inputs) == 1:
                channel_img, channel_name = channel_inputs[0]
                st.image(channel_img, caption=f"Input: {channel_name}", use_container_width=True)
            else:
                cols = st.columns(min(len(channel_inputs), 4))
                for idx, (channel_img, channel_name) in enumerate(channel_inputs):
                    with cols[idx % 4]:
                        st.image(channel_img, caption=f"Input: {channel_name}", use_container_width=True)
            
            st.success(f"Configuration saved for {item['name']}")


def analysis_page():
    """MedSAM Analysis interface with segmentation - Multi-image queue support"""
    st.title("MedSAM Analysis")
    
    # Check if we're in local mode and have images in queue
    is_local_mode = st.session_state.local_mode
    
    if is_local_mode:
        # Local mode with multi-image queue
        if not st.session_state.images:
            st.warning("Please upload images first in the Image Upload tab")
            return
        
        st.info(f"Image Queue: {len(st.session_state.images)} image(s)")
        
        # Analysis settings (common for all images)
        st.subheader("Analysis Settings")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            prompt_mode = st.selectbox(
                "Prompt Mode",
                options=["auto_box", "full_box", "point"],
                index=0,
                help="auto_box: Auto-detect tissue region; full_box: Use entire image; point: Use center point"
            )
        
        with col_set2:
            multimask_output = st.checkbox(
                "Multi-mask Output",
                value=False,
                help="Generate multiple mask predictions and select the best one"
            )
        
        # Advanced settings in expander
        with st.expander("Advanced Segmentation Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                min_area_ratio = st.slider(
                    "Min Area Ratio",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Minimum area ratio for tissue detection (used in auto_box mode)"
                )
            with col_b:
                morph_kernel_size = st.slider(
                    "Morph Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2,
                    help="Kernel size for morphological operations (used in auto_box mode)"
                )
        
        # Multi-channel merge settings
        with st.expander("Multi-Channel Merge Settings"):
            st.write("These settings apply when multi-channel mode is selected in the Channels page")
            merge_mode = st.selectbox(
                "Merge Mode",
                options=['union', 'intersection', 'voting'],
                index=0,
                help="union: Any channel positive; intersection: All channels positive; voting: k-of-n channels"
            )
            
            if merge_mode == 'voting':
                k_value = st.slider(
                    "K Value (k-of-n)",
                    min_value=1,
                    max_value=3,
                    value=1,
                    help="Require at least k channels to be positive"
                )
            else:
                k_value = 1
        
        # Post-processing settings
        with st.expander("Post-Processing Settings"):
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                min_area_px = st.number_input(
                    "Min Area (pixels)",
                    min_value=0,
                    max_value=10000,
                    value=0,
                    step=10,
                    help="Remove objects smaller than this (in pixels)"
                )
                
                fill_holes = st.checkbox(
                    "Fill Holes",
                    value=False,
                    help="Fill holes in segmented objects"
                )
                
                watershed_split = st.checkbox(
                    "Watershed Split",
                    value=False,
                    help="Apply watershed for instance segmentation"
                )
            
            with col_p2:
                morph_open_radius = st.number_input(
                    "Morphological Open Radius",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                    help="Opening radius (0 = skip)"
                )
                
                morph_close_radius = st.number_input(
                    "Morphological Close Radius",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                    help="Closing radius (0 = skip)"
                )
                
                if watershed_split:
                    min_distance = st.number_input(
                        "Min Distance for Watershed",
                        min_value=1,
                        max_value=50,
                        value=10,
                        step=1,
                        help="Minimum distance for watershed seed detection"
                    )
                else:
                    min_distance = 10
        
        # TODO: Add refinement from prior JSON segmentation here
        # This would allow users to load existing segmentation and refine it
        
        st.markdown("---")
        
        # Control buttons
        st.subheader("Controls")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            run_next = st.button("Run Next", type="primary", disabled=st.session_state.batch_running)
        
        with col_btn2:
            run_batch = st.button("Run Batch", type="primary", disabled=st.session_state.batch_running)
        
        with col_btn3:
            if st.session_state.batch_running:
                stop_batch = st.button("Stop Batch", type="secondary")
            else:
                stop_batch = False
        
        # Handle button actions
        if run_next:
            # Find next ready, skipped, or done image that is included
            # This allows re-running with new parameters
            next_item = None
            for item in st.session_state.images:
                if item['include'] and item['status'] in ['ready', 'skipped', 'done']:
                    next_item = item
                    break
            
            if next_item:
                next_item['status'] = 'processing'
                st.rerun()
        
        if run_batch:
            st.session_state.batch_running = True
            st.session_state.batch_index = 0
            # Mark all included items as ready to allow re-run
            for item in st.session_state.images:
                if item['include'] and item['status'] in ['done', 'failed']:
                    item['status'] = 'ready'
            st.rerun()
        
        if stop_batch:
            st.session_state.batch_running = False
            # Mark any processing items as skipped
            for item in st.session_state.images:
                if item['status'] == 'processing':
                    item['status'] = 'skipped'
            st.rerun()
        
        # Batch processing logic
        if st.session_state.batch_running:
            # Find the next image to process
            processing_item = None
            for item in st.session_state.images:
                if item['include'] and item['status'] == 'processing':
                    processing_item = item
                    break
            
            if processing_item is None:
                # Find next ready item
                for item in st.session_state.images:
                    if item['include'] and item['status'] == 'ready':
                        item['status'] = 'processing'
                        processing_item = item
                        break
            
            if processing_item:
                # Process this item
                st.info(f"Processing: {processing_item['name']}")
                try:
                    # Build post-processing config
                    post_processing = {
                        'min_area_px': min_area_px,
                        'fill_holes': fill_holes,
                        'morph_open_radius': morph_open_radius,
                        'morph_close_radius': morph_close_radius,
                        'watershed_split': watershed_split,
                        'min_distance': min_distance
                    }
                    
                    # Build merge config
                    merge_config = {
                        'mode': merge_mode,
                        'k_value': k_value
                    }
                    
                    result = run_analysis_on_item(
                        processing_item,
                        prompt_mode=prompt_mode,
                        multimask_output=multimask_output,
                        min_area_ratio=min_area_ratio,
                        morph_kernel_size=morph_kernel_size,
                        post_processing=post_processing,
                        merge_config=merge_config
                    )
                    processing_item['result'] = result
                    processing_item['status'] = 'done'
                    st.success(f"Completed: {processing_item['name']}")
                except Exception as e:
                    processing_item['status'] = 'failed'
                    processing_item['error'] = str(e)
                    st.error(f"Failed: {processing_item['name']}: {str(e)}")
                
                # Check if there are more items to process
                has_more = any(item['include'] and item['status'] == 'ready' for item in st.session_state.images)
                if has_more:
                    # Continue batch
                    st.rerun()
                else:
                    # Batch complete
                    st.session_state.batch_running = False
                    st.success("Batch processing complete!")
                    st.rerun()
            else:
                # No more items to process
                st.session_state.batch_running = False
                st.success("Batch processing complete!")
                st.rerun()
        
        # Single item processing (Run Next button)
        if not st.session_state.batch_running:
            processing_item = None
            for item in st.session_state.images:
                if item['status'] == 'processing':
                    processing_item = item
                    break
            
            if processing_item:
                st.info(f"Processing: {processing_item['name']}")
                try:
                    # Build post-processing config
                    post_processing = {
                        'min_area_px': min_area_px,
                        'fill_holes': fill_holes,
                        'morph_open_radius': morph_open_radius,
                        'morph_close_radius': morph_close_radius,
                        'watershed_split': watershed_split,
                        'min_distance': min_distance
                    }
                    
                    # Build merge config
                    merge_config = {
                        'mode': merge_mode,
                        'k_value': k_value
                    }
                    
                    result = run_analysis_on_item(
                        processing_item,
                        prompt_mode=prompt_mode,
                        multimask_output=multimask_output,
                        min_area_ratio=min_area_ratio,
                        morph_kernel_size=morph_kernel_size,
                        post_processing=post_processing,
                        merge_config=merge_config
                    )
                    processing_item['result'] = result
                    processing_item['status'] = 'done'
                    st.success(f"Completed: {processing_item['name']}")
                except Exception as e:
                    processing_item['status'] = 'failed'
                    processing_item['error'] = str(e)
                    st.error(f"Failed: {processing_item['name']}: {str(e)}")
        
        st.markdown("---")
        
        # Display queue status
        st.subheader("Queue Status")
        
        for i, item in enumerate(st.session_state.images):
            with st.expander(f"{i+1}. {item['name']} - {item['status'].upper()}", expanded=(item['status'] in ['processing', 'done'])):
                if item['status'] == 'done' and item['result']:
                    # Display results
                    result = item['result']
                    
                    # Statistics
                    st.write("**Statistics**")
                    col1, col2, col3 = st.columns(3)
                    stats = result['statistics']
                    
                    with col1:
                        st.metric("Positive Pixels", f"{stats['num_positive_pixels']:,}")
                    with col2:
                        st.metric("Coverage", f"{stats['coverage_percent']:.2f}%")
                    with col3:
                        if 'area_mm2' in stats:
                            st.metric("Area", f"{stats['area_mm2']:.4f} mm²")
                    
                    # Visualizations
                    st.write("**Visualizations**")
                    
                    # Create binary mask for visualization - use post-processed mask if available
                    mask = result.get('binary_mask', result['mask'])
                    if mask.dtype == np.bool_:
                        mask_bin = mask
                    else:
                        unique_vals = np.unique(mask)
                        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                            mask_bin = mask.astype(bool)
                        else:
                            mask_bin = mask > 0.5
                    
                    mask_vis = (mask_bin.astype(np.uint8)) * 255
                    
                    if result.get('img_with_box') is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.image(result['image'], caption="Original", use_column_width="always")
                        with col2:
                            st.image(mask_vis, caption="Mask", clamp=True, use_column_width="always")
                        with col3:
                            st.image(result['overlay'], caption="Overlay", use_column_width="always")
                        with col4:
                            st.image(result['img_with_box'], caption="Prompt Box", use_column_width="always")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(result['image'], caption="Original", use_column_width="always")
                        with col2:
                            st.image(mask_vis, caption="Mask", clamp=True, use_column_width="always")
                        with col3:
                            st.image(result['overlay'], caption="Overlay", use_column_width="always")
                    
                    # Show instance segmentation if available
                    if result.get('instance_mask') is not None and result['instance_mask'] is not None:
                        st.write("**Instance Segmentation**")
                        st.write(f"Found {len(result.get('measurements', []))} objects")
                        
                        # Colorize instance mask for visualization
                        instance_mask = result['instance_mask']
                        if np.max(instance_mask) > 0:
                            import matplotlib.pyplot as plt
                            import matplotlib.colors as mcolors
                            
                            # Create a colormap
                            n_instances = np.max(instance_mask)
                            colors = plt.cm.tab20(np.linspace(0, 1, min(n_instances, 20)))
                            
                            # Create RGB visualization
                            instance_vis = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
                            for i in range(1, n_instances + 1):
                                color_idx = (i - 1) % len(colors)
                                mask_i = instance_mask == i
                                instance_vis[mask_i] = (colors[color_idx][:3] * 255).astype(np.uint8)
                            
                            st.image(instance_vis, caption="Instance Segmentation", use_column_width="always")
                    
                    # Show channel masks if multi-channel
                    if result.get('channel_masks'):
                        st.write("**Channel Masks**")
                        channel_masks = result['channel_masks']
                        cols = st.columns(len(channel_masks))
                        for idx, (channel_name, ch_mask) in enumerate(channel_masks.items()):
                            with cols[idx]:
                                ch_mask_vis = (ch_mask > 0).astype(np.uint8) * 255
                                st.image(ch_mask_vis, caption=f"{channel_name} Channel", use_column_width="always")
                    
                elif item['status'] == 'failed':
                    st.error(f"**Error:** {item['error']}")
                    if st.button(f"Retry {item['name']}", key=f"retry_{i}"):
                        item['status'] = 'ready'
                        item['error'] = None
                        st.rerun()
                
                elif item['status'] == 'processing':
                    st.info("Processing in progress...")
                
                elif item['status'] == 'ready':
                    st.info("Ready for processing")
                
                elif item['status'] == 'skipped':
                    st.warning("Skipped")
        
    else:
        # Original Halo mode logic (keep existing for backward compatibility)
        if st.session_state.selected_slide is None:
            st.warning("Please select a slide first")
            return
        
        slide = st.session_state.selected_slide
        st.info(f"Analyzing: **{slide['name']}**")
        
        # Check if in local mode or Halo mode
        is_local_mode = st.session_state.local_mode or slide['id'].startswith('local_')
        
        # ROI selection (only for Halo mode or if image is already loaded)
        if not is_local_mode or st.session_state.current_image is None:
            st.subheader("Region of Interest (ROI)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x = st.number_input("X coordinate (pixels)", min_value=0, max_value=slide['width'], value=0, step=100)
                width = st.number_input("Width (pixels)", min_value=1, max_value=slide['width'], value=1024, step=100)
            
            with col2:
                y = st.number_input("Y coordinate (pixels)", min_value=0, max_value=slide['height'], value=0, step=100)
                height = st.number_input("Height (pixels)", min_value=1, max_value=slide['height'], value=1024, step=100)
            
            # Validate ROI
            if x + width > slide['width']:
                st.error(f"ROI extends beyond slide width ({slide['width']} px)")
                return
            if y + height > slide['height']:
                st.error(f"ROI extends beyond slide height ({slide['height']} px)")
                return
        else:
            # For local mode with pre-loaded image, use full image
            x, y = 0, 0
            width, height = slide['width'], slide['height']
            st.info(f"Analyzing full image: {width} × {height} pixels")
        
        st.markdown("---")
        
        # Analysis settings
        st.subheader("Analysis Settings")
        
        # Segmentation prompt settings
        st.write("**Segmentation Prompt Mode**")
        prompt_mode = st.selectbox(
            "Prompt Mode",
            options=["auto_box", "full_box", "point"],
            index=0,
            help="auto_box: Auto-detect tissue region; full_box: Use entire image; point: Use center point"
        )
        
        # Advanced settings in expander
        with st.expander("Advanced Segmentation Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                min_area_ratio = st.slider(
                    "Min Area Ratio",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Minimum area ratio for tissue detection (used in auto_box mode)"
                )
                morph_kernel_size = st.slider(
                    "Morph Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2,
                    help="Kernel size for morphological operations (used in auto_box mode)"
                )
            with col_b:
                multimask_output = st.checkbox(
                    "Multi-mask Output",
                    value=False,
                    help="Generate multiple mask predictions and select the best one"
                )
        
        use_prompts = st.checkbox("Use point/box prompts", value=False, 
                                 help="Enable interactive prompts for segmentation")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Get image based on mode
                        if is_local_mode and st.session_state.current_image is not None:
                            # Use pre-loaded image from local upload
                            st.info("Using uploaded image...")
                            image = st.session_state.current_image
                            
                            # If ROI is specified and not full image, crop it
                            if x > 0 or y > 0 or width < slide['width'] or height < slide['height']:
                                image = image[y:y+height, x:x+width]
                        else:
                            # Download region from Halo API
                            st.info("Downloading region from Halo... This may take a moment for large regions.")
                            region_data = st.session_state.api.download_region(
                                slide['id'], x, y, width, height
                            )
                            
                            if not region_data:
                                st.error("Failed to download region - no data received")
                                return
                            
                            # Load image
                            st.info("Loading image...")
                            image = load_image_from_bytes(region_data)
                        
                        st.session_state.current_image = image
                        
                        # Initialize predictor if needed
                        if st.session_state.predictor is None:
                            st.info("Loading MedSAM model...")
                            st.session_state.predictor = UtilsMedSAMPredictor(
                                Config.MEDSAM_CHECKPOINT,
                                model_type=Config.MODEL_TYPE,
                                device=Config.DEVICE
                            )
                        
                        # Run inference directly on original image (no preprocessing)
                        st.info(f"Running MedSAM segmentation with {prompt_mode} prompt...")
                        
                        # Compute prompt box for visualization
                        prompt_box = None
                        if prompt_mode == "auto_box":
                            image_rgb = _ensure_rgb_uint8(image)
                            prompt_box = _compute_tissue_bbox(image_rgb, min_area_ratio, morph_kernel_size)
                            st.info(f"Detected tissue box: {prompt_box}")
                        elif prompt_mode == "full_box":
                            h, w = image.shape[:2]
                            prompt_box = np.array([0, 0, w - 1, h - 1])
                            st.info(f"Using full image box: {prompt_box}")
                        
                        mask = st.session_state.predictor.predict(
                            image,
                            prompt_mode=prompt_mode,
                            multimask_output=multimask_output,
                            min_area_ratio=min_area_ratio,
                            morph_kernel_size=morph_kernel_size
                        )
                        
                        # Store mask directly (already at original image size)
                        st.session_state.current_mask = mask
                        
                        # Compute statistics
                        mpp = slide.get('mpp')
                        stats = compute_mask_statistics(mask, mpp)
                        
                        # Store results
                        st.session_state.analysis_results = {
                            'image': image,
                            'mask': mask,
                            'roi': (x, y, width, height),
                            'statistics': stats,
                            'slide_id': slide['id'],
                            'slide_name': slide['name'],
                            'timestamp': datetime.now().isoformat(),
                            'prompt_box': prompt_box,
                            'prompt_mode': prompt_mode
                        }
                        
                        st.success("Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.code(traceback.format_exc())
        
        with col2:
            if st.button("Clear Results"):
                st.session_state.analysis_results = None
                st.session_state.current_image = None
                st.session_state.current_mask = None
                st.success("Cleared")
        
        # Display results
        if st.session_state.analysis_results is not None:
            st.markdown("---")
            st.subheader("Results")
            
            results = st.session_state.analysis_results
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            stats = results['statistics']
            
            with col1:
                st.metric("Positive Pixels", f"{stats['num_positive_pixels']:,}")
            with col2:
                st.metric("Coverage", f"{stats['coverage_percent']:.2f}%")
            with col3:
                if 'area_um2' in stats:
                    st.metric("Area", f"{stats['area_um2']:.2f} µm²")
            with col4:
                if 'area_mm2' in stats:
                    st.metric("Area", f"{stats['area_mm2']:.4f} mm²")
            
            # Visualizations
            st.markdown("### Visualization")
            
            # Debug information for mask
            mask = results['mask']
            st.write("**Debug Info:**")
            st.write(f"Prompt mode: {results.get('prompt_mode', 'unknown')}")
            if results.get('prompt_box') is not None:
                prompt_box = results['prompt_box']
                st.write(f"Prompt box: [{prompt_box[0]}, {prompt_box[1]}, {prompt_box[2]}, {prompt_box[3]}]")
                box_area = (prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1])
                img_area = mask.shape[0] * mask.shape[1]
                st.write(f"Prompt box area: {box_area:,} pixels ({100*box_area/img_area:.1f}% of image)")
            st.write(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")
            st.write(f"Mask min/max: {float(np.min(mask))}, {float(np.max(mask))}")
            
            # Create binary mask - handle both boolean and numeric masks
            # For boolean masks, use directly; for numeric, threshold at 0.5
            if mask.dtype == np.bool_:
                mask_bin = mask
            else:
                # Check if already binary (0/1) or needs thresholding
                unique_vals = np.unique(mask)
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                    mask_bin = mask.astype(bool)  # Already binary
                else:
                    mask_bin = mask > 0.5  # Threshold probabilistic output
            
            st.write(f"Binary mask unique values: {np.unique(mask_bin)}")
            st.write(f"Binary mask sum (positive pixels): {int(mask_bin.sum())}")
            
            # Create prompt box overlay for visualization if available
            if results.get('prompt_box') is not None:
                prompt_box = results['prompt_box']
                img_with_box = results['image'].copy()
                # Draw rectangle on image
                x1, y1, x2, y2 = prompt_box.astype(int)
                img_with_box = cv2.rectangle(img_with_box, (x1, y1), (x2, y2), 
                                            PROMPT_BOX_COLOR, PROMPT_BOX_THICKNESS)
                # Add text label
                label_y = max(y1 - PROMPT_BOX_LABEL_Y_OFFSET, PROMPT_BOX_LABEL_MIN_Y)
                cv2.putText(img_with_box, f"Prompt: {results.get('prompt_mode', 'box')}", 
                           (x1, label_y), PROMPT_BOX_LABEL_FONT, PROMPT_BOX_LABEL_SCALE,
                           PROMPT_BOX_COLOR, PROMPT_BOX_LABEL_THICKNESS)
            
            # Display images in columns
            if results.get('prompt_box') is not None:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(results['image'], caption="Original Image")
            
            with col2:
                # Display binary mask properly - convert to uint8 for visualization
                # Use explicit parentheses for clarity
                mask_vis = (mask_bin.astype(np.uint8)) * 255
                st.image(mask_vis, caption="Segmentation Mask (binary)", clamp=True)
            
            with col3:
                overlay = overlay_mask_on_image(
                    results['image'],
                    results['mask'],
                    color=(255, 0, 0),
                    alpha=0.5
                )
                st.image(overlay, caption="Overlay")
            
            if results.get('prompt_box') is not None:
                with col4:
                    st.image(img_with_box, caption="Prompt Box")


def export_page():
    """Export results to GeoJSON format"""
    st.title("Export Results")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results to export. Please run analysis first.")
        return
    
    results = st.session_state.analysis_results
    
    st.success(f"Results ready for export from: **{results['slide_name']}**")
    
    # Export settings
    st.subheader("Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        classification = st.text_input(
            "Classification Label",
            value="tissue_segmentation",
            help="Label for the segmented regions"
        )
        
        min_area = st.number_input(
            "Minimum Polygon Area (pixels)",
            min_value=1,
            value=Config.MIN_POLYGON_AREA,
            help="Filter out small polygons"
        )
    
    with col2:
        simplify = st.checkbox(
            "Simplify Polygons",
            value=True,
            help="Reduce polygon complexity"
        )
        
        if simplify:
            tolerance = st.slider(
                "Simplification Tolerance",
                min_value=0.1,
                max_value=5.0,
                value=Config.SIMPLIFY_TOLERANCE,
                help="Higher = more simplified"
            )
        else:
            tolerance = 1.0
    
    st.markdown("---")
    
    if st.button("Generate GeoJSON", type="primary", ):
        with st.spinner("Converting mask to GeoJSON..."):
            try:
                # Convert mask to polygons
                st.info("Extracting polygons from mask...")
                polygons = mask_to_polygons(results['mask'], min_area=min_area)
                
                if len(polygons) == 0:
                    st.warning("No polygons found. Try reducing minimum area.")
                    return
                
                # Create GeoJSON
                st.info(f"Creating GeoJSON with {len(polygons)} features...")
                geojson = polygons_to_geojson(
                    polygons,
                    properties={"classification": classification}
                )
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"annotations_{results['slide_name']}_{timestamp}.geojson"
                output_path = Config.get_temp_path(filename)
                
                with open(output_path, 'w') as f:
                    json.dump(geojson, f, indent=2)
                
                # Store in session state
                st.session_state.geojson = geojson
                st.session_state.geojson_path = output_path
                
                st.success(f"Exported {len(polygons)} polygons to GeoJSON!")
                
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                st.code(traceback.format_exc())
    
    # Display and download
    if 'geojson' in st.session_state:
        st.markdown("---")
        st.subheader("GeoJSON Preview")
        
        # Statistics
        num_features = len(st.session_state.geojson['features'])
        st.metric("Number of Features", num_features)
        
        # Preview
        with st.expander("View GeoJSON"):
            st.json(st.session_state.geojson)
        
        # Download button
        with open(st.session_state.geojson_path, 'r') as f:
            geojson_str = f.read()
        
        st.download_button(
            label="Download GeoJSON",
            data=geojson_str,
            file_name=st.session_state.geojson_path.name,
            mime="application/json"
        )
        
        st.info("""
        **Next Steps:**
        1. Download the GeoJSON file
        2. Open your slide in Halo (or view in GIS software)
        3. Import the GeoJSON as annotations
        4. Visualize and refine results
        """)


def tabulation_page():
    """Tabulation page showing summary results across all images"""
    st.title("Tabulation")
    
    st.markdown("""
    Summary of MedSAM analysis results across all processed images.
    """)
    
    # Check if we have any processed images
    processed_images = [item for item in st.session_state.images if item.get('status') == 'done' and item.get('result')]
    
    if not processed_images:
        st.warning("No processed images yet. Please run MedSAM Analysis first.")
        return
    
    st.info(f"Showing results for {len(processed_images)} processed image(s)")
    
    st.markdown("---")
    
    # Build summary table
    summary_data = []
    for item in processed_images:
        result = item['result']
        stats = result['statistics']
        
        row = {
            'Filename': item['name'],
            'Positive Pixels': stats['num_positive_pixels'],
            'Coverage (%)': f"{stats['coverage_percent']:.2f}",
            'Total Pixels': stats['total_pixels']
        }
        
        # Add instance info if available
        if result.get('measurements'):
            measurements = result['measurements']
            row['Object Count'] = len(measurements)
            if measurements:
                areas = [m['area'] for m in measurements]
                row['Mean Area (px)'] = f"{np.mean(areas):.1f}"
                row['Total Area (px)'] = sum(areas)
        else:
            row['Object Count'] = '-'
            row['Mean Area (px)'] = '-'
            row['Total Area (px)'] = '-'
        
        summary_data.append(row)
    
    # Display as DataFrame
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Download options
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Summary CSV",
            data=csv,
            file_name="medsam_summary.csv",
            mime="text/csv",
        )
    
    with col2:
        # Download all masks as ZIP
        if st.button("Prepare Mask Downloads"):
            with st.spinner("Preparing downloads..."):
                import zipfile
                from io import BytesIO
                
                # Create ZIP file in memory
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for item in processed_images:
                        result = item['result']
                        
                        # Add binary mask as PNG
                        mask = result.get('binary_mask', result['mask'])
                        mask_img = Image.fromarray(mask)
                        img_buffer = BytesIO()
                        mask_img.save(img_buffer, format='PNG')
                        zip_file.writestr(f"{item['name']}_mask.png", img_buffer.getvalue())
                        
                        # Add instance mask as TIFF if available
                        if result.get('instance_mask') is not None and result['instance_mask'] is not None:
                            instance_mask = result['instance_mask']
                            instance_img = Image.fromarray(instance_mask.astype(np.int32))
                            tiff_buffer = BytesIO()
                            instance_img.save(tiff_buffer, format='TIFF')
                            zip_file.writestr(f"{item['name']}_instances.tif", tiff_buffer.getvalue())
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="Download All Masks (ZIP)",
                    data=zip_buffer,
                    file_name="medsam_masks.zip",
                    mime="application/zip",
                )
    
    st.markdown("---")
    
    # Individual image downloads
    st.subheader("Individual Image Downloads")
    
    for i, item in enumerate(processed_images):
        with st.expander(f"{i+1}. {item['name']}"):
            result = item['result']
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Download mask PNG
                mask = result.get('binary_mask', result['mask'])
                mask_img = Image.fromarray(mask)
                img_buffer = BytesIO()
                mask_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="Download Mask PNG",
                    data=img_buffer,
                    file_name=f"{item['name']}_mask.png",
                    mime="image/png",
                    key=f"mask_png_{i}"
                )
            
            with col_b:
                # Download instance mask TIFF if available
                if result.get('instance_mask') is not None and result['instance_mask'] is not None:
                    instance_mask = result['instance_mask']
                    instance_img = Image.fromarray(instance_mask.astype(np.int32))
                    tiff_buffer = BytesIO()
                    instance_img.save(tiff_buffer, format='TIFF')
                    tiff_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Instance TIFF",
                        data=tiff_buffer,
                        file_name=f"{item['name']}_instances.tif",
                        mime="image/tiff",
                        key=f"instance_tiff_{i}"
                    )
                else:
                    st.info("No instance segmentation available")


def import_page():
    """Import annotations to Halo (optional feature)"""
    st.title("Import to Halo")
    
    st.info("This feature requires additional Halo API permissions")
    
    st.markdown("""
    ### Manual Import Instructions
    
    1. **Download GeoJSON**from the Export page
    2. **Open Halo**and navigate to your slide
    3. **Import Annotations**: 
       - File → Import → GeoJSON
       - Select the downloaded file
    4. **Review and Save**the imported annotations
    
    ### Programmatic Import (Coming Soon)
    
    Automatic upload of annotations via API will be available in a future release.
    """)


def main():
    """Main application"""
    st.sidebar.title("XHaloPathAnalyzer")
    st.sidebar.markdown("---")
    
    if not st.session_state.authenticated:
        # Show only authentication
        authentication_page()
    else:
        # Show connection status
        if st.session_state.local_mode:
            st.sidebar.success("Local Mode Active")
        else:
            st.sidebar.success("Connected to Halo")
        
        # Determine navigation options based on mode
        if st.session_state.local_mode:
            nav_options = [
                "Image Upload",
                "Channels",
                "MedSAM Analysis",
                "Tabulation",
                "Export",
                "Settings"
            ]
        else:
            nav_options = [
                "Slide Selection",
                "MedSAM Analysis",
                "Tabulation",
                "Export",
                "Import",
                "Settings"
            ]
        
        page = st.sidebar.radio("Navigation", nav_options)
        
        st.sidebar.markdown("---")
        
        # Show current slide/image info
        if st.session_state.selected_slide:
            st.sidebar.info(f"**Current {'Image' if st.session_state.local_mode else 'Slide'}:**\n{st.session_state.selected_slide['name']}")
        
        # Logout/Exit button
        if st.sidebar.button("Exit to Start"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Route to pages
        if page == "Image Upload":
            image_upload_page()
        elif page == "Slide Selection":
            slide_selection_page()
        elif page == "Channels":
            channels_page()
        elif page == "MedSAM Analysis":
            analysis_page()
        elif page == "Tabulation":
            tabulation_page()
        elif page == "Export" or page == "Export":
            export_page()
        elif page == "Import":
            import_page()
        elif page == "Settings":
            st.title("Settings")
            st.info("Configuration settings coming soon")
            st.write(f"**Device:** {Config.DEVICE}")
            st.write(f"**Model:** {Config.MODEL_TYPE}")
            st.write(f"**Checkpoint:** {Config.MEDSAM_CHECKPOINT}")
            st.write(f"**Mode:** {'Local' if st.session_state.local_mode else 'Halo API'}")
            
            # Halo Link Debug Section
            st.markdown("---")
            st.subheader("Halo Link Integration")
            
            if st.button("Run Halo Link Smoke Test"):
                with st.spinner("Running Halo Link smoke test..."):
                    try:
                        from xhalo.halolink.smoketest import run_smoke_test
                        
                        # Run smoke test
                        results = run_smoke_test(verbose=False)
                        
                        if results["success"]:
                            st.success("Halo Link smoke test passed!")
                        else:
                            st.error(f"Halo Link smoke test failed: {results.get('error', 'Unknown error')}")
                        
                        # Display step-by-step results
                        st.markdown("#### Test Results")
                        for step_name, step_result in results.get("steps", {}).items():
                            if step_result.get("success"):
                                if step_result.get("skipped"):
                                    st.info(f"{step_name}: {step_result.get('reason', 'Skipped')}")
                                else:
                                    st.success(f" {step_name}")
                            else:
                                st.error(f" {step_name}: {step_result.get('error', 'Failed')}")
                        
                        # Show detailed results in expander
                        with st.expander("View Detailed Results"):
                            st.json(results)
                            
                    except ImportError as e:
                        st.error(f"Halo Link module not available: {e}")
                    except Exception as e:
                        st.error(f"Error running smoke test: {e}")
                        logger.exception("Halo Link smoke test error")
            
            # Show current configuration
            st.markdown("#### Current Configuration")
            halolink_config = {
                "HALOLINK_BASE_URL": Config.HALOLINK_BASE_URL or "(not set)",
                "HALOLINK_GRAPHQL_URL": Config.HALOLINK_GRAPHQL_URL or "(not set)",
                "HALOLINK_GRAPHQL_PATH": Config.HALOLINK_GRAPHQL_PATH or "(not set)",
                "HALOLINK_CLIENT_ID": "***" if Config.HALOLINK_CLIENT_ID else "(not set)",
                "HALOLINK_CLIENT_SECRET": "***" if Config.HALOLINK_CLIENT_SECRET else "(not set)",
                "HALOLINK_SCOPE": Config.HALOLINK_SCOPE or "(not set)",
            }
            
            for key, value in halolink_config.items():
                st.text(f"{key}: {value}")



if __name__ == "__main__":
    main()
