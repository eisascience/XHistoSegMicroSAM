# Comprehensive Guide: Building a Web-Based GUI for Halo Digital Pathology Image Analysis

## Table of Contents
1. [Introduction](#1-introduction)
2. [Architecture](#2-architecture)
3. [Setup Instructions](#3-setup-instructions)
4. [Core Code Implementation](#4-core-code-implementation)
5. [Advanced Features](#5-advanced-features)
6. [Testing and Debugging](#6-testing-and-debugging)
7. [Deployment Guide](#7-deployment-guide)
8. [Limitations and Extensions](#8-limitations-and-extensions)
9. [Full Example Workflow](#9-full-example-workflow)

---

## 1. Introduction

### Overview
This guide provides a complete, step-by-step approach to building an **OS-agnostic, web-based GUI application** for custom image analysis on microscopy and whole-slide images (WSIs) stored in the **Halo digital pathology platform** (from Indica Labs). The application treats Halo as both a data source and visualization layer while performing advanced analysis externally in Python.

### Key Features
- **Programmatic data retrieval**: Fetch WSI data and metadata from Halo using its GraphQL API
- **External analysis**: Perform segmentation with models like SAM or MedSAM in Python
- **Result integration**: Push analysis results (masks, annotations) back to Halo
- **User-friendly GUI**: Streamlit-based interface for complete workflows
- **Workflow steps**: Authentication, slide selection, export, analysis, visualization, and import

### Architecture Diagram (Text-Based)
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web GUI                         │
│  ┌──────────┬──────────┬──────────┬────────────┬──────────┐ │
│  │  Auth/   │  Slide   │  Export  │  Analysis  │  Import/ │ │
│  │ Config   │ Selection│   Data   │ (MedSAM)   │   View   │ │
│  └──────────┴──────────┴──────────┴────────────┴──────────┘ │
└─────────────────────────────────────────────────────────────┘
                          
┌─────────────────────────────────────────────────────────────┐
│                  Python Backend Functions                    │
│  ┌──────────────┬──────────────────┬──────────────────────┐ │
│  │ GraphQL API  │ Image Processing │ ML Model Inference   │ │
│  │  Interface   │ (large_image)    │ (MedSAM/SAM)        │ │
│  └──────────────┴──────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          
┌─────────────────────────────────────────────────────────────┐
│              Halo Digital Pathology Platform                 │
│     (GraphQL API @ /graphql, API Token Authentication)       │
│  • Whole-Slide Images (WSI) Storage                         │
│  • Metadata & Annotations                                    │
│  • Visualization & Annotation Tools                          │
└─────────────────────────────────────────────────────────────┘
```

### Prerequisites

#### System Requirements
- **Operating System**: macOS, Linux, or Windows (OS-agnostic)
- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB+ recommended for large WSIs)
- **GPU**: Optional but recommended for ML inference (CUDA-compatible NVIDIA GPU)

#### Installation Commands
```bash
# Install Python 3.10+ (if not already installed)
# macOS (using Homebrew)
brew install python@3.10

# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv

# Windows
# Download from https://www.python.org/downloads/

# Verify installation
python3 --version  # Should be 3.10 or higher
```

---

## 2. Architecture

### High-Level Components

#### 2.1 Streamlit Application Structure
The application follows a modular architecture:

```
XHaloPathAnalyzer/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Template for environment variables
├── .gitignore            # Git ignore patterns
├── utils/
│   ├── __init__.py
│   ├── halo_api.py       # Halo GraphQL API functions
│   ├── image_proc.py     # Image processing utilities
│   ├── ml_models.py      # ML model integration (MedSAM)
│   └── geojson_utils.py  # GeoJSON import/export
├── models/               # Directory for model weights
│   └── medsam_vit_b.pth
└── temp/                 # Temporary file storage
```

#### 2.2 Backend Functions
- **API Queries**: GraphQL queries for fetching slides, metadata, and annotations
- **Export Functions**: Download WSI tiles or ROIs as TIFF/OME-TIFF
- **Analysis Functions**: Run MedSAM or custom models on exported images
- **Import Functions**: Convert analysis results to GeoJSON and push to Halo

#### 2.3 Data Flow
1. **Authentication**: User provides Halo API endpoint and token
2. **Discovery**: Query available slides via GraphQL
3. **Selection**: User selects slides and ROIs for analysis
4. **Export**: Download image data using Halo API
5. **Analysis**: Run MedSAM segmentation on exported images
6. **Visualization**: Display results in Streamlit (preview masks)
7. **Import**: Convert masks to GeoJSON and upload to Halo

---

## 3. Setup Instructions

### 3.1 Environment Setup

#### Create Virtual Environment
```bash
# Navigate to project directory
cd XHaloPathAnalyzer

# Option A: Using uv (Recommended, especially for Mac M2/ARM)
# Install uv first if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# Or: brew install uv  # macOS via Homebrew

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# Or: .venv\Scripts\activate  # Windows

# Install dependencies with uv (much faster!)
uv pip install -r requirements.txt

# Option B: Using traditional Python venv
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Mac M2/ARM Users:** Use uv for best compatibility with ARM architecture. It automatically handles ARM64 dependencies including PyTorch.

### 3.2 Library Installation

Create `requirements.txt`:
```txt
# Web Framework
streamlit>=1.28.0

# GraphQL API
gql[all]>=3.4.0
aiohttp>=3.9.0

# Image Processing
large-image[all]>=1.25.0
openslide-python>=1.3.0
tifffile>=2023.7.0
scikit-image>=0.21.0
numpy>=1.24.0
pillow>=10.0.0

# ML Models
torch>=2.0.0
torchvision>=0.15.0
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git

# Utilities
requests>=2.31.0
python-dotenv>=1.0.0
pandas>=2.0.0
matplotlib>=3.7.0
```

**Note**: For MedSAM, you'll need to clone the repository separately (see below).

### 3.3 MedSAM Repository Setup

```bash
# Clone MedSAM repository
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM

# Download pre-trained weights
mkdir -p ../models
wget -P ../models https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1

# Return to project directory
cd ..
```

### 3.3.1 Patch Segment Anything for CPU Compatibility

After installing dependencies, apply the patch to ensure CUDA checkpoints load on CPU machines:

```bash
python patch_segment_anything.py
```

This modifies `segment_anything/build_sam.py` to add `map_location="cpu"` to torch.load() calls.

### 3.4 Environment Variable Configuration

Create `.env` file:
```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your preferred editor
```

`.env.example` contents:
```bash
# Halo API Configuration
HALO_API_ENDPOINT=https://your-halo-instance.com/graphql
HALO_API_TOKEN=your_api_token_here

# Model Configuration
MEDSAM_CHECKPOINT=./models/medsam_vit_b.pth

# Application Settings
MAX_IMAGE_SIZE_MB=500
TEMP_DIR=./temp
LOG_LEVEL=INFO
```

### 3.5 OpenSlide Installation (OS-Specific)

OpenSlide is required for processing whole-slide images.

**macOS (including M2/ARM)**:
```bash
brew install openslide
```

**Ubuntu/Debian**:
```bash
sudo apt-get install openslide-tools python3-openslide
```

**Windows**:
Download pre-built binaries from: https://openslide.org/download/

**Note for Mac M2/ARM users**: Homebrew handles ARM architecture automatically. The openslide formula is fully compatible with Apple Silicon.

---

## 4. Core Code Implementation

### 4.1 Configuration Management (config.py)

The configuration module centralizes all application settings and provides validation for environment variables:

```python
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Application configuration management"""
    
    # Halo API settings
    HALO_API_ENDPOINT = os.getenv("HALO_API_ENDPOINT", "")
    HALO_API_TOKEN = os.getenv("HALO_API_TOKEN", "")
    
    # Model settings
    MEDSAM_CHECKPOINT = os.getenv("MEDSAM_CHECKPOINT", "./models/medsam_vit_b.pth")
    
    # Application settings
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "500"))
    TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.HALO_API_ENDPOINT:
            errors.append("HALO_API_ENDPOINT is required")
        if not cls.HALO_API_TOKEN:
            errors.append("HALO_API_TOKEN is required")
            
        if errors:
            raise ValueError("Configuration errors: " + ", ".join(errors))
        
        # Create temp directory if it doesn't exist
        Path(cls.TEMP_DIR).mkdir(parents=True, exist_ok=True)
```

### 4.2 Halo API Integration (utils/halo_api.py)

This module provides a clean interface to Halo's GraphQL API:

```python
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import requests
from typing import List, Dict, Optional

class HaloAPI:
    """Interface for Halo GraphQL API"""
    
    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        
        # Setup GraphQL client
        transport = AIOHTTPTransport(
            url=endpoint,
            headers=self.headers
        )
        self.client = Client(transport=transport, fetch_schema_from_transport=True)
    
    async def get_slides(self, limit: int = 100) -> List[Dict]:
        """Fetch list of available slides"""
        query = gql('''
            query GetSlides($limit: Int!) {
                slides(first: $limit) {
                    edges {
                        node {
                            id
                            name
                            width
                            height
                            mpp
                            studyId
                            createdAt
                        }
                    }
                }
            }
        ''')
        
        result = await self.client.execute_async(query, variable_values={"limit": limit})
        return [edge['node'] for edge in result['slides']['edges']]
    
    async def get_slide_metadata(self, slide_id: str) -> Dict:
        """Fetch detailed metadata for a specific slide"""
        query = gql('''
            query GetSlide($id: ID!) {
                slide(id: $id) {
                    id
                    name
                    width
                    height
                    mpp
                    tileSize
                    format
                    metadata
                }
            }
        ''')
        
        result = await self.client.execute_async(query, variable_values={"id": slide_id})
        return result['slide']
    
    def download_region(self, slide_id: str, x: int, y: int, 
                       width: int, height: int, level: int = 0) -> bytes:
        """Download a specific region from a slide"""
        url = f"{self.endpoint.replace('/graphql', '')}/slides/{slide_id}/region"
        params = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "level": level
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.content
```

### 4.3 Image Processing (utils/image_proc.py)

Utilities for handling WSI images:

```python
import numpy as np
from PIL import Image
import large_image
from typing import Tuple, Optional
import io

def load_image_region(image_data: bytes) -> np.ndarray:
    """Load image data into numpy array"""
    img = Image.open(io.BytesIO(image_data))
    return np.array(img)

def preprocess_for_medsam(image: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """Preprocess image for MedSAM inference"""
    # Resize to target size while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    
    # Pad to square
    h, w = image.shape[:2]
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2
    
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    return padded

def postprocess_mask(mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """Resize mask back to original image dimensions"""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_img.resize((original_shape[1], original_shape[0]), Image.NEAREST)
    return np.array(mask_resized) > 0
```

### 4.4 ML Model Integration (utils/ml_models.py)

Integration with MedSAM for segmentation:

```python
import torch
import numpy as np
from segment_anything import sam_model_registry
from typing import Tuple, Optional

class MedSAMPredictor:
    """Wrapper for MedSAM model inference"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
    
    def _load_model(self, checkpoint_path: str):
        """Load MedSAM model from checkpoint"""
        model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        model.to(self.device)
        return model
    
    def predict(self, image: np.ndarray, 
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run inference on image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            point_coords: Point prompts (N, 2) in (x, y) format
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt (x1, y1, x2, y2)
            
        Returns:
            Binary mask as numpy array
        """
        # Prepare image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            image_embedding = self.model.image_encoder(image_tensor)
        
        # Prepare prompts
        prompt_points = None
        prompt_labels = None
        prompt_box = None
        
        if point_coords is not None:
            prompt_points = torch.from_numpy(point_coords).float().to(self.device)
            prompt_labels = torch.from_numpy(point_labels).float().to(self.device)
        
        if box is not None:
            prompt_box = torch.from_numpy(box).float().to(self.device)
        
        # Decode mask
        with torch.no_grad():
            masks, scores, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                point_coords=prompt_points,
                point_labels=prompt_labels,
                boxes=prompt_box
            )
        
        # Return best mask
        best_mask_idx = scores.argmax()
        mask = masks[0, best_mask_idx].cpu().numpy()
        return mask > 0.5
```

### 4.5 GeoJSON Utilities (utils/geojson_utils.py)

Convert segmentation masks to GeoJSON format for Halo:

```python
import numpy as np
from skimage import measure
from typing import List, Dict
import json

def mask_to_polygons(mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """Convert binary mask to list of polygon contours"""
    contours = measure.find_contours(mask, 0.5)
    
    # Filter by area
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            area = polygon_area(contour)
            if area >= min_area:
                polygons.append(contour)
    
    return polygons

def polygon_area(polygon: np.ndarray) -> float:
    """Calculate polygon area using shoelace formula"""
    x = polygon[:, 1]
    y = polygon[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygons_to_geojson(polygons: List[np.ndarray], 
                        properties: Dict = None) -> Dict:
    """Convert polygons to GeoJSON FeatureCollection"""
    features = []
    
    for idx, polygon in enumerate(polygons):
        # Convert to [x, y] format
        coords = [[float(point[1]), float(point[0])] for point in polygon]
        coords.append(coords[0])  # Close polygon
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "id": idx,
                "classification": properties.get("classification", "detected_object") if properties else "detected_object"
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def save_geojson(geojson: Dict, filepath: str):
    """Save GeoJSON to file"""
    with open(filepath, 'w') as f:
        json.dump(geojson, f, indent=2)
```

---

## 5. Advanced Features

### 5.1 Streamlit Application (app.py)

The main application provides a multi-page interface:

```python
import streamlit as st
import asyncio
from config import Config
from utils.halo_api import HaloAPI
from utils.image_proc import *
from utils.ml_models import MedSAMPredictor
from utils.geojson_utils import *

# Page configuration
st.set_page_config(
    page_title="XHaloPathAnalyzer",
    page_icon="",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api' not in st.session_state:
    st.session_state.api = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None

def authentication_page():
    """Authentication and configuration page"""
    st.title(" Authentication")
    st.write("Configure your Halo API connection")
    
    endpoint = st.text_input("Halo API Endpoint", value=Config.HALO_API_ENDPOINT)
    token = st.text_input("API Token", value=Config.HALO_API_TOKEN, type="password")
    
    if st.button("Connect"):
        try:
            # Test connection
            api = HaloAPI(endpoint, token)
            st.session_state.api = api
            st.session_state.authenticated = True
            st.success(" Connected successfully!")
            st.rerun()
        except Exception as e:
            st.error(f" Connection failed: {str(e)}")

def slide_selection_page():
    """Slide selection interface"""
    st.title(" Slide Selection")
    
    if st.session_state.api is None:
        st.warning("Please authenticate first")
        return
    
    # Fetch slides
    with st.spinner("Loading slides..."):
        slides = asyncio.run(st.session_state.api.get_slides())
    
    if not slides:
        st.warning("No slides found")
        return
    
    # Display slides in table
    import pandas as pd
    df = pd.DataFrame(slides)
    selected_idx = st.selectbox("Select a slide", range(len(df)), 
                                format_func=lambda i: df.iloc[i]['name'])
    
    if selected_idx is not None:
        st.session_state.selected_slide = slides[selected_idx]
        st.json(slides[selected_idx])

def analysis_page():
    """Analysis interface with MedSAM"""
    st.title(" Analysis")
    
    if 'selected_slide' not in st.session_state:
        st.warning("Please select a slide first")
        return
    
    slide = st.session_state.selected_slide
    st.write(f"Analyzing: **{slide['name']}**")
    
    # ROI selection
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("X coordinate", min_value=0, value=0)
        y = st.number_input("Y coordinate", min_value=0, value=0)
    with col2:
        width = st.number_input("Width", min_value=1, value=1024)
        height = st.number_input("Height", min_value=1, value=1024)
    
    if st.button("Run Analysis"):
        with st.spinner("Processing..."):
            try:
                # Download region
                region_data = st.session_state.api.download_region(
                    slide['id'], x, y, width, height
                )
                image = load_image_region(region_data)
                
                # Initialize predictor if needed
                if st.session_state.predictor is None:
                    st.session_state.predictor = MedSAMPredictor(
                        Config.MEDSAM_CHECKPOINT,
                        device=Config.DEVICE
                    )
                
                # Preprocess
                preprocessed = preprocess_for_medsam(image)
                
                # Run inference (automatic mode - no prompts)
                mask = st.session_state.predictor.predict(preprocessed)
                
                # Postprocess
                final_mask = postprocess_mask(mask, image.shape[:2])
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                with col2:
                    st.image(final_mask, caption="Segmentation", use_column_width=True)
                
                # Store results
                st.session_state.analysis_results = {
                    'image': image,
                    'mask': final_mask,
                    'roi': (x, y, width, height)
                }
                
                st.success(" Analysis complete!")
                
            except Exception as e:
                st.error(f" Analysis failed: {str(e)}")

def export_page():
    """Export results to GeoJSON"""
    st.title(" Export Results")
    
    if 'analysis_results' not in st.session_state:
        st.warning("No analysis results to export")
        return
    
    results = st.session_state.analysis_results
    
    classification = st.text_input("Classification label", value="detected_cells")
    
    if st.button("Generate GeoJSON"):
        with st.spinner("Converting to GeoJSON..."):
            # Convert mask to polygons
            polygons = mask_to_polygons(results['mask'])
            
            # Create GeoJSON
            geojson = polygons_to_geojson(
                polygons,
                properties={"classification": classification}
            )
            
            # Save to file
            output_path = f"{Config.TEMP_DIR}/annotations.geojson"
            save_geojson(geojson, output_path)
            
            st.success(f" Exported {len(polygons)} objects")
            st.json(geojson)
            
            # Download button
            with open(output_path, 'r') as f:
                st.download_button(
                    "Download GeoJSON",
                    f.read(),
                    file_name="annotations.geojson",
                    mime="application/json"
                )

# Main app navigation
def main():
    st.sidebar.title("Navigation")
    
    if not st.session_state.authenticated:
        authentication_page()
    else:
        page = st.sidebar.radio(
            "Go to",
            ["Slide Selection", "Analysis", "Export"]
        )
        
        if page == "Slide Selection":
            slide_selection_page()
        elif page == "Analysis":
            analysis_page()
        elif page == "Export":
            export_page()

if __name__ == "__main__":
    main()
```

### 5.2 Batch Processing

For processing multiple slides, implement batch functionality:

```python
def batch_analysis(slide_ids: List[str], roi_config: Dict):
    """Process multiple slides in batch"""
    results = []
    
    for slide_id in slide_ids:
        try:
            # Download and process
            region_data = api.download_region(slide_id, **roi_config)
            image = load_image_region(region_data)
            
            # Run analysis
            preprocessed = preprocess_for_medsam(image)
            mask = predictor.predict(preprocessed)
            
            # Convert to GeoJSON
            polygons = mask_to_polygons(mask)
            geojson = polygons_to_geojson(polygons)
            
            results.append({
                'slide_id': slide_id,
                'geojson': geojson,
                'object_count': len(polygons)
            })
            
        except Exception as e:
            st.error(f"Failed to process {slide_id}: {str(e)}")
            continue
    
    return results
```

### 5.3 Custom Model Integration

To integrate custom models instead of MedSAM:

```python
class CustomModelPredictor:
    """Template for custom model integration"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = self.load_custom_model(model_path)
    
    def load_custom_model(self, path: str):
        """Load your custom model"""
        # Implement your model loading logic
        pass
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on image"""
        # Implement your inference logic
        pass
```

---

## 6. Testing and Debugging

### 6.1 Unit Testing

Create `tests/test_halo_api.py`:

```python
import pytest
from utils.halo_api import HaloAPI

@pytest.fixture
def api():
    return HaloAPI("http://test-endpoint.com/graphql", "test-token")

def test_get_slides(api):
    """Test slide retrieval"""
    slides = asyncio.run(api.get_slides(limit=10))
    assert isinstance(slides, list)

def test_download_region(api):
    """Test region download"""
    data = api.download_region("slide-123", 0, 0, 1024, 1024)
    assert isinstance(data, bytes)
    assert len(data) > 0
```

### 6.2 Integration Testing

Test the complete workflow:

```python
def test_complete_workflow():
    """Test end-to-end workflow"""
    # 1. Authenticate
    api = HaloAPI(endpoint, token)
    
    # 2. Get slides
    slides = asyncio.run(api.get_slides())
    assert len(slides) > 0
    
    # 3. Download region
    region_data = api.download_region(slides[0]['id'], 0, 0, 1024, 1024)
    image = load_image_region(region_data)
    
    # 4. Run analysis
    predictor = MedSAMPredictor(checkpoint_path)
    mask = predictor.predict(preprocess_for_medsam(image))
    
    # 5. Export to GeoJSON
    polygons = mask_to_polygons(mask)
    geojson = polygons_to_geojson(polygons)
    
    assert len(geojson['features']) > 0
```

### 6.3 Debugging Tips

**Common Issues**:

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or image resolution
   torch.cuda.empty_cache()
   ```

2. **GraphQL Connection Errors**
   ```python
   # Add retry logic
   from tenacity import retry, stop_after_attempt
   
   @retry(stop=stop_after_attempt(3))
   async def get_slides_with_retry(api):
       return await api.get_slides()
   ```

3. **Image Loading Failures**
   ```python
   # Add error handling
   try:
       image = load_image_region(data)
   except Exception as e:
       logger.error(f"Failed to load image: {str(e)}")
       # Fallback logic
   ```

---

## 7. Deployment Guide

### 7.1 Local Deployment

Run the application locally:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py --server.port 8501
```

Access at: `http://localhost:8501`

### 7.2 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libopens


## 4. Core Code Implementation

The core implementation consists of five main modules that work together to provide the complete functionality.

### 4.1 Configuration Management (config.py)

The configuration module provides centralized management of all application settings. It loads configuration from environment variables, validates required settings, and provides convenient access throughout the application.

**Key Features:**
- Automatic device detection (CUDA vs CPU)
- Environment variable loading with sensible defaults
- Configuration validation on startup
- Path management for models and temporary files
- Logging configuration

**Usage Example:**
```python
from config import Config

# Validate configuration (call at startup)
Config.validate()

# Access settings
device = Config.DEVICE  # 'cuda' or 'cpu'
checkpoint = Config.MEDSAM_CHECKPOINT
temp_dir = Config.TEMP_DIR

# Get temporary file path
output_path = Config.get_temp_path("results.json")
```

The Config class uses class methods to provide singleton-like behavior, ensuring consistent configuration access across all modules.

### 4.2 Halo API Integration (utils/halo_api.py)

The HaloAPI class encapsulates all interactions with the Halo digital pathology platform's GraphQL API. It provides async methods for querying slides, downloading image regions, and managing annotations.

**Core Functionality:**
- GraphQL client initialization with authentication
- Slide querying and metadata retrieval
- Image region downloading via REST API
- Annotation upload/download
- Error handling and retry logic

**Implementation Highlights:**
The class uses the `gql` library for GraphQL queries and `requests` for REST API calls. All API methods include comprehensive error handling and logging.

```python
from utils.halo_api import HaloAPI
import asyncio

# Initialize client
api = HaloAPI(endpoint="https://halo.example.com/graphql", token="your_token")

# Test connection
if asyncio.run(api.test_connection()):
    print("Connected successfully")

# Fetch slides
slides = asyncio.run(api.get_slides(limit=100))
for slide in slides:
    print(f"{slide['name']}: {slide['width']}x{slide['height']}")

# Download region
region_bytes = api.download_region(
    slide_id=slides[0]['id'],
    x=1000, y=2000,
    width=2048, height=2048,
    level=0
)
```

The async/await pattern ensures non-blocking I/O, allowing the application to remain responsive during network operations.

### 4.3 Image Processing Utilities (utils/image_proc.py)

This module provides comprehensive image processing functions for loading, preprocessing, and postprocessing images throughout the analysis pipeline.

**Key Functions:**
- `load_image_from_bytes`: Convert image data to numpy arrays
- `preprocess_for_medsam`: Prepare images for MedSAM inference
- `postprocess_mask`: Resize masks back to original dimensions
- `overlay_mask_on_image`: Create visualization overlays
- `compute_mask_statistics`: Calculate area, coverage, and other metrics

**Preprocessing Pipeline:**
1. Load image and convert to RGB
2. Resize to target size (default 1024px) maintaining aspect ratio
3. Pad to square dimensions
4. Store metadata for later postprocessing

**Postprocessing Pipeline:**
1. Remove padding from mask
2. Resize to original image dimensions
3. Apply threshold to create binary mask

```python
from utils.image_proc import *

# Load and preprocess
image = load_image_from_bytes(region_bytes)
preprocessed, metadata = preprocess_for_medsam(image, target_size=1024)

# After inference...
final_mask = postprocess_mask(predicted_mask, metadata)

# Compute statistics
stats = compute_mask_statistics(final_mask, mpp=0.25)
print(f"Coverage: {stats['coverage_percent']:.2f}%")
print(f"Area: {stats['area_mm2']:.4f} mm²")
```

### 4.4 ML Model Integration (utils/ml_models.py)

The MedSAMPredictor class wraps the MedSAM segmentation model, providing a clean interface for inference.

**Features:**
- Automatic model loading and device placement
- Support for point and box prompts
- Default center-point prompting for automatic segmentation
- Batch processing capabilities
- Memory-efficient inference with torch.no_grad()

**Implementation Details:**
The predictor uses the Segment Anything Model (SAM) architecture fine-tuned for medical imaging. The model consists of an image encoder (Vision Transformer), prompt encoder, and mask decoder.

```python
from utils.ml_models import MedSAMPredictor
import numpy as np

# Initialize predictor (load model once)
predictor = MedSAMPredictor(
    checkpoint_path="./models/medsam_vit_b.pth",
    model_type="vit_b",
    device="cuda"  # or "cpu"
)

# Automatic segmentation (uses center point)
mask = predictor.predict(preprocessed_image)

# With point prompts
points = np.array([[512, 512], [600, 600]])  # (x, y) coordinates
labels = np.array([1, 1])  # 1 = foreground, 0 = background
mask = predictor.predict_with_points(image, points, labels)

# With bounding box
box = np.array([100, 100, 500, 500])  # (x1, y1, x2, y2)
mask = predictor.predict_with_box(image, box)
```

The predictor handles all tensor conversions, device placement, and prompt encoding automatically.

### 4.5 GeoJSON Utilities (utils/geojson_utils.py)

This module converts binary segmentation masks to GeoJSON format compatible with Halo's annotation system.

**Core Functionality:**
- Contour extraction using scikit-image
- Polygon simplification with Douglas-Peucker algorithm
- Area-based filtering to remove small artifacts
- GeoJSON FeatureCollection generation
- File I/O operations

**Conversion Pipeline:**
1. Extract contours from binary mask
2. Filter polygons by minimum area
3. Simplify polygon geometry (optional)
4. Convert to GeoJSON coordinate format
5. Create FeatureCollection with metadata

```python
from utils.geojson_utils import *

# Convert mask to polygons
polygons = mask_to_polygons(
    mask,
    min_area=100  # Filter small objects
)

# Create GeoJSON with metadata
geojson = polygons_to_geojson(
    polygons,
    properties={"classification": "tumor", "confidence": 0.95},
    simplify=True,
    tolerance=1.0  # Simplification tolerance
)

# Save to file
save_geojson(geojson, "annotations.geojson")

# Load GeoJSON
loaded = load_geojson("annotations.geojson")
```

The GeoJSON format is widely supported and can be imported directly into Halo for visualization and further analysis.

---

## 5. Advanced Features

### 5.1 Batch Processing

For high-throughput analysis, the application can be extended to process multiple slides or ROIs automatically:

```python
def batch_analyze_slides(api, predictor, slide_ids, roi_configs):
    """
    Process multiple slides in batch mode.
    
    Args:
        api: HaloAPI instance
        predictor: MedSAMPredictor instance
        slide_ids: List of slide IDs to process
        roi_configs: List of ROI dictionaries (x, y, width, height)
    
    Returns:
        List of results with GeoJSON annotations
    """
    results = []
    
    for slide_id in slide_ids:
        for roi in roi_configs:
            try:
                # Download region
                data = api.download_region(slide_id, **roi)
                image = load_image_from_bytes(data)
                
                # Preprocess
                prep, meta = preprocess_for_medsam(image)
                
                # Analyze
                mask = predictor.predict(prep)
                final = postprocess_mask(mask, meta)
                
                # Convert to GeoJSON
                polygons = mask_to_polygons(final, min_area=100)
                geojson = polygons_to_geojson(polygons)
                
                results.append({
                    'slide_id': slide_id,
                    'roi': roi,
                    'geojson': geojson,
                    'stats': compute_mask_statistics(final)
                })
                
            except Exception as e:
                logger.error(f"Failed to process {slide_id}: {e}")
                continue
    
    return results
```

### 5.2 Custom Model Integration

To integrate a different segmentation model, create a custom predictor class:

```python
class CustomModelPredictor:
    """Template for custom model integration"""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, path):
        """Load your custom model"""
        # Implement model loading logic
        model = YourModelClass()
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        return model
    
    def predict(self, image):
        """Run inference"""
        # Implement inference logic
        with torch.no_grad():
            output = self.model(image)
        return output.cpu().numpy()
```

### 5.3 Interactive Segmentation with Prompts

For cases requiring user input, implement interactive prompting:

```python
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def interactive_segmentation(image, predictor):
    """Allow user to draw prompts on image"""
    
    # Display image with drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=Image.fromarray(image),
        drawing_mode="point",
        key="canvas"
    )
    
    if canvas_result.json_data is not None:
        # Extract point coordinates from canvas
        points = []
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle":
                points.append([obj["left"], obj["top"]])
        
        if len(points) > 0:
            # Run segmentation with user-provided points
            points = np.array(points)
            labels = np.ones(len(points))  # All foreground
            mask = predictor.predict_with_points(image, points, labels)
            st.image(mask, caption="Segmentation Result")
```

### 5.4 Multi-Scale Analysis

For analyzing structures at different scales:

```python
def multiscale_analysis(api, predictor, slide_id, center_x, center_y):
    """
    Analyze the same region at multiple magnifications.
    """
    scales = [0, 1, 2]  # Pyramid levels
    results = []
    
    for level in scales:
        # Download at different level
        data = api.download_region(
            slide_id, center_x, center_y,
            width=1024, height=1024,
            level=level
        )
        
        # Analyze
        image = load_image_from_bytes(data)
        prep, meta = preprocess_for_medsam(image)
        mask = predictor.predict(prep)
        final = postprocess_mask(mask, meta)
        
        results.append({
            'level': level,
            'mask': final,
            'stats': compute_mask_statistics(final)
        })
    
    return results
```

---

## 6. Testing and Debugging

### 6.1 Unit Testing

Create comprehensive unit tests for each module:

```python
# tests/test_image_proc.py
import pytest
import numpy as np
from utils.image_proc import *

def test_preprocess_for_medsam():
    # Create test image
    image = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
    
    # Preprocess
    preprocessed, metadata = preprocess_for_medsam(image, target_size=1024)
    
    # Verify output shape
    assert preprocessed.shape == (1024, 1024, 3)
    assert 'original_shape' in metadata
    assert metadata['original_shape'] == (512, 768)

def test_postprocess_mask():
    # Create test mask and metadata
    mask = np.ones((1024, 1024), dtype=bool)
    metadata = {
        'original_shape': (512, 768),
        'resized_shape': (512, 768),
        'padding': (256, 256, 128, 128),
        'target_size': 1024
    }
    
    # Postprocess
    final = postprocess_mask(mask, metadata)
    
    # Verify output shape matches original
    assert final.shape == (512, 768)
```

### 6.2 Integration Testing

Test the complete workflow:

```python
# tests/test_integration.py
def test_end_to_end_workflow():
    """Test complete analysis pipeline"""
    
    # Setup
    api = HaloAPI(test_endpoint, test_token)
    predictor = MedSAMPredictor(checkpoint_path, device="cpu")
    
    # Get slides
    slides = asyncio.run(api.get_slides(limit=1))
    assert len(slides) > 0
    
    # Download region
    data = api.download_region(slides[0]['id'], 0, 0, 512, 512)
    assert len(data) > 0
    
    # Process
    image = load_image_from_bytes(data)
    prep, meta = preprocess_for_medsam(image)
    mask = predictor.predict(prep)
    final = postprocess_mask(mask, meta)
    
    # Export
    polygons = mask_to_polygons(final)
    geojson = polygons_to_geojson(polygons)
    
    # Verify
    assert 'features' in geojson
    assert len(geojson['features']) > 0
```

### 6.3 Debugging Common Issues

**Issue: CUDA Out of Memory**
```python
# Solution 1: Clear cache
import torch
torch.cuda.empty_cache()

# Solution 2: Use CPU mode
predictor = MedSAMPredictor(checkpoint, device="cpu")

# Solution 3: Reduce image size
prep, meta = preprocess_for_medsam(image, target_size=512)
```

**Issue: GraphQL Connection Errors**
```python
# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_slides_with_retry(api):
    return await api.get_slides()
```

**Issue: Slow Performance**
```python
# Profile code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
mask = predictor.predict(image)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

---

## 7. Deployment Guide

### 7.1 Local Deployment

For development and testing:

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Run application
streamlit run app.py --server.port 8501

# Access at http://localhost:8501
```

### 7.2 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libopensli de-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model weights
RUN mkdir -p models && \
    wget -O models/medsam_vit_b.pth \
    https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1

# Expose Streamlit port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
# Build image
docker build -t xhalopathanalyzer .

# Run container
docker run -p 8501:8501 \
    -e HALO_API_ENDPOINT=https://your-halo.com/graphql \
    -e HALO_API_TOKEN=your_token \
    xhalopathanalyzer
```

### 7.3 Cloud Deployment (AWS, Azure, GCP)

For cloud deployment, use container services:

**AWS ECS:**
```bash
# Push to ECR
aws ecr create-repository --repository-name xhalopathanalyzer
docker tag xhalopathanalyzer:latest <account>.dkr.ecr.<region>.amazonaws.com/xhalopathanalyzer
docker push <account>.dkr.ecr.<region>.amazonaws.com/xhalopathanalyzer

# Deploy to ECS
aws ecs create-service \
    --cluster my-cluster \
    --service-name xhalopathanalyzer \
    --task-definition xhalopathanalyzer:1 \
    --desired-count 1
```

---

## 8. Limitations and Extensions

### 8.1 Current Limitations

1. **Single User Sessions**: No multi-user authentication or session management
2. **Limited Prompt Types**: Only supports point and box prompts, not text prompts
3. **No Real-Time Collaboration**: Users cannot share sessions or annotations in real-time
4. **Memory Constraints**: Large WSIs may exceed available RAM
5. **Model Support**: Currently only supports MedSAM (SAM-based) models

### 8.2 Potential Extensions

**1. Support for Additional Models**
- Cellpose for cell segmentation
- StarDist for nucleus detection
- Custom U-Net architectures
- Foundation models like DinoV2

**2. Advanced Visualization**
- Heatmaps for confidence scores
- 3D visualization for z-stack images
- Time-series analysis for live imaging

**3. Collaborative Features**
- Multi-user session sharing
- Real-time annotation collaboration
- Version control for annotations

**4. Performance Optimizations**
- Tile-based processing for large WSIs
- Distributed processing across multiple GPUs
- Caching of frequently accessed regions

**5. Enhanced Export Options**
- COCO format for training data
- QuPath-compatible formats
- OMERO integration

---

## 9. Full Example Workflow

Here's a complete end-to-end workflow demonstrating all features:

### Step 1: Setup and Authentication

```python
from config import Config
from utils.halo_api import HaloAPI

# Validate configuration
Config.validate()

# Connect to Halo
api = HaloAPI(
    endpoint=Config.HALO_API_ENDPOINT,
    token=Config.HALO_API_TOKEN
)

# Test connection
if asyncio.run(api.test_connection()):
    print(" Connected to Halo")
```

### Step 2: Browse and Select Slides

```python
# Fetch available slides
slides = asyncio.run(api.get_slides(limit=100))

# Filter by name
tumor_slides = [s for s in slides if 'tumor' in s['name'].lower()]

# Select first tumor slide
selected_slide = tumor_slides[0]
print(f"Selected: {selected_slide['name']}")
print(f"Dimensions: {selected_slide['width']}x{selected_slide['height']}")
print(f"MPP: {selected_slide.get('mpp', 'N/A')}")
```

### Step 3: Define Region of Interest

```python
# Define ROI (top-left 2048x2048 region)
roi = {
    'x': 0,
    'y': 0,
    'width': 2048,
    'height': 2048,
    'level': 0
}

# Download region
region_data = api.download_region(
    slide_id=selected_slide['id'],
    **roi
)
print(f"Downloaded {len(region_data)} bytes")
```

### Step 4: Run AI Analysis

```python
from utils.ml_models import MedSAMPredictor
from utils.image_proc import *

# Load image
image = load_image_from_bytes(region_data)
print(f"Image shape: {image.shape}")

# Initialize predictor
predictor = MedSAMPredictor(
    checkpoint_path=Config.MEDSAM_CHECKPOINT,
    model_type=Config.MODEL_TYPE,
    device=Config.DEVICE
)

# Preprocess
preprocessed, metadata = preprocess_for_medsam(image)

# Run segmentation
print("Running MedSAM inference...")
mask = predictor.predict(preprocessed)

# Postprocess
final_mask = postprocess_mask(mask, metadata)
print(f"Mask shape: {final_mask.shape}")

# Compute statistics
stats = compute_mask_statistics(final_mask, mpp=selected_slide.get('mpp'))
print(f"Coverage: {stats['coverage_percent']:.2f}%")
if 'area_mm2' in stats:
    print(f"Area: {stats['area_mm2']:.4f} mm²")
```

### Step 5: Visualize Results

```python
import matplotlib.pyplot as plt

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Segmentation mask
axes[1].imshow(final_mask, cmap='gray')
axes[1].set_title("Segmentation Mask")
axes[1].axis('off')

# Overlay
overlay = overlay_mask_on_image(image, final_mask, color=(255, 0, 0), alpha=0.5)
axes[2].imshow(overlay)
axes[2].set_title("Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("results.png", dpi=300)
print("Saved visualization to results.png")
```

### Step 6: Export to GeoJSON

```python
from utils.geojson_utils import *

# Convert mask to polygons
polygons = mask_to_polygons(
    final_mask,
    min_area=Config.MIN_POLYGON_AREA
)
print(f"Extracted {len(polygons)} polygons")

# Create GeoJSON
geojson = polygons_to_geojson(
    polygons,
    properties={
        "classification": "tumor",
        "slide_id": selected_slide['id'],
        "roi": roi
    },
    simplify=True,
    tolerance=Config.SIMPLIFY_TOLERANCE
)

# Save to file
output_path = Config.get_temp_path("annotations.geojson")
save_geojson(geojson, str(output_path))
print(f"Exported GeoJSON to {output_path}")
```

### Step 7: Import to Halo (Optional)

```python
# Upload annotations back to Halo
annotation_result = asyncio.run(
    api.upload_annotations(
        slide_id=selected_slide['id'],
        geojson=geojson,
        name="MedSAM Segmentation"
    )
)

print(f"Uploaded annotations: {annotation_result['id']}")
```

### Complete Script

```python
#!/usr/bin/env python3
"""
Complete workflow script for XHaloPathAnalyzer
"""

import asyncio
from config import Config
from utils.halo_api import HaloAPI
from utils.image_proc import *
from utils.ml_models import MedSAMPredictor
from utils.geojson_utils import *

async def main():
    # 1. Setup
    Config.validate()
    api = HaloAPI(Config.HALO_API_ENDPOINT, Config.HALO_API_TOKEN)
    
    # 2. Get slides
    slides = await api.get_slides(limit=10)
    selected = slides[0]
    
    # 3. Download region
    data = api.download_region(selected['id'], 0, 0, 1024, 1024)
    image = load_image_from_bytes(data)
    
    # 4. Analyze
    predictor = MedSAMPredictor(Config.MEDSAM_CHECKPOINT, device=Config.DEVICE)
    prep, meta = preprocess_for_medsam(image)
    mask = predictor.predict(prep)
    final = postprocess_mask(mask, meta)
    
    # 5. Export
    polygons = mask_to_polygons(final)
    geojson = polygons_to_geojson(polygons)
    save_geojson(geojson, "output.geojson")
    
    # 6. Upload
    result = await api.upload_annotations(selected['id'], geojson)
    
    print(f" Complete! Processed {selected['name']}")
    print(f"   Objects: {len(polygons)}")
    print(f"   Annotation ID: {result['id']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Conclusion

XHaloPathAnalyzer provides a comprehensive, user-friendly solution for integrating custom AI analysis with the Halo digital pathology platform. By automating the export-analyze-import workflow, it enables researchers to apply state-of-the-art machine learning models to their pathology images efficiently and reproducibly.

The modular architecture makes it easy to extend with custom models, additional analysis methods, and new features. Whether you're analyzing a single slide or processing hundreds in batch mode, XHaloPathAnalyzer streamlines your workflow and lets you focus on the science rather than the technical details.

For support, bug reports, or feature requests, please visit the project repository or contact the development team.

**Happy Analyzing! **
