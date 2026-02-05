# XHaloPathAnalyzer 

**Web-Based GUI for Halo Digital Pathology Image Analysis**

A comprehensive, OS-agnostic application for custom image analysis on whole-slide images stored in the Halo digital pathology platform. Integrates MedSAM AI segmentation with Halo's API for seamless export-analyze-import workflows.

## Features

-  **Secure Authentication**: Connect to Halo via GraphQL API **OR** use Local Mode
-  **Local Image Upload**: Upload JPG, PNG, TIFF images directly without Halo connection
-  **Slide Management**: Browse, search, and select slides from Halo (API mode)
-  **AI Analysis**: MedSAM segmentation on regions of interest or uploaded images
-  **Visualization**: Side-by-side comparison and overlay views
-  **GeoJSON Export**: Convert results to Halo-compatible annotations
-  **Cross-Platform**: Works on Windows, macOS, and Linux
-  **GPU Accelerated**: Automatic CUDA/MPS detection and optimization (NVIDIA GPUs, Apple Silicon)

## Usage Modes

XHaloPathAnalyzer supports two operating modes:

### 1.  Halo API Mode (Default)
- Connect to your Halo digital pathology platform
- Browse and select slides from your Halo repository
- Download regions of interest for analysis
- Export results back to Halo

### 2.  Local Image Upload Mode (NEW!)
- **No Halo connection required**
- Upload images directly (JPG, PNG, TIFF)
- Analyze single or batch images
- Export segmentation masks and GeoJSON
- Perfect for:
  - Quick analysis of local images
  - Testing without Halo access
  - Batch processing workflows
  - Standalone image analysis

## Quick Start

### 1. Install uv (Recommended for Mac/M2)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer that works great on all platforms, especially Mac M2/ARM.

```bash
# Install uv (Mac/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew on macOS
brew install uv

# Or using pip
pip install uv
```

### 2. Install Dependencies

#### Option A: Using uv (Recommended for Mac M2/ARM)

```bash
# Create virtual environment with Python 3.11 using uv
# Note: Requires Python 3.11 to be installed on your system
uv venv -p 3.11 venv_xhpa_v1

# Activate virtual environment
source venv_xhpa_v1/bin/activate  # On Windows: venv_xhpa_v1\Scripts\activate

# Verify Python version (should be 3.11.x)
python -V

# Install packages with uv (much faster than pip!)
uv pip install -r requirements-uv.txt

# Install OpenSlide (platform-specific)
# macOS (including M2/ARM): 
brew install openslide

# Ubuntu: 
# sudo apt-get install openslide-tools

# Windows: 
# Download from https://openslide.org/download/
```

#### Option B: Using traditional pip

```bash
# Create virtual environment
python3 -m venv venv_xhpa_v1
source venv_xhpa_v1/bin/activate  # On Windows: venv_xhpa_v1\Scripts\activate

# Install packages
pip install -r requirements.txt

# Install OpenSlide (platform-specific)
# macOS: brew install openslide
# Ubuntu: sudo apt-get install openslide-tools
# Windows: Download from https://openslide.org/download/
```

**Note for Mac M2/ARM users:** 
- uv handles ARM architecture dependencies automatically
- PyTorch will install the correct ARM64 version
- If you encounter any issues, ensure you're using Python 3.9+ for best M2 compatibility

### 3. Download MedSAM Model
```bash
# Create models directory and download checkpoint (1.7GB)
mkdir -p models

# Option A: Using wget (if available)
wget -O models/medsam_vit_b.pth \
  https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1

# Option B: Using curl (default on macOS)
curl -L -o models/medsam_vit_b.pth \
  https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1
```

### 4. Patch Segment Anything (Optional but Recommended)

To ensure SAM checkpoints saved from CUDA devices can load on CPU/MPS machines, run the patch script:

```bash
python patch_segment_anything.py
```

This script modifies the `segment_anything` package to add `map_location="cpu"` to model loading, ensuring cross-platform compatibility.

**Alternative Manual Patch:**
If you prefer to patch manually, edit your installed `segment_anything/build_sam.py` file and change:
```python
state_dict = torch.load(f, weights_only=False)
```
to:
```python
state_dict = torch.load(
    f,
    map_location="cpu",
    weights_only=False,
)
```

### 5. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit with your Halo credentials
nano .env
```

### 6. Run Application
```bash
# Start Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
```

## Using Local Mode

If you want to analyze images without connecting to Halo:

1. **Start the application**: `streamlit run app.py`
2. **Select Mode**: Choose " Local Image Upload Mode" on the authentication page
3. **Upload Images**: Navigate to the "Image Upload" page and upload your JPG/PNG/TIFF files
4. **Select Image**: Choose which image to analyze
5. **Run Analysis**: Go to the "Analysis" page and click "Run Analysis"
6. **Export Results**: Download segmentation masks and GeoJSON from the "Export" page

**Note**: Local mode does not require Halo API credentials or the `.env` file configuration.

#  XHalo Path Analyzer

**Halo AI Workflow: A web-based GUI for digital pathology analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

XHalo Path Analyzer is a powerful, OS-agnostic workflow tool that bridges Halo's digital pathology platform with external AI/ML capabilities. It enables researchers to:

-  **Export WSIs/ROIs** from Halo via GraphQL API
-  **Run external ML models** (e.g., MedSAM segmentation) in Python
-  **Import results back** to Halo for visualization and analysis
-  **Process large images** using intelligent tiling strategies
-  **Generate GeoJSON** exports for interoperability
-  **Work independently** of vendor-specific tools

Built for exploratory AI in digital pathology, this tool provides a flexible, interactive environment for developing and deploying machine learning workflows.

## Key Features

###  Digital Pathology Integration
- **Halo GraphQL API Integration**: Direct connection to Halo for slide management
- **WSI/ROI Export**: Export whole slide images and regions of interest
- **Annotation Import**: Push AI-generated annotations back to Halo

###  AI/ML Capabilities
- **MedSAM Integration**: Medical Segment Anything Model for tissue segmentation
- **Tiled Processing**: Handle large pathology images efficiently
- **Custom Model Support**: Extensible architecture for other ML models

###  Visualization & Analysis
- **Interactive Web UI**: Built with Streamlit for ease of use
- **Real-time Visualization**: See segmentation results immediately
- **Overlay Views**: Compare original images with segmentation masks
- **Statistics**: Automatic calculation of coverage metrics

###  Data Export
- **GeoJSON Export**: Industry-standard format for annotations
- **Mask Export**: Save binary segmentation masks
- **Halo Import**: Direct upload of results to Halo platform

## Technology Stack

- **Frontend**: Streamlit
- **API Integration**: gql (GraphQL client)
- **Image Processing**: large_image, Pillow, OpenCV
- **Machine Learning**: PyTorch, MedSAM
- **Geospatial**: geojson, shapely
- **Visualization**: matplotlib, plotly

## Installation

### Prerequisites

- Python 3.8 or higher (3.9+ recommended for Mac M2/ARM)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip package manager
- [Homebrew](https://brew.sh/) (for Mac users, to install OpenSlide)
- (Optional) CUDA-capable GPU or Apple Silicon Mac for faster inference

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/eisascience/XHaloPathAnalyzer.git
cd XHaloPathAnalyzer
```

2. **Install uv (Recommended for Mac M2/ARM)**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv

# Or via pip
pip install uv
```

3. **Create a virtual environment**

**Using uv (faster, recommended):**
```bash
# Requires Python 3.11 to be installed on your system
uv venv -p 3.11 venv_xhpa_v1
source venv_xhpa_v1/bin/activate  # On Windows: venv_xhpa_v1\Scripts\activate
python -V  # should be 3.11.x
```

**Using traditional venv:**
```bash
python -m venv venv_xhpa_v1
source venv_xhpa_v1/bin/activate  # On Windows: venv_xhpa_v1\Scripts\activate
```

4. **Install dependencies**

**Using uv (faster, especially on Mac M2):**
```bash
uv pip install -r requirements-uv.txt
uv pip install -e .
```

**Using pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

5. **Install OpenSlide (Required for WSI processing)**
```bash
# macOS (including M2/ARM)
brew install openslide

# Ubuntu/Debian
sudo apt-get install openslide-tools

# Windows
# Download from https://openslide.org/download/
```

### Mac M2/ARM Specific Notes

- **Python Version**: Use Python 3.9 or higher for best M2 compatibility
- **PyTorch**: The ARM64 version will be installed automatically
- **OpenSlide**: Install via Homebrew (`brew install openslide`)
- **uv Benefits**: uv handles ARM dependencies much better than pip
- **Architecture Check**: Run `uname -m` (should show `arm64` on M2)

If you encounter issues on M2:
```bash
# Verify you're using ARM Python (not Rosetta)
python -c "import platform; print(platform.machine())"
# Should output: arm64

# If it shows x86_64, install native ARM Python:
brew install python@3.11
```

## Usage

### Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Or use the CLI:

```bash
xhalo-analyzer web --port 8501 --host localhost
```

Then open your browser to `http://localhost:8501`

### Web Interface Workflow

1. **Configure Halo API Connection**
   - Enter your Halo API URL and authentication key in the sidebar
   - Or use the Mock API for testing without a real Halo instance

2. **Initialize MedSAM**
   - Select device (CPU/CUDA/MPS - automatically detected)
   - Optionally provide path to MedSAM checkpoint
   - Click "Initialize MedSAM"

3. **Select/Upload Image**
   - Load slides from Halo using the GraphQL API
   - Or upload a local image file for analysis

4. **Run Segmentation**
   - Adjust processing parameters (tile size, overlap, min area)
   - Click "Run Segmentation"
   - View results with mask and overlay visualizations

5. **Export Results**
   - Export as GeoJSON for interoperability
   - Download segmentation mask as PNG
   - Import annotations directly back to Halo

### Command-Line Interface

Process images directly from the command line:

```bash
# Process an image and save results
xhalo-analyzer process input.tif \
    --output mask.png \
    --geojson annotations.geojson \
    --tile-size 1024
```

### Python API

Use XHalo Path Analyzer programmatically:

```python
from xhalo.api import MockHaloAPIClient
from xhalo.ml import MedSAMPredictor, segment_tissue
from xhalo.utils import load_image, mask_to_geojson
import asyncio

# Initialize API client
client = MockHaloAPIClient()

# Load slides
slides = asyncio.run(client.list_slides())
print(f"Found {len(slides)} slides")

# Load an image
image = load_image("path/to/image.tif")

# Run segmentation
predictor = MedSAMPredictor(device="cpu")
mask = predictor.predict_tiles(image, tile_size=1024)

# Export to GeoJSON
geojson_data = mask_to_geojson(mask, min_area=100)

# Import back to Halo
annotations = convert_to_halo_annotations(mask)
success = asyncio.run(
    client.import_annotations(slide_id, annotations)
)
```

## Configuration

### Halo API Setup

To connect to a real Halo instance:

1. Obtain your Halo GraphQL API endpoint URL
2. Generate an API key/token from your Halo instance
3. Enter these credentials in the web UI sidebar

### MedSAM Model

To use the full MedSAM model:

1. Download the MedSAM checkpoint from the [official repository](https://github.com/bowang-lab/MedSAM)
2. Provide the path to the checkpoint in the web UI
3. Select appropriate device (CUDA for NVIDIA GPUs, MPS for Apple Silicon, or CPU)

Note: The application includes a mock segmentation mode for testing without the full model.

## Project Structure

```
XHaloPathAnalyzer/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── GUIDE.md              # Comprehensive 5000+ word guide
├── utils/
│   ├── halo_api.py       # Halo GraphQL API integration
│   ├── image_proc.py     # Image processing utilities
│   ├── ml_models.py      # MedSAM model wrapper
│   └── geojson_utils.py  # GeoJSON conversion
├── models/               # Model weights directory
└── temp/                 # Temporary files directory
```

## Documentation

See **[GUIDE.md](GUIDE.md)** for comprehensive documentation including:
- Detailed architecture explanation
- Complete setup instructions
- Code implementation details
- Advanced features and extensions
- Testing and debugging guide
- Deployment instructions
- Full example workflows

## Requirements

- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: Optional (NVIDIA CUDA or Apple Silicon MPS for acceleration)
- **Storage**: 10GB for models and cache
- **Halo**: API access with valid token

## Key Technologies

- **Streamlit**: Web framework for interactive UI
- **PyTorch**: Deep learning framework
- **MedSAM**: Medical image segmentation model
- **GraphQL**: API communication with Halo
- **OpenSlide**: Whole-slide image processing
- **scikit-image**: Image processing and analysis

## Usage Example

```python
from config import Config
from utils.halo_api import HaloAPI
from utils.ml_models import MedSAMPredictor
from utils.image_proc import *
from utils.geojson_utils import *
import asyncio

# Setup
Config.validate()
api = HaloAPI(Config.HALO_API_ENDPOINT, Config.HALO_API_TOKEN)

# Get slides
slides = asyncio.run(api.get_slides())
slide = slides[0]

# Download region
data = api.download_region(slide['id'], 0, 0, 1024, 1024)
image = load_image_from_bytes(data)

# Analyze with MedSAM
predictor = MedSAMPredictor(Config.MEDSAM_CHECKPOINT)
preprocessed, metadata = preprocess_for_medsam(image)
mask = predictor.predict(preprocessed)
final_mask = postprocess_mask(mask, metadata)

# Export to GeoJSON
polygons = mask_to_polygons(final_mask)
geojson = polygons_to_geojson(polygons)
save_geojson(geojson, "annotations.geojson")
```

## License

This project is provided as-is for research and educational purposes.

## Support

For questions, issues, or contributions, please open an issue on GitHub.

---

**Built with care for the digital pathology community**
X-Halo-Patho-Analyzer
├── app.py                          # Main Streamlit application
├── setup.py                        # Package configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── xhalo/                          # Main package
│   ├── __init__.py
│   ├── cli.py                      # Command-line interface
│   ├── api/                        # Halo API integration
│   │   ├── __init__.py
│   │   └── halo_client.py         # GraphQL client
│   ├── ml/                         # Machine learning models
│   │   ├── __init__.py
│   │   └── medsam.py              # MedSAM integration
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── image_utils.py         # Image processing
│       └── geojson_utils.py       # GeoJSON conversion
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=xhalo tests/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Deployment

### Local Deployment

The application can be run locally as described in the Usage section.

### Cloud Deployment

#### Streamlit Cloud

1. Push your repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your repository

#### Docker (Coming Soon)

```bash
# Build Docker image
docker build -t xhalo-analyzer .

# Run container
docker run -p 8501:8501 xhalo-analyzer
```

#### Other Cloud Platforms

The application can be deployed to:
- AWS EC2 / ECS
- Google Cloud Run
- Azure Web Apps
- Heroku

See the [deployment documentation](docs/deployment.md) for detailed instructions.

## Use Cases

### Research Applications
- Automated tissue segmentation in pathology studies
- High-throughput analysis of large slide collections
- Validation of manual annotations
- Exploratory analysis with AI models

### Clinical Workflows
- Pre-screening of samples for pathologist review
- Quantitative assessment of tissue characteristics
- Standardized measurement protocols
- Integration with existing LIMS/pathology systems

### AI/ML Development
- Rapid prototyping of segmentation models
- Model validation and comparison
- Ground truth generation for training data
- Deployment of custom models in production

## Limitations

- Mock MedSAM implementation for demonstration (full model integration requires checkpoint)
- GraphQL API schema may need adaptation for specific Halo versions
- Large image processing requires adequate RAM
- Real-time processing depends on hardware capabilities

## Roadmap

- [ ] Support for additional ML models
- [ ] Enhanced visualization options
- [ ] Batch processing capabilities
- [ ] Multi-user collaboration features
- [ ] Docker containerization
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline
- [ ] Extended documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MedSAM](https://github.com/bowang-lab/MedSAM) - Medical Segment Anything Model
- [Halo](https://indicalab.com/halo/) - Digital pathology platform by Indica Labs
- [Streamlit](https://streamlit.io/) - Web application framework

## Support

For issues, questions, or contributions:
-  Open an issue on [GitHub](https://github.com/eisascience/XHaloPathAnalyzer/issues)
-  Check the [documentation](docs/)
-  Join discussions in the repository

## Citation

## Halo Link Integration

XHaloPathAnalyzer now includes support for Halo Link, providing OIDC discovery, OAuth2 authentication, and GraphQL integration.

### Prerequisites

 **VPN Requirement**: You must be connected to your organization's VPN to access Halo Link services.

### Configuration

Add the following environment variables to your `.env` file:

```bash
# Required: Base URL for Halo Link
HALOLINK_BASE_URL=https://halolink.example.com

# Optional: Direct GraphQL endpoint (preferred if you know it)
HALOLINK_GRAPHQL_URL=https://halolink.example.com/api/graphql

# Optional: GraphQL path (used if GRAPHQL_URL not set)
HALOLINK_GRAPHQL_PATH=/api/graphql

# Optional: OAuth2 credentials (required for authenticated access)
HALOLINK_CLIENT_ID=your_client_id
HALOLINK_CLIENT_SECRET=your_client_secret

# Optional: OAuth2 scope
HALOLINK_SCOPE=read write
```

### Finding Your GraphQL Endpoint

If you don't know your GraphQL endpoint URL, you can find it using browser DevTools:

1. Open your Halo Link web interface in a browser (Chrome/Firefox)
2. Open Developer Tools (F12 or Right-click → Inspect)
3. Go to the **Network** tab
4. Interact with Halo Link (e.g., browse slides, view data)
5. Filter network requests by "graphql" or "api"
6. Look for POST requests to endpoints like:
   - `https://halolink.example.com/api/graphql`
   - `https://halolink.example.com/graphql`
7. Copy the full URL and set it as `HALOLINK_GRAPHQL_URL`

### Testing Your Configuration

Test your Halo Link configuration using the CLI smoke test:

```bash
# Basic test
python -m xhalo.halolink.smoketest

# Verbose output
python -m xhalo.halolink.smoketest --verbose

# Custom test query (default is '{ __typename }')
export HALOLINK_SMOKETEST_QUERY='{ __typename }'
python -m xhalo.halolink.smoketest
```

The smoke test will:
1.  Initialize the client
2.  Perform OIDC discovery
3.  Retrieve OAuth2 token (if credentials configured)
4.  Execute a test GraphQL query

### Using in Streamlit

The Halo Link integration is available in the Streamlit app:

1. Launch the app: `streamlit run app.py`
2. Navigate to ** Settings** page
3. Scroll to the ** Halo Link Integration** section
4. Click **Run Halo Link Smoke Test** to test your configuration
5. View results and detailed output

### Troubleshooting

**Connection Failed / Timeout**
-  Verify you're connected to VPN
-  Check `HALOLINK_BASE_URL` is correct
-  Verify the server is accessible from your network

**OIDC Discovery Failed**
-  Verify `HALOLINK_BASE_URL` points to the correct server
-  Ensure the server supports OIDC (has `/.well-known/openid-configuration`)
-  Check VPN connection

**Token Retrieval Failed**
-  Verify `HALOLINK_CLIENT_ID` and `HALOLINK_CLIENT_SECRET` are correct
-  Check that your client credentials have not expired
-  Ensure you have proper permissions

**GraphQL Query Failed**
-  Verify `HALOLINK_GRAPHQL_URL` or `HALOLINK_GRAPHQL_PATH` is correct
-  Check the query syntax is valid
-  Ensure your token has appropriate permissions for the query

### Example Configuration

```bash
# Example for production environment
HALOLINK_BASE_URL=https://halolink.ohsu.edu
HALOLINK_GRAPHQL_URL=https://halolink.ohsu.edu/api/v1/graphql
HALOLINK_CLIENT_ID=my-client-id
HALOLINK_CLIENT_SECRET=my-secret-key
HALOLINK_SCOPE=read write

# Example for development/testing (no auth)
HALOLINK_BASE_URL=https://halolink-dev.example.com
HALOLINK_GRAPHQL_PATH=/graphql
```

---

If you use this tool in your research, please cite:

```bibtex
@software{xhalo_path_analyzer,
  title = {XHalo Path Analyzer: Web-based AI Workflow for Digital Pathology},
  author = {Eisa Science},
  year = {2026},
  url = {https://github.com/eisascience/XHaloPathAnalyzer}
}
```

---

**Built for exploratory AI in digital pathology** 
