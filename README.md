# XHistoSegMicroSAM

> **⚠️ Python Version Requirement**: This application ONLY works with **Python 3.9**. 
> Python 3.10+ is not supported due to dependency conflicts between micro-sam and python-elf.

Histopathology instance segmentation using micro-sam with Streamlit interface.

## Key Features

- Instance segmentation for histopathology images
- Two modes: Interactive (with prompts) or Automatic (no prompts)
- MicroSAM models optimized for histology
- Halo API integration or standalone local mode
- GeoJSON export compatible with Halo

## Installation

### Prerequisites

- **Python 3.9** (REQUIRED - other versions will not work)
- **Conda** (Miniconda or Anaconda) for managing the environment
- **Git** for cloning the repository

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/eisascience/XHistoSegMicroSAM.git
cd XHistoSegMicroSAM

# Run the setup script
./setup.sh
```

The setup script will:
1. Create a conda environment with Python 3.9
2. Install system dependencies (python-elf, vigra, nifty)
3. Install all Python packages
4. Install torch-em and micro-sam v1.3.0
5. Verify the installation

### Manual Installation

If you prefer to install manually or the script doesn't work on your system:

```bash
# 1. Create conda environment
conda create -n xhisto python=3.9 -y
conda activate xhisto

# 2. Install conda packages (REQUIRED - not available via pip)
conda install -c conda-forge python-elf vigra nifty -y

# 3. Install Python packages
pip install -r requirements-py39.txt

# 4. Install torch-em (required dependency of micro-sam)
pip install git+https://github.com/constantinpape/torch-em.git

# 5. Install micro-sam v1.3.0 (last version with Python 3.9 support)
pip install git+https://github.com/computational-cell-analytics/micro-sam.git@v1.3.0

# 6. Verify installation
python -c "from xhalo.ml import MicroSAMPredictor; print('Installation successful!')"
```

### Why Python 3.9?

There is an impossible dependency conflict with Python 3.10+:

- `micro-sam` v1.7.1+ requires Python >=3.10 and has hard imports of `python-elf`
- `python-elf` requires `numba` which requires `llvmlite`
- `llvmlite` for Python <3.10 only supports Python <3.10
- `llvmlite` for Python 3.10+ is incompatible with the `numba` version that `python-elf` needs

**Solution**: Python 3.9 with `micro-sam` v1.3.0 (last version supporting Python 3.9 with histopathology models)

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed explanation and common issues.

## Configuration

Copy `.env.example` to `.env` and configure settings:

```bash
cp .env.example .env
```

### Essential Settings

```bash
# Operating Mode
LOCAL_MODE=true  # Set to true for standalone operation (no Halo needed)

# MicroSAM Model Configuration
MICROSAM_MODEL_TYPE=vit_b_histopathology  # Options: vit_b_histopathology, vit_l_histopathology
ENABLE_TILING=true  # Recommended for large images
TILE_SHAPE=1024,1024  # Tile size for processing
HALO_SIZE=256,256  # Overlap between tiles

# Optional: Embeddings Cache (speeds up re-processing)
ENABLE_EMBEDDINGS_CACHE=false
EMBEDDINGS_CACHE_DIR=./cache/embeddings

# Optional: Model Cache Location
MICROSAM_CACHEDIR=/path/to/cache  # Default: ~/.cache/micro_sam/models
```

### Model Options

- **vit_b_histopathology** (default): Faster, suitable for most histology images
- **vit_l_histopathology**: More accurate but slower, for challenging cases

## Usage

### Starting the Application

```bash
# Activate the conda environment
conda activate xhisto

# Start Streamlit application
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Segmentation Modes

The application supports multiple segmentation modes:

#### Automatic Instance Segmentation (Recommended)

These modes work with Python 3.9 + micro-sam v1.3.0 and provide full automatic segmentation:

1. **APG (Automatic Polygon Generation)**: Fully automatic instance segmentation
2. **AIS (Automatic Instance Segmentation)**: Alternative automatic method
3. **AMG (Automatic Mask Generation)**: Generates masks for all objects

#### Prompt-Based Modes

For more control or specific regions:

1. **auto_box**: Automatically detects tissue region using thresholding
2. **auto_box_from_threshold**: 
   - Generates bounding boxes from a thresholded channel (e.g., DAPI)
   - Supports Otsu or manual thresholding
   - Filters boxes by area (min/max size)
   - Excellent for nuclei/cell segmentation
3. **full_box**: Uses the entire image as the prompt box
4. **point**: Uses point prompts (center point or user-specified)

### Workflow: Local Mode

1. **Select Mode**: Choose "Local Mode" in the sidebar
2. **Upload Image**: Upload your histopathology image (PNG/JPG/TIFF/SVS)
3. **Configure Processing**:
   - Select channels for processing
   - Choose segmentation mode (automatic or prompt-based)
   - Adjust parameters (threshold, box filtering, etc.)
4. **Run Segmentation**: Click "Run Segmentation"
5. **Review Results**: View overlays and instance masks
6. **Download**: Export results as GeoJSON, PNG masks, or visualizations

### Workflow: Halo Mode

1. **Configure Credentials**: Set up Halo API credentials in `.env`
2. **Connect**: Use the Halo connection interface in the sidebar
3. **Select Image**: Choose image from your Halo database
4. **Process**: Same segmentation workflow as Local Mode
5. **Upload Results**: Optionally upload annotations back to Halo

## Pipeline Framework (Advanced)

The application now includes a modular pipeline framework for extensible, biology-specific workflows.

### Analysis Modes

In the **MicroSAM Analysis** tab, you can choose between two modes:

- **Classic Mode**: Traditional single/multi-channel analysis (default behavior)
- **Pipeline Mode**: Advanced modular workflows for specialized biological applications

### Available Pipelines

1. **Basic Single Channel**: Standard segmentation for any image type (replicates classic mode)
2. **Multi-Channel Hierarchical**: Nucleus-guided cell segmentation with compartmental analysis

### Using Pipeline Mode

1. **Select Pipeline Mode** in the MicroSAM Analysis tab
2. **Choose a Pipeline** from the dropdown menu
3. **Configure Pipeline Parameters** using the pipeline-specific UI
4. **Select an Image** from your queue
5. **Run Pipeline** and view results
6. **Export Results** in pipeline-specific formats

### Creating Custom Pipelines

You can create your own analysis pipelines for specialized workflows:

1. Create a new file in `pipelines/` directory
2. Inherit from `BasePipeline` class
3. Implement required methods:
   - `configure_ui()` - Define Streamlit UI controls
   - `validate_channels()` - Check channel requirements
   - `process()` - Implement analysis logic
   - `visualize()` - Display results
   - `export_data()` - Export in desired formats
4. Register your pipeline in `pipelines/__init__.py`

See `pipelines/README.md` for a complete pipeline development guide with examples.

### Pipeline Benefits

- ✅ **Generalizable**: Core app works for any images
- ✅ **Extensible**: Add new pipelines without modifying core code
- ✅ **Maintainable**: Each pipeline is self-contained
- ✅ **Shareable**: Pipelines are standalone Python files
- ✅ **Documented**: Pipeline development guide included

## Test Script

Test the installation with the provided test script:

```bash
# Activate environment
conda activate xhisto

# Run test with synthetic image
python scripts/01_microsam_auto_test.py

# Or test with your own image
python scripts/01_microsam_auto_test.py path/to/image.png
```

Output will be saved to the `test_output/` directory.

## Models

MicroSAM models are downloaded automatically on first use:

- **vit_b_histopathology** (default): Faster, ~300 MB
- **vit_l_histopathology**: More accurate, ~400 MB

Models are cached in `~/.cache/micro_sam/models` (or `$MICROSAM_CACHEDIR` if set).

## Output Formats

The application generates multiple output formats:

- **Instance Masks**: 16-bit PNG with unique integer IDs per object
- **Visualizations**: Color overlays showing segmentation results
- **GeoJSON**: Polygon features compatible with Halo (includes instance_id property)
- **Statistics**: CSV with object counts and measurements

## System Requirements

### Python Environment
- **Python 3.9** (REQUIRED)
- Conda (Miniconda or Anaconda)

### Hardware Requirements
- **Minimum**: 8 GB RAM, 2 GB disk space
- **Recommended**: 16 GB RAM, GPU with 4+ GB VRAM
- **GPU Support**: CUDA (NVIDIA), ROCm (AMD), or MPS (Apple Silicon)

### Dependencies
- PyTorch 2.6.0
- micro-sam v1.3.0
- python-elf (conda only)
- vigra (conda only)
- nifty (conda only)
- torch-em (GitHub install)
- See `requirements-py39.txt` for complete list

### Operating Systems
- Linux (tested on Ubuntu 20.04+)
- macOS (tested on macOS 12+)
- Windows (with WSL2 recommended)

## Troubleshooting

Common issues and solutions are documented in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Quick checks:

```bash
# Verify Python version
python --version  # Should show 3.9.x

# Verify key dependencies
conda list | grep -E "elf|vigra|nifty"
pip show micro-sam torch

# Test import
python -c "from xhalo.ml import MicroSAMPredictor; print('OK')"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

MIT License
