# XHistoSegMicroSAM

Histopathology instance segmentation using micro-sam with Streamlit interface.

## Key Features

- Instance segmentation for histopathology images
- Two modes: Interactive (with prompts) or Automatic (no prompts)
- MicroSAM models optimized for histology
- Halo API integration or standalone local mode
- GeoJSON export compatible with Halo

## Installation

### Standard Installation (pip)

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For macOS with Homebrew
brew install openslide
```

### Install with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. To install with uv:

```bash
# Clone the repository
git clone https://github.com/eisascience/XHistoSegMicroSAM.git
cd XHistoSegMicroSAM

# Install Python 3.11 via uv
uv python install 3.11

# Create virtual environment
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements-uv.txt

# Verify micro-sam installation
python -c "import micro_sam; print('micro_sam OK', micro_sam.__version__)"

# Optional: Set MicroSAM cache directory
export MICROSAM_CACHEDIR="$PWD/.cache/microsam"
```

**Note about link-mode warning**: During installation, you may see a warning:
```
Failed to clone files; falling back to full copy
```
This is not an error and installation will proceed normally. If you prefer to suppress this warning, you can set:
```bash
export UV_LINK_MODE=copy
```
or use `--link-mode=copy` when running uv commands.

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# For local mode (no Halo needed)
LOCAL_MODE=true

# MicroSAM settings
MICROSAM_MODEL_TYPE=vit_b_histopathology
ENABLE_TILING=true
TILE_SHAPE=1024,1024
HALO_SIZE=256,256

# Optional: embeddings cache
ENABLE_EMBEDDINGS_CACHE=false
EMBEDDINGS_CACHE_DIR=./cache/embeddings

# Optional: model cache location
MICROSAM_CACHEDIR=/path/to/cache
```

## Usage

```bash
# Start application
streamlit run app.py
```

### Local Mode
1. Select "Local Mode" in sidebar
2. Upload image (PNG/JPG/TIFF)
3. Choose Interactive or Automatic mode
4. Run segmentation
5. Download results

### Halo Mode
Configure Halo credentials in `.env` then connect through the UI.

## Test Script

```bash
# Run test with synthetic image
python scripts/01_microsam_auto_test.py

# Or with your image
python scripts/01_microsam_auto_test.py path/to/image.png
```

Output saved to `test_output/` directory.

## Models

MicroSAM models download automatically on first use:
- `vit_b_histopathology` (default, faster)
- `vit_l_histopathology` (larger, more accurate)

## Output Formats

- Instance masks: 16-bit PNG with integer IDs
- Visualizations: Color overlays
- GeoJSON: Polygon features with instance_id property

## Requirements

- Python 3.8+
- PyTorch 2.6.0
- micro-sam >= 1.0.0
- See requirements.txt for full list

## License

MIT License
