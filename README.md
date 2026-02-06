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

# Verify micro-sam and segment-anything installation
python -c "import micro_sam; import segment_anything; print('OK')"

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

### Segmentation Modes

#### Prompt-Based Modes (Python 3.11 Compatible)

The application supports several prompt-based segmentation modes that work without `python-elf`:

1. **auto_box**: Automatically detects tissue region using thresholding and morphological operations
2. **auto_box_from_threshold**: *(Recommended for nuclei/cell segmentation)*
   - Generates bounding boxes from a thresholded channel (e.g., DAPI)
   - Supports Otsu or manual thresholding
   - Filters boxes by area (min/max size)
   - Optional dilation to capture full nuclei
   - Returns instance segmentation with unique IDs per nucleus
3. **full_box**: Uses the entire image as the prompt box
4. **point**: Uses point prompts (center point or user-specified)

#### Automatic Instance Segmentation (Requires conda environment)

Full automatic segmentation modes (APG/AIS) require `python-elf`, which is **not available in Python 3.11** due to numba/llvmlite constraints.

**To use automatic modes:**

```bash
# Create conda environment with Python 3.9
conda create -n microsam-auto python=3.9
conda activate microsam-auto

# Install python-elf from conda-forge
conda install -c conda-forge python-elf

# Install other dependencies
pip install -r requirements.txt
```

For most use cases (especially nuclei/cell segmentation), the **auto_box_from_threshold** mode provides excellent results without requiring `python-elf`.

### Local Mode
1. Select "Local Mode" in sidebar
2. Upload image (PNG/JPG/TIFF)
3. Configure channel preprocessing if needed
4. Choose prompt mode (auto_box_from_threshold recommended for nuclei)
5. Adjust threshold and box filtering parameters
6. Run segmentation
7. Download results

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

- Python 3.8+ (Python 3.11 recommended for uv installation)
- PyTorch 2.6.0
- micro-sam >= 1.0.0
- See requirements.txt or requirements-uv.txt for full list

**Note:** `python-elf` is NOT required for prompt-based segmentation modes. It is only needed for full automatic instance segmentation (APG/AIS modes) and requires Python <3.10 in a conda environment.

## License

MIT License
