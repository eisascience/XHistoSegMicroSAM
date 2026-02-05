# XHistoSegMicroSAM

Histopathology instance segmentation using micro-sam with Streamlit interface.

## Key Features

- Instance segmentation for histopathology images
- Two modes: Interactive (with prompts) or Automatic (no prompts)
- MicroSAM models optimized for histology
- Halo API integration or standalone local mode
- GeoJSON export compatible with Halo

## Installation

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For macOS with Homebrew
brew install openslide
```

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
