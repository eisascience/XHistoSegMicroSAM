# Migration Guide: MedSAM to MicroSAM

This document describes the migration from MedSAM to MicroSAM in XHistoSegMicroSAM.

## What Changed

### Dependencies

**Removed:**
- `segment-anything` package
- MedSAM checkpoint files

**Added:**
- `micro-sam>=1.0.0`
- `zarr>=2.16.0` (for embeddings cache)

### Configuration

**Old Settings (Removed):**
```bash
MEDSAM_CHECKPOINT=./models/medsam_vit_b.pth
MODEL_TYPE=vit_b
```

**New Settings:**
```bash
MICROSAM_MODEL_TYPE=vit_b_histopathology
MICROSAM_CACHEDIR=/optional/cache/path
ENABLE_TILING=true
TILE_SHAPE=1024,1024
HALO_SIZE=256,256
ENABLE_EMBEDDINGS_CACHE=false
EMBEDDINGS_CACHE_DIR=./cache/embeddings
```

### API Changes

**Old Code (MedSAM):**
```python
from utils.ml_models import MedSAMPredictor

predictor = MedSAMPredictor(
    checkpoint_path="./models/medsam_vit_b.pth",
    model_type="vit_b",
    device="cuda"
)

mask = predictor.predict(
    image,
    prompt_mode="auto_box",
    multimask_output=False
)
```

**New Code (MicroSAM):**
```python
from xhalo.ml import MicroSAMPredictor

predictor = MicroSAMPredictor(
    model_type="vit_b_histopathology",
    device="cuda",
    tile_shape=(1024, 1024),
    halo=(256, 256)
)

# Interactive mode with prompts
instance_mask = predictor.predict_from_prompts(
    image,
    boxes=np.array([[x1, y1, x2, y2]]),
    tiled=True
)

# Automatic mode (no prompts)
instance_mask = predictor.predict_auto_instances(
    image,
    segmentation_mode="apg",
    tiled=True
)
```

## Key Differences

### 1. Output Format

**MedSAM:**
- Binary mask (0 or 255)
- Single object per prediction

**MicroSAM:**
- Instance-labeled mask (0=background, 1..N=instances)
- Multiple objects per prediction
- Each instance has unique integer ID

### 2. Segmentation Modes

**MedSAM:**
- Prompt-based only
- Modes: auto_box, full_box, point

**MicroSAM:**
- Interactive: Box and point prompts
- Automatic: No prompts required (APG/AIS/AMG modes)

### 3. Model Architecture

**MedSAM:**
- Based on original SAM
- Generic models (vit_b, vit_l, vit_h)

**MicroSAM:**
- Optimized for microscopy/histopathology
- Specialized models (vit_b_histopathology, vit_l_histopathology)

### 4. Tiling

**MedSAM:**
- Manual tiling implementation

**MicroSAM:**
- Built-in tiled inference
- Automatic halo stitching
- Optimized for large images

## Migration Steps

### 1. Update Dependencies

```bash
# Remove old environment (optional)
deactivate
rm -rf venv/

# Create new environment
python3 -m venv venv
source venv/bin/activate

# Install new requirements
pip install -r requirements.txt
```

### 2. Update Configuration

```bash
# Update .env file
cp .env.example .env
nano .env

# Remove old settings:
# - MEDSAM_CHECKPOINT
# - MODEL_TYPE

# Add new settings:
# - MICROSAM_MODEL_TYPE=vit_b_histopathology
# - ENABLE_TILING=true
# - TILE_SHAPE=1024,1024
# - HALO_SIZE=256,256
```

### 3. Update Code (if using API)

If you have custom scripts using the API, update imports and method calls as shown above.

### 4. Test

```bash
# Run test script
python scripts/01_microsam_auto_test.py

# Or start the app
streamlit run app.py
```

## Features Comparison

| Feature | MedSAM | MicroSAM |
|---------|--------|----------|
| Prompt-based segmentation | ✅ | ✅ |
| Automatic segmentation | ❌ | ✅ |
| Instance segmentation | ❌ | ✅ |
| Histopathology-optimized | ❌ | ✅ |
| Built-in tiling | ❌ | ✅ |
| Embeddings cache | ❌ | ✅ |
| Model auto-download | ❌ | ✅ |
| GeoJSON export | ✅ | ✅ |

## Troubleshooting

### "micro_sam not found"

```bash
pip install micro-sam
```

### Models not downloading

Check network connection and set cache directory:
```bash
export MICROSAM_CACHEDIR=/path/to/cache
```

### Out of memory errors

Reduce tile size:
```bash
TILE_SHAPE=512,512
```

### Slow inference

Enable embeddings cache:
```bash
ENABLE_EMBEDDINGS_CACHE=true
```

## Getting Help

- GitHub Issues: https://github.com/eisascience/XHistoSegMicroSAM/issues
- MicroSAM Docs: https://computational-cell-analytics.github.io/micro-sam/

## Backward Compatibility

**Note:** This migration intentionally removes MedSAM support. There is no backward compatibility. If you need to use both models, maintain separate branches or installations.
