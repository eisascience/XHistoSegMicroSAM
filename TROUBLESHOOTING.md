# Troubleshooting Guide

This guide documents common installation and runtime issues encountered with XHistoSegMicroSAM.

## Table of Contents
- [Python Version Requirements](#python-version-requirements)
- [Installation Issues](#installation-issues)
- [Runtime Issues](#runtime-issues)
- [Model Download Issues](#model-download-issues)
- [Performance Issues](#performance-issues)

## Python Version Requirements

### Why Python 3.9 Only?

**The Dependency Deadlock:**

XHistoSegMicroSAM requires automatic instance segmentation features from micro-sam, which creates an impossible dependency situation with Python 3.10+:

1. **micro-sam v1.7.1+** (latest versions):
   - Requires Python >=3.10
   - Has hard import of `python-elf` module at the module level
   - Includes histopathology models we need

2. **python-elf** (required by micro-sam):
   - Depends on `numba` which depends on `llvmlite`
   - `llvmlite` 0.36 (required by numba for Python <3.10) only supports Python <3.10
   - `llvmlite` 0.40+ (for Python 3.10+) is incompatible with the version of numba that python-elf needs

3. **The Solution:**
   - **micro-sam v1.3.0** is the last version that:
     - Supports Python 3.9
     - Includes histopathology models (vit_b_histopathology, vit_l_histopathology)
     - Works with python-elf in Python 3.9

**Bottom line:** Python 3.9 with micro-sam v1.3.0 is the ONLY working configuration.

### Tested Non-Working Configurations

❌ Python 3.11 + micro-sam v1.7.1 → python-elf cannot be installed  
❌ Python 3.10 + micro-sam v1.7.1 → python-elf cannot be installed  
❌ Python 3.9 + micro-sam v1.7.1 → Works but histopathology models may have changed

✅ Python 3.9 + micro-sam v1.3.0 → **WORKS**

## Installation Issues

### Error: "No module named 'elf'"

**Symptom:**
```
ModuleNotFoundError: No module named 'elf'
```

**Cause:**
`python-elf` is not available via pip and must be installed through conda.

**Solution:**
```bash
conda install -c conda-forge python-elf -y
```

**Why this happens:**
- `python-elf` is a binary package with C extensions
- It's only distributed through conda-forge
- pip installation is not supported

### Error: "No module named 'vigra'"

**Symptom:**
```
ModuleNotFoundError: No module named 'vigra'
```

**Cause:**
`vigra` (Vision with Generic Algorithms) is required by micro-sam for image processing but is only available through conda.

**Solution:**
```bash
conda install -c conda-forge vigra -y
```

### Error: "No module named 'nifty'"

**Symptom:**
```
ModuleNotFoundError: No module named 'nifty'
```

**Cause:**
`nifty` is a graph library required by micro-sam for instance segmentation.

**Solution:**
```bash
conda install -c conda-forge nifty -y
```

### Error: "No module named 'torch_em'"

**Symptom:**
```
ModuleNotFoundError: No module named 'torch_em'
```

**Cause:**
`torch-em` (PyTorch for microscopy) is a dependency of micro-sam but not automatically installed.

**Solution:**
```bash
pip install git+https://github.com/constantinpape/torch-em.git
```

**Note:** torch-em is not on PyPI, must be installed from GitHub.

### Error: NumPy 2.x Compatibility Issues

**Symptom:**
```
AttributeError: module 'numpy' has no attribute 'float'
# OR
TypeError: 'numpy.float_' object cannot be interpreted as an integer
```

**Cause:**
NumPy 2.0+ removed many deprecated aliases. Some dependencies (opencv-python, scipy) have issues with NumPy 2.x.

**Solution:**
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

**Why we pin numpy<2.0:**
- opencv-python versions <4.10.0 have compatibility issues with NumPy 2.x
- We also pin opencv-python<4.10.0 to ensure compatibility

### Error: "File 'vit_b_histopathology' is not in the registry"

**Symptom:**
```
ValueError: File 'vit_b_histopathology' is not in the registry
```

**Cause:**
Using the wrong micro-sam version. Histopathology models were added in v1.2.0+ and available through v1.3.0.

**Solution:**
Ensure you're installing the correct micro-sam version:
```bash
pip install git+https://github.com/computational-cell-analytics/micro-sam.git@v1.3.0
```

### Missing Dependencies: xxhash, kornia, pooch, imageio

**Symptoms:**
```
ModuleNotFoundError: No module named 'xxhash'
ModuleNotFoundError: No module named 'kornia'
ModuleNotFoundError: No module named 'pooch'
ModuleNotFoundError: No module named 'imageio'
```

**Cause:**
These are indirect dependencies of micro-sam that aren't always automatically installed.

**Solution:**
These are included in `requirements-py39.txt`. Install them:
```bash
pip install xxhash kornia pooch imageio h5py napari-plugin-engine
```

### UV Installation Failures

**Problem:**
The old `requirements-uv.txt` with Python 3.11 does NOT work.

**Why:**
- Designed for Python 3.11
- Tries to install micro-sam v1.7.1 which requires python-elf
- python-elf cannot be installed in Python 3.11

**Solution:**
Do NOT use uv or requirements-uv.txt. Use conda with Python 3.9 and `requirements-py39.txt` instead.

## Runtime Issues

### Empty Masks / No Segmentation Results

**Symptoms:**
- Segmentation completes but returns empty masks
- No objects detected
- Zero instances in output

**Causes & Solutions:**

1. **Wrong channel selected:**
   - Make sure you're segmenting the correct channel
   - For nuclei: use DAPI or Hoechst channel (usually blue/channel 0)
   - For cells: use membrane or cytoplasm channel

2. **Threshold too high/low:**
   - For auto_box_from_threshold mode: adjust threshold value
   - Try Otsu automatic thresholding first
   - If Otsu fails, switch to manual and adjust slider

3. **Box filtering too aggressive:**
   - Check min_area and max_area parameters
   - Lower min_area if small objects are missed
   - Increase max_area if large objects are filtered out

4. **Model not suitable:**
   - Try different model: vit_b_histopathology (faster) vs vit_l_histopathology (more accurate)
   - Ensure you're using histopathology-specific models, not generic SAM

### Out of Memory (OOM) Errors

**Symptom:**
```
RuntimeError: CUDA out of memory
# OR
Killed (process runs out of RAM)
```

**Solutions:**

1. **Enable tiling** (in .env):
   ```bash
   ENABLE_TILING=true
   TILE_SHAPE=1024,1024
   HALO_SIZE=256,256
   ```

2. **Use smaller model:**
   ```bash
   MICROSAM_MODEL_TYPE=vit_b_histopathology  # instead of vit_l
   ```

3. **Reduce image size:**
   - Downsample large whole-slide images
   - Process regions of interest (ROI) instead of entire slide

4. **Disable embeddings cache** (if enabled):
   ```bash
   ENABLE_EMBEDDINGS_CACHE=false
   ```

5. **Increase system swap:**
   ```bash
   # Linux
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Slow Inference

**Symptoms:**
- Segmentation takes very long (>5 minutes for small images)
- UI becomes unresponsive

**Solutions:**

1. **Check if GPU is being used:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True if CUDA GPU available
   print(torch.backends.mps.is_available())  # Should be True if Apple Silicon
   ```

2. **Ensure PyTorch with GPU support:**
   ```bash
   # For CUDA
   pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   
   # For ROCm (AMD)
   pip install torch==2.6.0+rocm5.7 torchvision==0.21.0+rocm5.7 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. **Use smaller tile size:**
   ```bash
   TILE_SHAPE=512,512  # instead of 1024,1024
   ```

4. **Use faster model:**
   ```bash
   MICROSAM_MODEL_TYPE=vit_b_histopathology  # instead of vit_l
   ```

## Model Download Issues

### Models Not Downloading

**Symptom:**
- Application hangs on first run
- No progress indication
- Eventually times out

**Cause:**
Models are downloaded automatically on first use but download progress isn't always visible.

**Solution:**

1. **Check model cache location:**
   ```bash
   echo $MICROSAM_CACHEDIR
   # Default: ~/.cache/micro_sam/models
   ```

2. **Pre-download models manually:**
   ```python
   from micro_sam.util import get_sam_model
   
   model_type = "vit_b_histopathology"
   model = get_sam_model(model_type=model_type)
   print(f"Model {model_type} downloaded successfully")
   ```

3. **Check network connectivity:**
   - Models are downloaded from GitHub/Zenodo
   - Ensure firewall allows HTTPS connections
   - Try from a different network if corporate firewall blocks

4. **Set custom cache directory:**
   ```bash
   export MICROSAM_CACHEDIR="/path/to/cache"
   ```

### Model Loading Errors

**Symptom:**
```
RuntimeError: Error loading model checkpoint
# OR
OSError: Unable to load weights from pytorch checkpoint
```

**Solutions:**

1. **Clear corrupted cache:**
   ```bash
   rm -rf ~/.cache/micro_sam/models/*
   # Or if custom location:
   rm -rf $MICROSAM_CACHEDIR/*
   ```

2. **Re-download models:**
   - Application will automatically re-download on next run

3. **Check disk space:**
   ```bash
   df -h ~/.cache/micro_sam/models
   ```
   - Models are ~300-400 MB each
   - Ensure at least 1 GB free space

## Performance Issues

### Application Startup Slow

**Cause:**
Streamlit loads all imports at startup, including heavy ML libraries.

**Solution:**
This is normal. First startup takes 10-30 seconds. Subsequent runs are faster.

### Embeddings Cache Not Working

**Symptom:**
- Re-processing same image is just as slow as first time
- Cache directory is empty or not being created

**Solutions:**

1. **Check cache configuration in .env:**
   ```bash
   ENABLE_EMBEDDINGS_CACHE=true
   EMBEDDINGS_CACHE_DIR=./cache/embeddings
   ```

2. **Ensure cache directory is writable:**
   ```bash
   mkdir -p ./cache/embeddings
   chmod 755 ./cache/embeddings
   ```

3. **Check cache is being used:**
   - Look for `.zarr` files in cache directory after first run
   - File names are hashes of image + parameters

4. **Clear corrupted cache:**
   ```bash
   rm -rf ./cache/embeddings/*
   ```

## Getting More Help

If you encounter issues not covered here:

1. **Check the logs:**
   - Streamlit logs are printed to console
   - Look for stack traces and error messages

2. **Enable debug mode** (in .env):
   ```bash
   DEBUG=true
   ```

3. **Test with minimal example:**
   ```bash
   python scripts/01_microsam_auto_test.py
   ```

4. **Verify installation:**
   ```bash
   python -c "from xhalo.ml import MicroSAMPredictor; print('Installation successful!')"
   ```

5. **Check versions:**
   ```bash
   python --version  # Should be 3.9.x
   pip show micro-sam torch numpy
   conda list | grep -E "elf|vigra|nifty"
   ```

6. **Open an issue:**
   - Include Python version, OS, error message
   - Include output of version checks above
   - Describe what you were trying to do
