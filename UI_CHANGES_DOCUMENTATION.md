# UI Changes for Python 3.11 Compatibility

## New Prompt Mode: auto_box_from_threshold

The new `auto_box_from_threshold` mode provides excellent cell/nucleus segmentation without requiring `python-elf`.

### Prompt Mode Selection
```
Prompt Mode dropdown now includes:
├── auto_box                    (Auto-detect tissue region)
├── auto_box_from_threshold    (⭐ Recommended for nuclei - NEW!)
├── full_box                    (Use entire image)
└── point                       (Use center point)
```

### Threshold-based Box Generation Settings

When `auto_box_from_threshold` is selected, users see:

```
┌─────────────────────────────────────────────────────────────┐
│ Threshold-based Box Generation Settings                     │
│ ▼ Configure Threshold Parameters                            │
│ ┌─────────────────────┬────────────────────────────────────┐│
│ │ Threshold Mode      │ Min Box Area (px)                  ││
│ │ ○ otsu (selected)   │ [100        ] ▲▼                   ││
│ │ ○ manual            │                                     ││
│ │ ○ off               │ Max Box Area (px)                  ││
│ │                     │ [100000     ] ▲▼                   ││
│ │ (If manual selected)│                                     ││
│ │ Threshold Value     │                                     ││
│ │ ━━━━━━━●━━━━━ 0.50  │                                     ││
│ └─────────────────────┴────────────────────────────────────┘│
│                                                              │
│ Dilation Radius                                              │
│ ━━●━━━━━━━━━━━━ 0                                           │
│ (Pixels to dilate mask before boxing)                       │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

1. **Select analysis channel** (e.g., DAPI in Channels page)
2. **Normalize and preprocess** (percentile clipping, smoothing)
3. **Choose auto_box_from_threshold mode**
4. **Configure threshold**:
   - Otsu: Automatic threshold
   - Manual: Adjust slider
   - Off: Use full channel
5. **Set size filters**:
   - Min area: Filter out noise
   - Max area: Filter out large regions
6. **Optional dilation**: Capture full nuclei borders
7. **Run Analysis** → Get instance segmentation with unique IDs

### elf Availability Info Message

If `python-elf` is not available, users see:

```
ℹ️ python-elf is not available. Automatic instance segmentation 
   (APG/AIS) is disabled.

Available modes:
- point: Interactive point prompts
- auto_box: Auto-detect tissue bounding box
- auto_box_from_threshold: Generate boxes from thresholded channel 
  (recommended for nuclei)
- full_box: Use entire image

To enable automatic modes:
Create a conda environment with Python <3.10 and install:

conda create -n microsam-auto python=3.9
conda activate microsam-auto
conda install -c conda-forge python-elf
```

## Code Quality Improvements

### Before: Duplicated Code (120+ lines × 2)
```python
# In analysis_page() - local mode
if prompt_mode == "auto_box_from_threshold":
    st.write("**Threshold-based Box Generation Settings**")
    with st.expander("Configure Threshold Parameters", expanded=True):
        # 60+ lines of UI code...

# In analysis_page() - Halo mode  
if prompt_mode == "auto_box_from_threshold":
    st.write("**Threshold-based Box Generation Settings**")
    with st.expander("Configure Threshold Parameters", expanded=True):
        # 60+ lines of DUPLICATE UI code...
```

### After: Reusable Helper Function
```python
# Single helper function
def render_threshold_params_ui() -> Dict[str, Any]:
    """Render UI controls for threshold-based box generation."""
    # 60+ lines of UI code (defined once)
    return threshold_params

# Used in both locations
if prompt_mode == "auto_box_from_threshold":
    threshold_params = render_threshold_params_ui()
```

**Result**: Reduced code duplication by 120+ lines ✅

## Key Features

✅ **No crashes**: Graceful fallback when elf unavailable  
✅ **Clear guidance**: Info messages explain available modes  
✅ **Powerful alternative**: auto_box_from_threshold for nuclei segmentation  
✅ **User-friendly**: Intuitive controls with helpful tooltips  
✅ **Maintainable**: Reduced code duplication  
✅ **Well-documented**: README updated with full instructions
