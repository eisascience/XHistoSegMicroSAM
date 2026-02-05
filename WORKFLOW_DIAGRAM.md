# XHaloPathAnalyzer Workflow Diagram

## Mode Selection Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Start Application │
│ streamlit run app.py │
└────────────────────────┬────────────────────────────────────┘
 │
 ▼
 ┌───────────────────────────────────┐
 │ Authentication / Mode Selection │
 └───────────┬───────────────┬───────┘
 │ │
 ┌────────────▼────────┐ │
 │ Halo API Mode │ │
 │ (Original) │ │
 └────────────┬────────┘ │
 │ │
 ┌──────────▼──────────┐ │
 │ Enter Credentials │ │
 │ - API Endpoint │ │
 │ - API Token │ │
 └──────────┬──────────┘ │
 │ │
 ┌─────────▼────────┐ │
 │ Test Connection │ │
 └─────────┬────────┘ │
 │ │
 ┌────────────▼────────┐ │
 │ Local Mode │◄────┘
 │ (NEW!) │
 └────────────┬────────┘
 │
 ┌───────────▼───────────┐
 │ No credentials │
 │ needed │
 └───────────┬───────────┘
 │
 ▼
 ┌───────────────────────┐
 │ Authenticated │
 │ Proceed to Analysis │
 └───────────────────────┘
```

## Halo API Mode Workflow (Original)

```
┌──────────────────────────────────────────────────────────────┐
│ Halo API Mode │
└───────────────────────────┬──────────────────────────────────┘
 │
 ┌──────────────▼──────────────┐
 │ Slide Selection │
 │ - Browse Halo slides │
 │ - Filter by name/study │
 │ - View slide metadata │
 └──────────────┬──────────────┘
 │
 ┌───────▼───────┐
 │ Select Slide │
 └───────┬───────┘
 │
 ┌──────────────▼──────────────┐
 │ Analysis │
 │ - Define ROI (x, y, w, h) │
 │ - Download from Halo │
 │ - Run MedSAM │
 │ - View results │
 └──────────────┬──────────────┘
 │
 ┌──────────────▼──────────────┐
 │ Export │
 │ - Generate GeoJSON │
 │ - Download results │
 └──────────────┬──────────────┘
 │
 ┌──────────────▼──────────────┐
 │ Import to Halo │
 │ - Manual instructions │
 └─────────────────────────────┘
```

## Local Mode Workflow (NEW!)

```
┌──────────────────────────────────────────────────────────────┐
│ Local Image Mode │
└───────────────────────────┬──────────────────────────────────┘
 │
 ┌──────────────▼──────────────┐
 │ Image Upload │
 │ - Upload JPG/PNG/TIFF │
 │ - View uploaded list │
 │ - See file details │
 │ - Preview image │
 └──────────────┬──────────────┘
 │
 ┌───────▼───────┐
 │ Select Image │
 └───────┬───────┘
 │
 ┌───────▼───────┐
 │ Load Image │
 └───────┬───────┘
 │
 ┌──────────────▼──────────────┐
 │ Analysis │
 │ - Image pre-loaded │
 │ - Optional ROI selection │
 │ - Run MedSAM │
 │ - View results │
 └──────────────┬──────────────┘
 │
 ┌──────────────▼──────────────┐
 │ Export │
 │ - Generate GeoJSON │
 │ - Download mask as PNG │
 │ - Download GeoJSON │
 └─────────────────────────────┘
```

## Analysis Pipeline (Both Modes)

```
┌─────────────────────────────────────────────────────────────┐
│ Analysis Pipeline │
│ (Shared by both modes) │
└────────────────────────────┬────────────────────────────────┘
 │
 ┌──────────▼──────────┐
 │ Get Image │
 │ Halo: Download │
 │ Local: Pre-loaded │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Preprocess │
 │ - Resize │
 │ - Normalize │
 │ - Pad to 1024x1024 │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Initialize MedSAM │
 │ - Load model │
 │ - Select device │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Run Inference │
 │ - Segment tissue │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Postprocess │
 │ - Unpad │
 │ - Resize to orig │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Compute Stats │
 │ - Coverage % │
 │ - Area metrics │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Visualize │
 │ - Original │
 │ - Mask │
 │ - Overlay │
 └─────────────────────┘
```

## Export Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Export Pipeline │
└────────────────────────────┬────────────────────────────────┘
 │
 ┌──────────▼──────────┐
 │ Configure Export │
 │ - Classification │
 │ - Min area │
 │ - Simplification │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Extract Polygons │
 │ - Find contours │
 │ - Filter by area │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Generate GeoJSON │
 │ - Create features │
 │ - Add properties │
 │ - Simplify shapes │
 └──────────┬──────────┘
 │
 ┌──────────▼──────────┐
 │ Save & Download │
 │ - Write to file │
 │ - Offer download │
 └─────────────────────┘
```

## UI Components

### Navigation (Dynamic based on mode)

**Halo API Mode:**
```
├── Slide Selection
├── Analysis 
├── Export
├── Import
└── Settings
```

**Local Mode:**
```
├── Image Upload ← NEW!
├── Analysis
├── Export
└── Settings
```

## File Structure

```
XHaloPathAnalyzer/
├── app.py # Main application
│ ├── init_session_state() # Session initialization
│ ├── authentication_page() # Mode selection
│ ├── slide_selection_page() # Halo mode
│ ├── image_upload_page() # Local mode (NEW!)
│ ├── analysis_page() # Both modes
│ ├── export_page() # Both modes 
│ ├── import_page() # Halo mode
│ └── main() # Router
│
├── config.py # Configuration
│ ├── LOCAL_MODE flag # NEW!
│ └── validate(require_halo_api) # Modified
│
├── utils/
│ ├── halo_api.py # Halo integration
│ ├── image_proc.py # Image processing
│ ├── ml_models.py # MedSAM wrapper
│ └── geojson_utils.py # GeoJSON export
│
├── tests/
│ └── test_config.py # Updated tests
│
├── validate_local_mode.py # Validation script
└── LOCAL_MODE_IMPLEMENTATION.md # This document
```

## Key Decision Points in Code

### Mode Detection
```python
is_local_mode = st.session_state.local_mode or slide['id'].startswith('local_')
```

### Image Source Selection
```python
if is_local_mode and st.session_state.current_image is not None:
 # Use pre-loaded image
 image = st.session_state.current_image
else:
 # Download from Halo API
 region_data = st.session_state.api.download_region(...)
 image = load_image_from_bytes(region_data)
```

### Navigation Routing
```python
if st.session_state.local_mode:
 nav_options = ["Image Upload", "Analysis", "Export", "Settings"]
else:
 nav_options = ["Slide Selection", "Analysis", "Export", "Import", "Settings"]
```

## Benefits Summary

### For Users
- Quick testing without Halo setup
- Analyze local images instantly
- No API credentials needed
- Batch processing capability

### For Developers 
- Maintains code simplicity
- No breaking changes
- Shared analysis pipeline
- Easy to test and debug

### For Research
- Rapid prototyping
- Offline analysis
- Flexible workflows
- Standard formats (GeoJSON)
