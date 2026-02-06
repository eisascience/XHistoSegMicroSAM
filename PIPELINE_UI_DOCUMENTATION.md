# Pipeline Framework UI Changes

## MicroSAM Analysis Tab - New Interface

### 1. Analysis Mode Selection

When users navigate to the **MicroSAM Analysis** tab, they will see:

```
MicroSAM Analysis
─────────────────────────────────────────────────────────

### Analysis Mode

┌─────────────┬───────────────────────────────────────────────┐
│ ○ Classic   │ ℹ️ Classic Mode: Standard segmentation       │
│ ● Pipeline  │    workflow with post-processing options      │
└─────────────┴───────────────────────────────────────────────┘

─────────────────────────────────────────────────────────
```

### 2. Pipeline Mode Interface

When **Pipeline** mode is selected:

```
## Pipeline Selection

┌──────────────────────┬────────────────────────────────────────┐
│ Select Pipeline:     │ Description: Nucleus-guided cell       │
│ ▼ Multi-Channel      │ segmentation with compartmental        │
│   Hierarchical       │ analysis                               │
│                      │ Version: 1.0.0                         │
│                      │ Author: XHistoSeg Team                 │
└──────────────────────┴────────────────────────────────────────┘

▼ Pipeline Details
  Required Channels: nucleus
  Optional Channels: cell_marker, signal

─────────────────────────────────────────────────────────

## Pipeline Configuration

### Channel Assignment
Assign roles to your image channels

Nucleus Channel (e.g., DAPI):        ▼ Channel_0
Cell Marker Channels (e.g., CD5):    □ Channel_1
                                     □ Channel_2
Signal Channels (e.g., vRNA):        □ Channel_3
                                     □ Channel_4

### Segmentation Parameters

Nucleus Segmentation Mode:           ▼ auto_box_from_threshold
Cell Segmentation Mode:              ▼ point
☑ Enable compartmental analysis (nuclear vs cytoplasmic)

─────────────────────────────────────────────────────────

## Image Selection

Select image to process:             ▼ image001.tif

┌─────────────────────┬────────────────────────┐
│   🚀 Run Pipeline   │   🗑️ Clear Results    │
└─────────────────────┴────────────────────────┘

─────────────────────────────────────────────────────────

## Pipeline Results

ℹ️ Pipeline: Multi-Channel Hierarchical | Image: image001.tif

Multi-Channel Segmentation Results
───────────────────────────────────

┌─────────────────┬─────────────────┬─────────────────┐
│  Nuclei         │  Cell           │  Objects        │
│  Detected       │  Channels       │  Measured       │
│                 │                 │                 │
│  1,247          │  2              │  1,247          │
└─────────────────┴─────────────────┴─────────────────┘

### Segmentation Masks

[Nuclei]          [Channel_1]       [Channel_2]
[Mask Image]      [Mask Image]      [Mask Image]

### Compartmental Measurements

nucleus_id  Channel_3_nuclear  Channel_3_cytoplasmic  Channel_3_ratio
1           125.3              98.2                   1.28
2           142.7              105.1                  1.36
3           118.9              92.4                   1.29
...

─────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────┐
│             📊 Export Pipeline Results              │
└─────────────────────────────────────────────────────┘
```

### 3. Classic Mode Interface

When **Classic** mode is selected, the original interface is displayed with all existing functionality unchanged.

## Key Features

1. **Mode Toggle**: Radio buttons to switch between Classic and Pipeline modes
2. **Pipeline Selection**: Dropdown with available pipelines
3. **Pipeline Info**: Expandable details about selected pipeline
4. **Dynamic Configuration**: Pipeline-specific UI controls
5. **Image Selection**: Choose from uploaded images
6. **Results Display**: Pipeline-specific visualizations
7. **Export**: Pipeline-specific export formats

## Benefits

- **Backward Compatible**: Classic mode preserves all existing functionality
- **Progressive Enhancement**: Users can explore pipeline mode when ready
- **Clear Separation**: Mode toggle makes it clear which interface is active
- **Intuitive**: Pipeline mode guides users through configuration steps
- **Flexible**: Each pipeline can define its own UI and workflow

## Example Use Cases

### Basic Single Channel Pipeline
- Works with any image (grayscale, RGB, multi-channel)
- Standard segmentation workflow
- Same as classic mode but via pipeline framework

### Multi-Channel Hierarchical Pipeline
- Requires multi-channel images
- Nucleus detection → Cell segmentation
- Compartmental analysis (nuclear vs cytoplasmic)
- Per-object measurements with ratios

### Future Pipelines (Examples)
- **Infected Macrophages**: SIV/macrophage co-localization analysis
- **Phagocytosis Detection**: Cell-particle interaction analysis
- **Tissue Classification**: Multi-region tissue type classification
- **Custom Biology**: User-defined workflows for specific research questions
