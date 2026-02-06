# Pipeline Development Guide

## Overview

Pipelines define complete analysis workflows for XHistoSegMicroSAM, including:
- Channel requirements and assignments
- Segmentation strategy
- Measurements and analysis
- Visualization
- Data export formats

## Creating a Custom Pipeline

### 1. Create a New Pipeline File

Create a new file in `pipelines/` (e.g., `my_pipeline.py`)

### 2. Import BasePipeline

```python
from .base import BasePipeline
import numpy as np
from typing import Dict, List, Any
```

### 3. Implement Required Methods

Your pipeline must implement all abstract methods from `BasePipeline`:

#### `configure_ui(self, st) -> Dict[str, Any]`

Render pipeline-specific UI controls using Streamlit.

```python
def configure_ui(self, st):
    """UI for pipeline parameters"""
    config = {}
    
    config['param1'] = st.slider("Parameter 1", 0, 100, 50)
    config['param2'] = st.selectbox("Parameter 2", ['option1', 'option2'])
    
    return config
```

#### `validate_channels(self, available_channels: List[str]) -> bool`

Check if the image has required channels.

```python
def validate_channels(self, available_channels):
    """Check if required channels are available"""
    return 'nucleus' in available_channels
```

#### `process(self, image, channels, predictor, config) -> Dict[str, Any]`

Run the analysis pipeline.

```python
def process(self, image, channels, predictor, config):
    """Run analysis"""
    # Extract channels
    nucleus_img = channels['nucleus']
    
    # Run segmentation
    mask = predictor.predict(nucleus_img, ...)
    
    # Return results
    return {
        'mask': mask,
        'statistics': {...}
    }
```

#### `visualize(self, results, st)`

Display results using Streamlit.

```python
def visualize(self, results, st):
    """Display results"""
    st.subheader("Results")
    st.metric("Objects Detected", results['num_objects'])
    st.image(results['mask'], caption="Segmentation Mask")
```

#### `export_data(self, results) -> Dict[str, Any]`

Export results in standard formats.

```python
def export_data(self, results):
    """Export results"""
    import pandas as pd
    
    return {
        'statistics_csv': pd.DataFrame(results['statistics']),
        'mask': results['mask']
    }
```

## Example Template

```python
from .base import BasePipeline
import numpy as np
from typing import Dict, List, Any


class MyPipeline(BasePipeline):
    """
    Custom pipeline description.
    """
    
    name = "My Pipeline"
    description = "Description of what this pipeline does"
    version = "1.0.0"
    author = "Your Name"
    
    required_channels = ['nucleus']
    optional_channels = ['signal']
    
    def configure_ui(self, st):
        config = {}
        config['threshold'] = st.slider("Threshold", 0.0, 1.0, 0.5)
        return config
    
    def validate_channels(self, available_channels):
        return 'nucleus' in available_channels
    
    def process(self, image, channels, predictor, config):
        # Your analysis logic here
        results = {}
        return results
    
    def visualize(self, results, st):
        # Your visualization code here
        st.write("Results")
    
    def export_data(self, results):
        # Your export logic here
        return {}
```

## Registering Your Pipeline

Add your pipeline to `__init__.py`:

```python
from .my_pipeline import MyPipeline

AVAILABLE_PIPELINES = {
    'basic': BasicSingleChannelPipeline,
    'multi_channel': MultiChannelHierarchicalPipeline,
    'my_pipeline': MyPipeline,  # Add your pipeline here
}
```

## Testing Your Pipeline

Test your pipeline locally:

```python
from pipelines import get_pipeline

# Get your pipeline
pipeline = get_pipeline('my_pipeline')

# Test configuration
config = {...}  # Simulate UI config

# Test processing
results = pipeline.process(image, channels, predictor, config)

# Test visualization
# (would need Streamlit context)

# Test export
exports = pipeline.export_data(results)
```

## Best Practices

1. **Keep pipelines focused**: Each pipeline should address a specific analysis workflow
2. **Document well**: Add docstrings to explain what your pipeline does
3. **Handle errors gracefully**: Use try-except blocks for robust error handling
4. **Validate inputs**: Check that channels and parameters are valid
5. **Export useful data**: Include measurements, masks, and metadata in exports
6. **Provide clear visualizations**: Help users understand results

## Examples

### Basic Single Channel Pipeline

See `basic_single_channel.py` for a minimal example that works with any image.

### Multi-Channel Hierarchical Pipeline

See `multi_channel_hierarchical.py` for a more complex example with:
- Multiple channel assignments
- Hierarchical object detection (nucleus -> cell)
- Compartmental analysis
- Relationship tracking

## Pipeline Metadata

Set class attributes to provide metadata:

```python
class MyPipeline(BasePipeline):
    name = "My Pipeline"                    # Display name
    description = "What it does"            # Brief description
    version = "1.0.0"                       # Semantic version
    author = "Your Name"                    # Author/maintainer
    required_channels = ['nucleus']         # Must have these channels
    optional_channels = ['signal']          # Optional channels
```

## Channel Access

Channels are provided as a dictionary in the `process()` method:

```python
def process(self, image, channels, predictor, config):
    # Access by name
    nucleus = channels['nucleus']
    signal = channels.get('signal', None)  # Optional channel
    
    # channels is Dict[str, np.ndarray]
    # Each channel is a 2D numpy array (H, W)
```

## Using the MicroSAM Predictor

The predictor is provided in the `process()` method:

```python
def process(self, image, channels, predictor, config):
    # Convert to RGB for MicroSAM
    nucleus_rgb = np.stack([nucleus]*3, axis=-1)
    
    # Run prediction
    mask = predictor.predict(
        nucleus_rgb,
        prompt_mode='auto_box',
        multimask_output=False
    )
```

Available prompt modes:
- `'auto_box'` - Automatic tissue detection
- `'auto_box_from_threshold'` - Threshold-based detection
- `'full_box'` - Use entire image
- `'point'` - Point prompts

## Tips

- Start with `basic_single_channel.py` as a template
- Use `st.spinner()` for long-running operations
- Cache results in `st.session_state` if needed
- Provide progress indicators for multi-step workflows
- Test with different image types and sizes

## Support

For questions or issues:
1. Check existing pipelines for examples
2. Review the `BasePipeline` class documentation
3. Open an issue on GitHub
