#!/usr/bin/env python3
"""
Test script for MicroSAM automatic instance segmentation.

Tests the MicroSAM backend with a simple test image and saves:
- Instance segmentation mask (16-bit PNG with instance IDs)
- Color visualization of instances
- GeoJSON export

Usage:
    python scripts/01_microsam_auto_test.py [image_path]
"""

import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xhalo.ml import MicroSAMPredictor
from utils.geojson_utils import instance_mask_to_polygons, instance_polygons_to_geojson, save_geojson

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(size=(512, 512)):
    """
    Create a simple test image with synthetic objects.
    
    Returns:
        RGB image array (H, W, 3)
    """
    logger.info("Creating synthetic test image")
    img = np.ones((*size, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw some simple shapes as test objects
    # Circle 1
    center1 = (size[0]//4, size[1]//4)
    radius1 = 50
    y, x = np.ogrid[:size[0], :size[1]]
    mask1 = (x - center1[1])**2 + (y - center1[0])**2 <= radius1**2
    img[mask1] = [150, 50, 50]  # Dark red
    
    # Circle 2
    center2 = (size[0]//4, 3*size[1]//4)
    radius2 = 60
    mask2 = (x - center2[1])**2 + (y - center2[0])**2 <= radius2**2
    img[mask2] = [50, 150, 50]  # Dark green
    
    # Circle 3
    center3 = (3*size[0]//4, size[1]//2)
    radius3 = 70
    mask3 = (x - center3[1])**2 + (y - center3[0])**2 <= radius3**2
    img[mask3] = [50, 50, 150]  # Dark blue
    
    return img


def visualize_instances(image, instance_mask, output_path):
    """
    Create a color visualization of instance segmentation.
    
    Args:
        image: Original RGB image
        instance_mask: Instance mask with 0=background, 1..N=instances
        output_path: Path to save visualization
    """
    logger.info(f"Creating instance visualization: {output_path}")
    
    # Get unique instance IDs
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    num_instances = len(instance_ids)
    
    # Create colormap
    np.random.seed(42)  # For reproducibility
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max(20, num_instances)))
    
    # Create color overlay
    overlay = image.copy()
    for idx, instance_id in enumerate(instance_ids):
        mask = instance_mask == instance_id
        color = (colors[idx % len(colors)][:3] * 255).astype(np.uint8)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Instance mask
    axes[1].imshow(instance_mask, cmap='nipy_spectral', interpolation='nearest')
    axes[1].set_title(f'Instance Mask ({num_instances} instances)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Instance Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")


def main():
    """Main test function."""
    # Parse arguments
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)
        logger.info(f"Loading image from: {image_path}")
        image = np.array(Image.open(image_path).convert('RGB'))
    else:
        logger.info("No image provided, using synthetic test image")
        image = create_test_image()
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save input image
    input_path = output_dir / "input_image.png"
    Image.fromarray(image).save(input_path)
    logger.info(f"Saved input image to {input_path}")
    
    # Initialize MicroSAM predictor
    logger.info("Initializing MicroSAM predictor...")
    try:
        predictor = MicroSAMPredictor(
            model_type="vit_b_histopathology",
            tile_shape=(1024, 1024),
            halo=(256, 256)
        )
    except Exception as e:
        logger.error(f"Failed to initialize MicroSAM: {e}")
        logger.error("Make sure micro-sam is installed: pip install micro-sam")
        sys.exit(1)
    
    # Run automatic instance segmentation
    logger.info("Running automatic instance segmentation...")
    try:
        instance_mask = predictor.predict_auto_instances(
            image=image,
            segmentation_mode="apg",
            min_size=25,
            foreground_threshold=0.5,
            tiled=True
        )
        
        # Get instance statistics
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]
        num_instances = len(instance_ids)
        
        logger.info(f"Segmentation complete: {num_instances} instances found")
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save instance mask as 16-bit PNG
    mask_path = output_dir / "instance_mask.png"
    Image.fromarray(instance_mask.astype(np.uint16)).save(mask_path)
    logger.info(f"Saved instance mask to {mask_path}")
    
    # Create and save visualization
    viz_path = output_dir / "instance_visualization.png"
    visualize_instances(image, instance_mask, viz_path)
    
    # Convert to GeoJSON
    logger.info("Converting to GeoJSON...")
    try:
        polygons_with_ids = instance_mask_to_polygons(instance_mask, min_area=50)
        geojson = instance_polygons_to_geojson(
            polygons_with_ids,
            simplify=True,
            tolerance=1.0
        )
        
        geojson_path = output_dir / "instance_segmentation.geojson"
        save_geojson(geojson, str(geojson_path))
        
        logger.info(f"Saved GeoJSON with {len(geojson['features'])} features to {geojson_path}")
        
    except Exception as e:
        logger.error(f"GeoJSON export failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 50)
    logger.info("Test completed successfully!")
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
