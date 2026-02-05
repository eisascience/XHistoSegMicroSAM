"""
Example: Processing an image file
"""

import sys
import numpy as np
from PIL import Image
from xhalo.ml import segment_tissue
from xhalo.utils import load_image, mask_to_geojson, save_geojson, overlay_mask


def main():
    """Process an image file and generate segmentation"""
    
    # For demonstration, create a synthetic image
    print("Creating synthetic test image...")
    
    # Create a test image with some structure
    width, height = 1024, 1024
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some "tissue" regions (darker areas)
    for i in range(5):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        size = np.random.randint(100, 300)
        
        # Draw circular regions
        y_grid, x_grid = np.ogrid[:height, :width]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= size**2
        image[mask] = np.random.randint(100, 200, 3)
    
    print(f"Image size: {image.shape}")
    
    # Run segmentation
    print("\nRunning tissue segmentation...")
    mask = segment_tissue(
        image,
        tile_size=512,
        overlap=64
    )
    print(f"Segmentation complete. Mask shape: {mask.shape}")
    
    # Calculate statistics
    total_pixels = mask.size
    tissue_pixels = np.sum(mask > 0)
    coverage = (tissue_pixels / total_pixels) * 100
    
    print(f"\nSegmentation statistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Tissue pixels: {tissue_pixels:,}")
    print(f"  Coverage: {coverage:.2f}%")
    
    # Save mask
    mask_file = "output_mask.png"
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(mask_file)
    print(f"\n Saved mask to: {mask_file}")
    
    # Create overlay
    overlay_file = "output_overlay.png"
    overlay_img = overlay_mask(image, mask, alpha=0.4, color=(255, 0, 0))
    overlay_pil = Image.fromarray(overlay_img)
    overlay_pil.save(overlay_file)
    print(f" Saved overlay to: {overlay_file}")
    
    # Export GeoJSON
    geojson_file = "output_annotations.geojson"
    geojson_data = mask_to_geojson(
        mask,
        min_area=100,
        simplify_tolerance=2.0,
        properties={
            "type": "tissue",
            "source": "MedSAM",
            "coverage_percent": coverage
        }
    )
    save_geojson(geojson_data, geojson_file)
    print(f" Saved GeoJSON to: {geojson_file}")
    print(f"   Features: {len(geojson_data['features'])}")
    
    print("\n Processing complete!")


if __name__ == "__main__":
    main()
