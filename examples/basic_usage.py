"""
Example: Basic usage of XHalo Path Analyzer API
"""

import asyncio
import numpy as np
from xhalo.api import MockHaloAPIClient
from xhalo.ml import MedSAMPredictor
from xhalo.utils import mask_to_geojson, convert_to_halo_annotations


async def main():
    """Demonstrate basic API usage"""
    
    # Initialize mock Halo API client
    print("Initializing Halo API client...")
    client = MockHaloAPIClient()
    
    # List available slides
    print("\nFetching slides...")
    slides = await client.list_slides()
    print(f"Found {len(slides)} slides:")
    for slide in slides:
        print(f"  - {slide['name']} (ID: {slide['id']})")
    
    # Get detailed info for first slide
    if slides:
        slide_id = slides[0]['id']
        print(f"\nGetting details for slide {slide_id}...")
        slide_info = await client.get_slide_info(slide_id)
        print(f"  Dimensions: {slide_info['width']} x {slide_info['height']}")
        print(f"  Magnification: {slide_info.get('magnification', 'N/A')}x")
        
        # List ROIs
        print(f"\nFetching ROIs for slide {slide_id}...")
        rois = await client.list_rois(slide_id)
        print(f"Found {len(rois)} ROIs")
    
    # Create a sample image for segmentation
    print("\nCreating sample image...")
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize MedSAM predictor
    print("\nInitializing MedSAM predictor...")
    predictor = MedSAMPredictor(device="cpu")
    
    # Run segmentation
    print("Running segmentation...")
    mask = predictor.predict(sample_image)
    print(f"Segmentation complete. Mask shape: {mask.shape}")
    
    # Convert to GeoJSON
    print("\nConverting to GeoJSON...")
    geojson_data = mask_to_geojson(
        mask,
        min_area=50,
        properties={"source": "MedSAM", "type": "tissue"}
    )
    print(f"Generated GeoJSON with {len(geojson_data['features'])} features")
    
    # Convert to Halo annotations
    print("\nConverting to Halo annotation format...")
    annotations = convert_to_halo_annotations(
        mask,
        annotation_type="tissue",
        layer_name="AI Segmentation"
    )
    print(f"Created {len(annotations)} annotations")
    
    # Import annotations back to Halo
    if slides:
        print(f"\nImporting annotations to Halo (slide {slide_id})...")
        success = await client.import_annotations(
            slide_id,
            annotations,
            "AI Segmentation"
        )
        print(f"Import {'successful' if success else 'failed'}")
    
    print("\n Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
