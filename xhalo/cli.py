"""
Command-line interface for XHalo Path Analyzer
"""

import argparse
import sys
import logging


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="XHalo Path Analyzer - Digital Pathology AI Workflow"
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Web UI command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run the web server on'
    )
    web_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind to'
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process image from command line')
    process_parser.add_argument(
        'input',
        type=str,
        help='Input image path'
    )
    process_parser.add_argument(
        '--output',
        type=str,
        help='Output path for segmentation mask'
    )
    process_parser.add_argument(
        '--geojson',
        type=str,
        help='Output path for GeoJSON export'
    )
    process_parser.add_argument(
        '--tile-size',
        type=int,
        default=1024,
        help='Tile size for processing'
    )
    
    args = parser.parse_args()
    
    if args.command == 'web':
        launch_web_ui(args.port, args.host)
    elif args.command == 'process':
        process_image(args)
    else:
        parser.print_help()
        sys.exit(1)


def launch_web_ui(port: int, host: str):
    """Launch Streamlit web interface"""
    import subprocess
    import os
    
    app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    app_path = os.path.abspath(app_path)
    
    print(f"Launching Halo AI Workflow on {host}:{port}")
    print(f"App path: {app_path}")
    
    subprocess.run([
        'streamlit', 'run', app_path,
        '--server.port', str(port),
        '--server.address', host
    ])


def process_image(args):
    """Process image from command line"""
    from xhalo.utils import load_image, save_geojson, mask_to_geojson
    from xhalo.ml import segment_tissue
    from PIL import Image
    import numpy as np
    
    print(f"Processing image: {args.input}")
    
    # Load image
    image = load_image(args.input)
    print(f"Image size: {image.shape}")
    
    # Run segmentation
    print("Running segmentation...")
    mask = segment_tissue(image, tile_size=args.tile_size)
    
    # Save mask if requested
    if args.output:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(args.output)
        print(f"Saved mask to: {args.output}")
    
    # Export GeoJSON if requested
    if args.geojson:
        geojson_data = mask_to_geojson(mask)
        save_geojson(geojson_data, args.geojson)
        print(f"Saved GeoJSON to: {args.geojson}")
    
    print("Processing complete!")


if __name__ == '__main__':
    main()
