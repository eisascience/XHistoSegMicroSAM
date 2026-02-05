#!/usr/bin/env python3
"""
Patch script for segment-anything to ensure CUDA checkpoints load on CPU machines.

This script modifies the segment_anything/build_sam.py file to add map_location="cpu"
parameter to torch.load() calls, ensuring that models saved on CUDA devices can be
loaded on CPU-only machines.

Usage:
    python patch_segment_anything.py
"""

import sys
import os
import re
from pathlib import Path


def find_segment_anything_path():
    """Find the installed segment_anything package location."""
    try:
        import segment_anything
        return Path(segment_anything.__file__).parent
    except ImportError:
        print("Error: segment_anything package not found. Please install it first.")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588")
        return None


def patch_build_sam(segment_anything_path):
    """Patch the build_sam.py file to add map_location parameter."""
    build_sam_path = segment_anything_path / "build_sam.py"
    
    if not build_sam_path.exists():
        print(f"Error: {build_sam_path} not found.")
        return False
    
    # Read the current content
    with open(build_sam_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'map_location="cpu"' in content or 'map_location = "cpu"' in content:
        print(" build_sam.py is already patched.")
        return True
    
    # Pattern to match torch.load with weights_only parameter
    # Looking for: state_dict = torch.load(f, weights_only=False)
    # This pattern is flexible with whitespace and captures leading indentation
    pattern = r'^(\s*)(state_dict\s*=\s*torch\.load\s*\(\s*f\s*,\s*)(weights_only\s*=\s*False\s*)\)'
    
    # Replacement with map_location added, preserving indentation
    def replacement(match):
        indent = match.group(1)
        return f'{indent}state_dict = torch.load(\n{indent}    f,\n{indent}    map_location="cpu",\n{indent}    weights_only=False\n{indent})'
    
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if new_content == content:
        print("Warning: Pattern not found. The file may have a different structure.")
        print("Please manually edit the file to add map_location='cpu' to torch.load() calls.")
        return False
    
    # Backup the original file
    backup_path = build_sam_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f" Created backup at {backup_path}")
    
    # Write the patched content
    with open(build_sam_path, 'w') as f:
        f.write(new_content)
    
    print(f" Successfully patched {build_sam_path}")
    print("\nThe change ensures that SAM checkpoints saved from CUDA devices")
    print("can be loaded on CPU-only machines.")
    return True


def main():
    """Main function to run the patch."""
    print("Segment Anything Patch Utility")
    print("=" * 50)
    
    # Find segment_anything installation
    segment_anything_path = find_segment_anything_path()
    if not segment_anything_path:
        return 1
    
    print(f"Found segment_anything at: {segment_anything_path}")
    
    # Apply patch
    success = patch_build_sam(segment_anything_path)
    
    if success:
        print("\n Patch completed successfully!")
        return 0
    else:
        print("\n Patch failed. Manual intervention may be required.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
