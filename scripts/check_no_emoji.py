#!/usr/bin/env python3
"""
Check for emoji characters in tracked files.
Exits with non-zero status if any emoji is found.
"""
import os
import sys
import re
import subprocess
from pathlib import Path


def is_emoji(char):
    """Check if a character is an emoji."""
    code_point = ord(char)
    # Common emoji ranges
    emoji_ranges = [
        (0x1F300, 0x1F9FF),  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
        (0x2600, 0x26FF),    # Miscellaneous Symbols
        (0x2700, 0x27BF),    # Dingbats
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x2B50, 0x2B50),    # Star
        (0x2705, 0x2705),    # Check mark
        (0x274C, 0x274C),    # Cross mark
        (0x2139, 0x2139),    # Information
        (0x2194, 0x2199),    # Arrows
        (0x231A, 0x231B),    # Watch, Hourglass
        (0x23E9, 0x23F3),    # Media controls
        (0x25AA, 0x25AB),    # Squares
        (0x25B6, 0x25B6),    # Play button
        (0x25C0, 0x25C0),    # Reverse button
        (0x25FB, 0x25FE),    # Squares
    ]
    
    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True
    return False


def check_file(file_path):
    """Check a file for emoji characters."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        emojis_found = []
        for line_num, line in enumerate(content.split('\n'), 1):
            for col_num, char in enumerate(line, 1):
                if is_emoji(char):
                    emojis_found.append({
                        'line': line_num,
                        'col': col_num,
                        'char': char,
                        'code': hex(ord(char)),
                        'context': line.strip()[:80]
                    })
        
        return emojis_found
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return []


def get_tracked_files():
    """Get list of git-tracked files."""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository", file=sys.stderr)
        sys.exit(1)


def main():
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    tracked_files = get_tracked_files()
    
    # Filter to only check text files (by extension)
    text_extensions = {'.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini'}
    files_to_check = [
        f for f in tracked_files
        if Path(f).suffix in text_extensions or Path(f).name in ['PKG-INFO', 'LICENSE', 'README']
    ]
    
    total_emojis = 0
    files_with_emojis = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        emojis = check_file(file_path)
        if emojis:
            files_with_emojis.append(file_path)
            total_emojis += len(emojis)
            print(f"\n{file_path}:")
            for emoji in emojis:
                print(f"  Line {emoji['line']}, Col {emoji['col']}: "
                      f"{emoji['char']} ({emoji['code']})")
                print(f"    Context: {emoji['context']}")
    
    print(f"\n{'='*60}")
    print(f"Total files checked: {len(files_to_check)}")
    print(f"Files with emojis: {len(files_with_emojis)}")
    print(f"Total emojis found: {total_emojis}")
    print(f"{'='*60}")
    
    if total_emojis > 0:
        print("\nERROR: Emoji characters found in repository!", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nSUCCESS: No emoji characters found!")
        sys.exit(0)


if __name__ == '__main__':
    main()
