#!/usr/bin/env python3
"""
Cleanup script for old preview files
Removes preview files older than N days to prevent disk bloat
"""

import os
import sys
from pathlib import Path
from time import time

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import OUTPUT_DIR

def cleanup_previews(days: int = 30):
    """Remove preview files older than specified days"""
    previews_dir = Path(OUTPUT_DIR) / "previews"
    
    if not previews_dir.exists():
        print(f"Previews directory {previews_dir} does not exist")
        return
    
    cutoff_time = time() - (days * 86400)  # days to seconds
    removed_count = 0
    total_size = 0
    
    print(f"Cleaning up previews older than {days} days...")
    print(f"Cutoff time: {time() - cutoff_time:.0f} seconds ago")
    
    for preview_file in previews_dir.glob("*.m4a"):
        try:
            file_stat = preview_file.stat()
            file_mtime = file_stat.st_mtime
            file_size = file_stat.st_size
            
            if file_mtime < cutoff_time:
                total_size += file_size
                preview_file.unlink()
                removed_count += 1
                print(f"Removed: {preview_file.name} ({file_size} bytes)")
        except Exception as e:
            print(f"Error removing {preview_file.name}: {e}")
    
    print(f"Cleanup complete: {removed_count} files removed, {total_size / 1024 / 1024:.2f} MB freed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old preview files")
    parser.add_argument("--days", type=int, default=30, help="Remove files older than N days (default: 30)")
    
    args = parser.parse_args()
    cleanup_previews(args.days)
