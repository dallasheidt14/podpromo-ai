#!/usr/bin/env python3
"""
Script to fix missing preview URLs for existing YouTube clips.
This adds preview URLs to clips that were processed before the fix.
"""

import json
import os
from pathlib import Path
from services.preview_service import ensure_preview

def fix_episode_previews(episode_id: str):
    """Fix preview URLs for a specific episode."""
    # Load the episode data
    storage_dir = Path("uploads") / episode_id
    episode_file = storage_dir / "clips.json"
    
    if not episode_file.exists():
        print(f"Episode file not found: {episode_file}")
        return False
    
    with open(episode_file, 'r', encoding='utf-8') as f:
        clips = json.load(f)
    
    if not clips or not isinstance(clips, list):
        print(f"No clips found for episode {episode_id}")
        return False
    print(f"Found {len(clips)} clips for episode {episode_id}")
    
    # Try to find the video file in uploads
    video_path = Path("uploads") / f"{episode_id}.mp4"
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"Using video path: {video_path}")
    
    # Fix each clip
    updated = 0
    for clip in clips:
        if 'preview_url' not in clip or not clip['preview_url']:
            print(f"Fixing clip {clip['id']}...")
            
            # Generate preview URL
            preview_url = ensure_preview(
                source_media=video_path,
                episode_id=episode_id,
                clip_id=clip['id'],
                start_sec=float(clip.get('start', 0)),
                end_sec=float(clip.get('end', 0)),
                max_preview_sec=20.0
            )
            
            if preview_url:
                clip['preview_url'] = preview_url
                updated += 1
                print(f"  Added preview URL: {preview_url}")
            else:
                print(f"  Failed to generate preview for clip {clip['id']}")
    
    if updated > 0:
        # Save the updated clips data
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(clips, f, indent=2, ensure_ascii=False)
        print(f"Updated {updated} clips with preview URLs")
        return True
    else:
        print("No clips needed updating")
        return False

if __name__ == "__main__":
    episode_id = "cf93e5db-6d30-44e8-ab96-daa271ee8c9e"
    print(f"Fixing preview URLs for episode {episode_id}...")
    success = fix_episode_previews(episode_id)
    if success:
        print("✅ Preview URLs fixed successfully!")
    else:
        print("❌ Failed to fix preview URLs")
