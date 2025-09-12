#!/usr/bin/env python3
"""
Script to update the in-memory episode cache with the latest clips data.
This forces the episode service to reload the updated clips.json file.
"""

import json
import asyncio
from pathlib import Path
from services.episode_service import EpisodeService

async def update_episode_cache(episode_id: str):
    """Update the in-memory cache for a specific episode."""
    # Load the updated clips data
    clips_file = Path("uploads") / episode_id / "clips.json"
    
    if not clips_file.exists():
        print(f"Clips file not found: {clips_file}")
        return False
    
    with open(clips_file, 'r', encoding='utf-8') as f:
        updated_clips = json.load(f)
    
    print(f"Loaded {len(updated_clips)} clips from file")
    
    # Get the episode service instance
    episode_service = EpisodeService()
    
    # Get the current episode
    episode = await episode_service.get_episode(episode_id)
    if not episode:
        print(f"Episode {episode_id} not found in cache")
        return False
    
    print(f"Found episode {episode_id} in cache")
    
    # Update the clips in the episode
    episode.clips = updated_clips
    
    # Save the updated episode back to the service
    episode_service.episodes[episode_id] = episode
    episode_service._save_episode(episode)
    
    print(f"Updated episode {episode_id} with {len(updated_clips)} clips")
    
    # Verify the update
    updated_episode = await episode_service.get_episode(episode_id)
    if updated_episode and updated_episode.clips:
        has_preview = sum(1 for clip in updated_episode.clips if 'preview_url' in clip and clip['preview_url'])
        print(f"Verification: {has_preview}/{len(updated_episode.clips)} clips have preview URLs")
        return True
    else:
        print("Verification failed: episode not found or no clips")
        return False

if __name__ == "__main__":
    episode_id = "cf93e5db-6d30-44e8-ab96-daa271ee8c9e"
    print(f"Updating episode cache for {episode_id}...")
    
    success = asyncio.run(update_episode_cache(episode_id))
    if success:
        print("✅ Episode cache updated successfully!")
    else:
        print("❌ Failed to update episode cache")
