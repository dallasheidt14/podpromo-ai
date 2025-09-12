#!/usr/bin/env python3
"""
Debug script to check clip data and preview URLs for troubleshooting.
"""

import requests
import json

def debug_episode_clips(episode_id: str):
    """Debug clip data for a specific episode."""
    print(f"🔍 Debugging episode: {episode_id}")
    
    # Get clips from API
    try:
        response = requests.get(f'http://localhost:8000/api/episodes/{episode_id}/clips')
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
        
        data = response.json()
        clips = data.get('clips', [])
        
        print(f"📊 Found {len(clips)} clips")
        
        for i, clip in enumerate(clips):
            print(f"\n🎬 Clip {i+1}:")
            print(f"  ID: {clip.get('id')}")
            print(f"  Title: {clip.get('title', 'No title')}")
            print(f"  Score: {clip.get('score', 'No score')}")
            print(f"  preview_url: {clip.get('preview_url', 'NOT FOUND')}")
            print(f"  previewUrl: {clip.get('previewUrl', 'NOT FOUND')}")
            
            # Test if preview file is accessible
            preview_url = clip.get('preview_url') or clip.get('previewUrl')
            if preview_url:
                if not preview_url.startswith('http'):
                    preview_url = f'http://localhost:8000{preview_url}'
                
                try:
                    head_response = requests.head(preview_url, timeout=5)
                    print(f"  Preview Status: {head_response.status_code}")
                    print(f"  Content-Type: {head_response.headers.get('content-type', 'unknown')}")
                    print(f"  Content-Length: {head_response.headers.get('content-length', 'unknown')}")
                except Exception as e:
                    print(f"  Preview Error: {e}")
            else:
                print("  ❌ No preview URL found")
        
        print(f"\n✅ Debug complete for episode {episode_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Debug the YouTube episode
    episode_id = "cf93e5db-6d30-44e8-ab96-daa271ee8c9e"
    debug_episode_clips(episode_id)
    
    # Also debug any recent episodes
    print("\n" + "="*50)
    print("🔍 Recent episodes:")
    try:
        response = requests.get('http://localhost:8000/api/episodes')
        if response.status_code == 200:
            episodes = response.json()
            if isinstance(episodes, list) and episodes:
                recent = sorted(episodes, key=lambda x: x.get('uploadedAt', ''), reverse=True)[:3]
                for ep in recent:
                    print(f"  - {ep.get('id')}: {ep.get('title', 'No title')} ({ep.get('status', 'unknown')})")
    except Exception as e:
        print(f"Error getting episodes: {e}")
