#!/usr/bin/env python3
"""
Test enhanced pipeline logs
"""

import asyncio
import logging
from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

# Set up logging to see the enhanced pipeline logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def test_logs():
    """Test that we can see the enhanced pipeline logs"""
    
    episode_service = EpisodeService()
    clip_service = ClipScoreService(episode_service)
    
    episode_id = '3afc4f7f-523f-4a92-8ea3-f39316d47c90'
    print('🎯 Testing enhanced pipeline logs...')
    
    try:
        candidates, meta = await clip_service.get_candidates(episode_id, platform='tiktok_reels')
        print(f'✅ Generated {len(candidates)} candidates')
        
        if isinstance(meta, dict) and 'seed_count' in meta:
            seed_count = meta.get('seed_count', 0)
            brute_count = meta.get('brute_count', 0)
            print(f'🎯 Seed clips: {seed_count}')
            print(f'🎯 Brute clips: {brute_count}')
        
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    asyncio.run(test_logs())
