#!/usr/bin/env python3
"""
Test the enhanced pipeline integration to verify it's being invoked
"""

import asyncio
import logging
from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

# Set up logging to see the enhanced pipeline logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def test_enhanced_pipeline():
    """Test that the enhanced pipeline is being invoked"""
    
    episode_service = EpisodeService()
    clip_service = ClipScoreService(episode_service)
    
    episode_id = '3afc4f7f-523f-4a92-8ea3-f39316d47c90'
    print(f'Testing enhanced pipeline with episode: {episode_id}')
    
    try:
        candidates, meta = await clip_service.get_candidates(episode_id, platform='tiktok_reels')
        print(f'✅ Success! Generated {len(candidates)} candidates')
        if isinstance(meta, dict):
            print(f'Meta keys: {list(meta.keys())}')
            
            # Check for enhanced pipeline logs
            if 'seed_count' in meta:
                print(f'Seed clips: {meta.get("seed_count", 0)}')
                print(f'Brute clips: {meta.get("brute_count", 0)}')
            else:
                print('No seed/brute counts in meta - may be using legacy pipeline')
        else:
            print(f'Meta is not a dict: {type(meta)} - {meta}')
        
        # Show first few candidates
        for i, candidate in enumerate(candidates[:3]):
            print(f'Candidate {i}: {candidate.get("text", "")[:100]}...')
            print(f'  Score: {candidate.get("final_score", 0):.3f}')
            print(f'  Seed: {candidate.get("seed_idx") is not None}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_pipeline())
