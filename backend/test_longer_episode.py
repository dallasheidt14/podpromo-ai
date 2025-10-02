#!/usr/bin/env python3
"""
Test with a longer episode to see if our fixes work better with more content
"""

import asyncio
import logging
from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

# Set up logging to see the enhanced pipeline logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def test_longer_episode():
    """Test with a longer episode"""
    
    episode_service = EpisodeService()
    clip_service = ClipScoreService(episode_service)
    
    # Try an episode that has words.json file
    episode_id = 'b2d45df2-efa4-4c35-8ac3-6595049b40ef'  # This one has words.json
    print(f'🎯 Testing with longer episode: {episode_id}')
    
    try:
        candidates, meta = await clip_service.get_candidates(episode_id, platform='tiktok_reels')
        print(f'✅ Success! Generated {len(candidates)} candidates')
        
        if isinstance(meta, dict):
            print(f'📊 Meta keys: {list(meta.keys())}')
            
            # Check for enhanced pipeline logs
            if 'seed_count' in meta:
                seed_count = meta.get('seed_count', 0)
                brute_count = meta.get('brute_count', 0)
                print(f'🎯 Seed clips: {seed_count}')
                print(f'🎯 Brute clips: {brute_count}')
                
                if seed_count > 0:
                    print('✅ EOS-based seed generation is working!')
                else:
                    print('⚠️  No seed clips generated - may need debugging')
                    
            else:
                print('❌ No seed/brute counts in meta - may be using legacy pipeline')
        else:
            print(f'❌ Meta is not a dict: {type(meta)} - {meta}')
        
        # Show duration distribution
        if candidates:
            durations = [c.get('duration', 0) for c in candidates]
            print(f'📏 Duration distribution: {durations}')
            print(f'📏 Duration range: {min(durations):.1f}s - {max(durations):.1f}s')
            
            # Show virality scores
            virality_scores = [c.get('virality_pct', 0) for c in candidates]
            print(f'🎯 Virality scores: {virality_scores}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_longer_episode())
