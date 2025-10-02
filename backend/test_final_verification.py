#!/usr/bin/env python3
"""
Final verification test for Phase 2 & 3 implementation
"""

import asyncio
import logging
from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

# Set up logging to see the enhanced pipeline logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def test_final_verification():
    """Test that all Phase 2 & 3 features are working"""
    
    episode_service = EpisodeService()
    clip_service = ClipScoreService(episode_service)
    
    episode_id = '3afc4f7f-523f-4a92-8ea3-f39316d47c90'
    print(f'🎯 Testing Phase 2 & 3 implementation with episode: {episode_id}')
    
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
        
        # Check for debug artifacts
        import os
        debug_dir = os.path.join("backend", "uploads", "debug")
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith('_enhanced_debug.json')]
            if debug_files:
                print(f'✅ Debug artifacts found: {len(debug_files)} files')
            else:
                print('⚠️  No debug artifacts found')
        else:
            print('⚠️  Debug directory does not exist')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_final_verification())
