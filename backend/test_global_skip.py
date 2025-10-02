#!/usr/bin/env python3
"""
Test script to verify global skip_quality_recheck works correctly
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

async def test_global_skip():
    """Test that enhanced pipeline results are preserved through entire flow"""
    
    # Initialize services
    episode_service = EpisodeService(None)  # No database needed for this test
    clip_service = ClipScoreService(episode_service)
    
    # Test with an episode that has both transcript and words
    episode_id = "6310f42f-7c6b-440e-9209-15e93299b01b"
    
    print(f"Testing global skip for episode: {episode_id}")
    
    try:
        # Get candidates using enhanced pipeline
        candidates, metadata = await clip_service.get_candidates(
            episode_id=episode_id,
            platform="tiktok_reels",
            genre="general"
        )
        
        print(f"\nFound {len(candidates)} candidates")
        print(f"Enhanced pipeline used: {metadata.get('use_enhanced', False)}")
        
        # Check if we got more candidates than before (should be 6 instead of 2)
        if len(candidates) >= 4:
            print("✅ SUCCESS: Got more candidates than before (global skip working)")
        else:
            print(f"⚠️  WARNING: Only got {len(candidates)} candidates (expected 4+)")
        
        # Check each candidate for quality
        for i, clip in enumerate(candidates[:4]):  # Check first 4 clips
            print(f"\n--- Clip {i+1} ---")
            print(f"Start: {clip.get('start', 0.0):.2f}s")
            print(f"End: {clip.get('end', 0.0):.2f}s") 
            print(f"Duration: {clip.get('duration', 0.0):.2f}s")
            print(f"Score: {clip.get('final_score', 0.0):.2f}")
            
            # Check if transcript is reasonable length for the duration
            transcript_text = clip.get('transcript', {}).get('text', '')
            expected_words = int(clip.get('duration', 0.0) * 2.5)  # ~2.5 words per second
            actual_words = len(transcript_text.split())
            
            print(f"Expected words: ~{expected_words}, Actual words: {actual_words}")
            
            if actual_words > expected_words * 1.5:
                print("⚠️  WARNING: Transcript seems too long for video duration")
            elif actual_words < expected_words * 0.5:
                print("⚠️  WARNING: Transcript seems too short for video duration")
            else:
                print("✅ Transcript length looks reasonable")
        
        return True
        
    except Exception as e:
        print(f"Error testing global skip: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_global_skip())
    if success:
        print("\n✅ Global skip test completed")
    else:
        print("\n❌ Global skip test failed")
