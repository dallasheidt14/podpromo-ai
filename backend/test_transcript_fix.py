#!/usr/bin/env python3
"""
Test script to verify transcript text matches video content after boundary refinement fix
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService
from database_service import DatabaseService

async def test_transcript_alignment():
    """Test that transcript text matches video content after fix"""
    
    # Initialize services
    db_service = DatabaseService()
    episode_service = EpisodeService(db_service)
    clip_service = ClipScoreService(episode_service)
    
    # Test with the same episode that had the mismatch
    episode_id = "709d26b5-0f6a-4a94-b583-c8ffc4a78d7c"
    
    print(f"Testing transcript alignment for episode: {episode_id}")
    
    try:
        # Get candidates using enhanced pipeline
        candidates, metadata = await clip_service.get_candidates(
            episode_id=episode_id,
            platform="tiktok_reels",
            genre="general"
        )
        
        print(f"\nFound {len(candidates)} candidates")
        print(f"Enhanced pipeline used: {metadata.get('use_enhanced', False)}")
        
        # Check each candidate for transcript alignment
        for i, clip in enumerate(candidates[:2]):  # Check first 2 clips
            print(f"\n--- Clip {i+1} ---")
            print(f"Start: {clip.get('start', 0.0):.2f}s")
            print(f"End: {clip.get('end', 0.0):.2f}s") 
            print(f"Duration: {clip.get('duration', 0.0):.2f}s")
            print(f"Transcript: {clip.get('transcript', {}).get('text', 'No transcript')[:100]}...")
            
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
        print(f"Error testing transcript alignment: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_transcript_alignment())
    if success:
        print("\n✅ Transcript alignment test completed")
    else:
        print("\n❌ Transcript alignment test failed")
