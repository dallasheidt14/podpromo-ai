#!/usr/bin/env python3
"""
Test Full Pipeline
Tests the complete pipeline: transcription → segmentation → scoring → candidates
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.episode_service import EpisodeService
from services.clip_score import ClipScoreService
from models import Episode

async def test_full_pipeline():
    """Test the complete pipeline with an existing audio file"""
    
    print("🎵 Testing Full Pipeline: Transcription → Scoring → Candidates")
    print("=" * 70)
    
    # Initialize services
    episode_service = EpisodeService()
    clip_score_service = ClipScoreService(episode_service)
    
    # Find an audio file (use a smaller one for faster testing)
    uploads_dir = Path("uploads")
    audio_files = list(uploads_dir.glob("*.mp3"))
    
    if not audio_files:
        print("No audio files found in uploads directory")
        return
    
    # Use a smaller file for faster testing
    test_file = min(audio_files, key=lambda f: f.stat().st_size)
    print(f"Testing with: {test_file.name} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Create a test episode ID
    test_episode_id = f"test_{test_file.stem}"
    
    try:
        # Step 1: Create episode and transcribe
        print(f"\n📝 Step 1: Creating episode and transcribing...")
        
        episode = Episode(
            id=test_episode_id,
            filename=test_file.name,
            original_name=test_file.name,
            size=test_file.stat().st_size,
            status="processing"
        )
        
        episode_service.episodes[test_episode_id] = episode
        episode.audio_path = str(test_file)
        
        # Transcribe
        transcript = await episode_service._transcribe_audio(str(test_file), test_episode_id)
        episode.transcript = transcript
        episode.status = "completed"
        
        print(f"✅ Transcription completed: {len(transcript)} segments")
        print(f"   Duration: {transcript[-1].end - transcript[0].start:.1f}s")
        
        # Step 2: Generate candidates with different genres
        print(f"\n🎯 Step 2: Generating candidates with different genres...")
        
        test_genres = ['general', 'comedy', 'fantasy_sports', 'business', 'education']
        
        for genre in test_genres:
            print(f"\n--- Testing Genre: {genre.upper()} ---")
            
            try:
                candidates = await clip_score_service.get_candidates(
                    episode_id=test_episode_id,
                    platform="tiktok_reels",
                    genre=genre
                )
                
                if candidates:
                    print(f"✅ Found {len(candidates)} candidates")
                    
                    # Show top 3 candidates with scores
                    for i, candidate in enumerate(candidates[:3]):
                        print(f"  {i+1}. Score: {candidate.get('score', 0):.3f}")
                        print(f"     Display Score: {candidate.get('display_score', 'N/A')}")
                        print(f"     Confidence: {candidate.get('confidence', 'N/A')}")
                        print(f"     Duration: {candidate['end'] - candidate['start']:.1f}s")
                        print(f"     Text: {candidate['text'][:80]}...")
                        
                        # Show features if available
                        if 'features' in candidate:
                            features = candidate['features']
                            print(f"     Features: hook={features.get('hook_score', 0):.3f}, "
                                  f"payoff={features.get('payoff_score', 0):.3f}, "
                                  f"arousal={features.get('arousal_score', 0):.3f}")
                        
                        print()
                else:
                    print(f"❌ No candidates generated for {genre}")
                    
            except Exception as e:
                print(f"❌ Error with {genre}: {e}")
        
        # Step 3: Test specific platform optimizations
        print(f"\n🚀 Step 3: Testing platform-specific optimizations...")
        
        platforms = ['tiktok_reels', 'shorts', 'linkedin_sq']
        
        for platform in platforms:
            print(f"\n--- Platform: {platform.upper()} ---")
            
            try:
                candidates = await clip_score_service.get_candidates(
                    episode_id=test_episode_id,
                    platform=platform,
                    genre='general'
                )
                
                if candidates:
                    print(f"✅ Found {len(candidates)} candidates")
                    
                    # Show platform-specific metrics
                    top_candidate = candidates[0]
                    if 'features' in top_candidate:
                        features = top_candidate['features']
                        platform_match = features.get('platform_len_match', 0)
                        print(f"   Platform Length Match: {platform_match:.3f}")
                        print(f"   Duration: {top_candidate['end'] - top_candidate['start']:.1f}s")
                        print(f"   Score: {top_candidate.get('score', 0):.3f}")
                else:
                    print(f"❌ No candidates for {platform}")
                    
            except Exception as e:
                print(f"❌ Error with {platform}: {e}")
        
        # Step 4: Show detailed scoring breakdown
        print(f"\n📊 Step 4: Detailed scoring breakdown for top candidate...")
        
        try:
            candidates = await clip_score_service.get_candidates(
                episode_id=test_episode_id,
                platform="tiktok_reels",
                genre="general"
            )
            
            if candidates:
                top_candidate = candidates[0]
                print(f"Top candidate details:")
                print(f"  Time: {top_candidate['start']:.1f}s - {top_candidate['end']:.1f}s")
                print(f"  Duration: {top_candidate['end'] - top_candidate['start']:.1f}s")
                print(f"  Raw Score: {top_candidate.get('score', 0):.3f}")
                print(f"  Display Score: {top_candidate.get('display_score', 'N/A')}")
                print(f"  Confidence: {top_candidate.get('confidence', 'N/A')}")
                print(f"  Synergy Multiplier: {top_candidate.get('synergy_mult', 'N/A')}")
                print(f"  Winning Path: {top_candidate.get('winning_path', 'N/A')}")
                
                if 'features' in top_candidate:
                    features = top_candidate['features']
                    print(f"\n  Feature Scores:")
                    print(f"    Hook Score: {features.get('hook_score', 0):.3f}")
                    print(f"    Payoff Score: {features.get('payoff_score', 0):.3f}")
                    print(f"    Arousal Score: {features.get('arousal_score', 0):.3f}")
                    print(f"    Emotion Score: {features.get('emotion_score', 0):.3f}")
                    print(f"    Question Score: {features.get('question_score', 0):.3f}")
                    print(f"    Info Density: {features.get('info_density', 0):.3f}")
                    print(f"    Loopability: {features.get('loopability', 0):.3f}")
                    print(f"    Platform Length Match: {features.get('platform_len_match', 0):.3f}")
                
                print(f"\n  Text Preview:")
                print(f"    {top_candidate['text'][:200]}...")
                
        except Exception as e:
            print(f"❌ Error getting detailed breakdown: {e}")
        
        print(f"\n✅ Full pipeline test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
