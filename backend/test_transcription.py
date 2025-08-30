#!/usr/bin/env python3
"""
Test Transcription System
Tests the transcription system with existing audio files without uploading
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.episode_service import EpisodeService
from models import Episode

async def test_transcription_with_existing_file():
    """Test transcription with an existing audio file"""
    
    # Initialize the service
    service = EpisodeService()
    
    # Get list of existing audio files
    uploads_dir = Path("uploads")
    audio_files = list(uploads_dir.glob("*.mp3"))
    
    if not audio_files:
        print("No audio files found in uploads directory")
        return
    
    # Use the first available file
    test_file = audio_files[0]
    print(f"Testing with file: {test_file.name}")
    
    # Create a test episode ID
    test_episode_id = f"test_{test_file.stem}"
    
    try:
        # Create a mock episode object
        episode = Episode(
            id=test_episode_id,
            filename=test_file.name,
            original_name=test_file.name,
            size=test_file.stat().st_size,
            status="uploading"
        )
        
        # Store in service
        service.episodes[test_episode_id] = episode
        
        # Set the audio path directly
        episode.audio_path = str(test_file)
        episode.status = "processing"
        
        print(f"Created test episode: {test_episode_id}")
        print(f"File size: {episode.size / 1024 / 1024:.1f} MB")
        
        # Test transcription
        print("\nStarting transcription test...")
        transcript = await service._transcribe_audio(str(test_file), test_episode_id)
        
        print(f"\n✅ Transcription successful!")
        print(f"Segments: {len(transcript)}")
        print(f"Duration: {transcript[-1].end - transcript[0].start:.1f}s")
        
        # Show first few segments
        print("\nFirst 3 segments:")
        for i, seg in enumerate(transcript[:3]):
            print(f"  {i+1}. [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text[:100]}...")
        
        # Show last few segments
        print("\nLast 3 segments:")
        for i, seg in enumerate(transcript[-3:]):
            print(f"  {len(transcript)-2+i}. [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text[:100]}...")
        
        # Test the full pipeline
        print("\n\nTesting full episode processing...")
        await service.process_episode(test_episode_id)
        
        # Check final status
        final_episode = service.episodes[test_episode_id]
        print(f"\nFinal status: {final_episode.status}")
        if final_episode.transcript:
            print(f"Final transcript segments: {len(final_episode.transcript)}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_multiple_files():
    """Test transcription with multiple files to check consistency"""
    
    service = EpisodeService()
    uploads_dir = Path("uploads")
    audio_files = list(uploads_dir.glob("*.mp3"))[:3]  # Test first 3 files
    
    print(f"Testing transcription with {len(audio_files)} files...")
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n--- Test {i+1}: {audio_file.name} ---")
        
        test_episode_id = f"test_{i}_{audio_file.stem}"
        
        try:
            # Create episode
            episode = Episode(
                id=test_episode_id,
                filename=audio_file.name,
                original_name=audio_file.name,
                size=audio_file.stat().st_size,
                status="processing"
            )
            
            service.episodes[test_episode_id] = episode
            episode.audio_path = str(audio_file)
            
            # Quick transcription test
            transcript = await service._transcribe_audio(str(audio_file), test_episode_id)
            print(f"✅ Success: {len(transcript)} segments, {transcript[-1].end:.1f}s duration")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🎵 Testing Transcription System")
    print("=" * 50)
    
    # Test single file
    print("\n1. Testing single file transcription...")
    asyncio.run(test_transcription_with_existing_file())
    
    # Test multiple files
    print("\n\n2. Testing multiple files...")
    asyncio.run(test_multiple_files())
    
    print("\n✅ Test completed!")
