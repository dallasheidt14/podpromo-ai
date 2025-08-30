#!/usr/bin/env python3
"""
Simple Whisper Test
Tests the Whisper model directly without the full service stack
"""

import whisper
import os
from pathlib import Path

def test_whisper_directly():
    """Test Whisper model directly"""
    print("🎵 Testing Whisper Model Directly")
    print("=" * 40)
    
    # Find an audio file
    uploads_dir = Path("uploads")
    audio_files = list(uploads_dir.glob("*.mp3"))
    
    if not audio_files:
        print("No audio files found in uploads directory")
        return
    
    # Use a smaller file for faster testing
    test_file = min(audio_files, key=lambda f: f.stat().st_size)
    print(f"Testing with: {test_file.name} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Model loaded successfully")
        
        # Test transcription
        print("Starting transcription...")
        result = model.transcribe(
            str(test_file),
            word_timestamps=False,  # Disable to avoid tensor issues
            fp16=False,
            verbose=False,
            beam_size=1,
            best_of=1,
            temperature=0.0
        )
        
        print("✅ Transcription successful!")
        print(f"Segments: {len(result['segments'])}")
        
        if result['segments']:
            first_seg = result['segments'][0]
            last_seg = result['segments'][-1]
            duration = last_seg['end'] - first_seg['start']
            print(f"Duration: {duration:.1f}s")
            
            # Show first segment
            print(f"\nFirst segment:")
            print(f"  Time: {first_seg['start']:.1f}s - {first_seg['end']:.1f}s")
            print(f"  Text: {first_seg['text'][:100]}...")
            
            # Show last segment
            print(f"\nLast segment:")
            print(f"  Time: {last_seg['start']:.1f}s - {last_seg['end']:.1f}s")
            print(f"  Text: {last_seg['text'][:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_whisper_directly()
