#!/usr/bin/env python3
"""
Test the enhanced pipeline integration to prevent CANDIDATE_PIPELINE_FAIL
"""
import asyncio
import json
import os
import glob
from pathlib import Path

async def test_enhanced_integration():
    """Test the enhanced pipeline integration with real episode data"""
    
    # Find latest transcript
    trans_dir = Path("uploads/transcripts")
    files = sorted(
        glob.glob(str(trans_dir / "*/words.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        print("❌ No transcript files found")
        return False
    
    latest = files[0]
    episode_id = Path(latest).parent.name
    print(f"Testing enhanced integration with episode: {episode_id}")
    
    # Load words data
    with open(latest, "r", encoding="utf-8") as f:
        words_data = json.load(f)
    
    print(f"Loaded {len(words_data)} words")
    
    # Debug: show first few words
    for i, word in enumerate(words_data[:3]):
        print(f"Word {i}: {word}")
    
    # Test the enhanced viral clips function directly
    from services.secret_sauce_pkg.features import find_viral_clips_enhanced
    
    # Create mock segments from words data
    segments = []
    current_segment = {
        "start": 0.0,
        "end": 0.0,
        "text": "",
        "words": []
    }
    
    for word in words_data[:100]:  # Use first 100 words for testing
        if not current_segment["text"]:
            current_segment["start"] = word.get("start", 0.0)
        
        current_segment["end"] = word.get("end", word.get("start", 0.0) + 1.0)
        current_segment["text"] += " " + word.get("word", word.get("text", ""))
        current_segment["words"].append(word)
        
        # Create segments every 20 words
        if len(current_segment["words"]) >= 20:
            current_segment["text"] = current_segment["text"].strip()
            if current_segment["text"]:  # Only add if text is not empty
                segments.append(current_segment)
            current_segment = {
                "start": word.get("end", word.get("start", 0.0) + 1.0),
                "end": word.get("end", word.get("start", 0.0) + 1.0),
                "text": "",
                "words": []
            }
    
    # Add the last segment if it has content
    if current_segment["text"]:
        current_segment["text"] = current_segment["text"].strip()
        if current_segment["text"]:  # Only add if text is not empty
            segments.append(current_segment)
    
    print(f"Created {len(segments)} test segments")
    
    # Debug: show segment details
    for i, seg in enumerate(segments[:3]):
        print(f"Segment {i}: text='{seg.get('text', '')[:50]}...', start={seg.get('start', 0):.2f}, end={seg.get('end', 0):.2f}")
    
    # Test the enhanced function
    try:
        result = find_viral_clips_enhanced(
            segments=segments,
            audio_file="test_audio.mp3",  # Mock audio file
            genre="general",
            platform="tiktok",
            fallback_mode=False,
            effective_eos_times=[],
            effective_word_end_times=[],
            eos_source="test",
            episode_words=words_data
        )
        
        print(f"✅ Enhanced function succeeded!")
        print(f"   Clips generated: {len(result.get('clips', []))}")
        print(f"   Genre: {result.get('genre', 'unknown')}")
        print(f"   Platform: {result.get('platform', 'unknown')}")
        
        if result.get('clips'):
            top_clip = result['clips'][0]
            print(f"   Top clip: {top_clip.get('text', 'No text')[:100]}...")
            print(f"   Top clip score: {top_clip.get('final_score', 0.0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_integration())
    exit(0 if success else 1)
