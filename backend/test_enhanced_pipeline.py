#!/usr/bin/env python3
"""
Test the enhanced pipeline fixes directly
"""
import asyncio
import json
import os
import glob
from pathlib import Path

async def test_enhanced_pipeline():
    """Test the enhanced pipeline with null-safe processing"""
    
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
    print(f"Testing with episode: {episode_id}")
    
    # Load words data
    with open(latest, "r", encoding="utf-8") as f:
        words_data = json.load(f)
    
    print(f"Loaded {len(words_data)} words")
    
    # Test the enhanced pipeline components
    from services.clip_score import (
        _ensure_clip_fields, _blend_final_score, _w_start, _w_end, _w_text,
        _enhanced_select_and_rank
    )
    
    # Test word adapters
    print("Testing word adapters...")
    test_word = {"start": 1.0, "end": 2.0, "text": "hello"}
    try:
        start = _w_start(test_word)
        end = _w_end(test_word)
        text = _w_text(test_word)
        print(f"  Word adapters work: start={start}, end={end}, text='{text}'")
    except Exception as e:
        print(f"  ❌ Word adapters failed: {e}")
        return False
    
    # Test defensive helpers
    print("Testing defensive helpers...")
    test_clip = {"start": 1.0, "end": 2.0, "text": "test"}
    try:
        safe_clip = _ensure_clip_fields(test_clip)
        score = _blend_final_score(safe_clip)
        print(f"  Defensive helpers work: score={score}")
    except Exception as e:
        print(f"  ❌ Defensive helpers failed: {e}")
        return False
    
    # Test enhanced selection with mock candidates
    print("Testing enhanced selection...")
    mock_candidates = [
        {"start": 1.0, "end": 5.0, "text": "This is a test clip", "final_score": 0.8},
        {"start": 10.0, "end": 15.0, "text": "Another test clip", "final_score": 0.6},
    ]
    
    try:
        # This will test the enhanced selection logic
        result = _enhanced_select_and_rank(mock_candidates, words_data, {})
        print(f"  Enhanced selection returned {len(result)} clips")
        if result:
            print(f"  Top clip: {result[0].get('text', 'No text')[:50]}...")
    except Exception as e:
        print(f"  ❌ Enhanced selection failed: {e}")
        return False
    
    print("✅ All enhanced pipeline components working!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_pipeline())
    exit(0 if success else 1)
