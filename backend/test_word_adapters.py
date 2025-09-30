#!/usr/bin/env python3
"""
Test the word adapter fixes to ensure they prevent KeyError crashes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.clip_score import _gap_after, _gap_before, _nearest_boundary_backward, _nearest_boundary_forward

def test_word_adapters():
    """Test word adapter functions with different word schemas"""
    
    print("Testing word adapter fixes...")
    
    # Test with different word schemas that could cause KeyError
    words_whisper = [
        {"start": 0.0, "end": 0.5, "text": "Hello", "word": "Hello"},
        {"start": 0.5, "end": 1.0, "text": "world", "word": "world"},
        {"start": 1.0, "end": 1.5, "text": "test", "word": "test"},
    ]
    
    words_whisperx = [
        {"start": 0.0, "end": 0.5, "word": "Hello"},
        {"start": 0.5, "end": 1.0, "word": "world"},
        {"start": 1.0, "end": 1.5, "word": "test"},
    ]
    
    words_legacy = [
        {"t": 0.0, "d": 0.5, "w": "Hello"},
        {"t": 0.5, "d": 0.5, "w": "world"},
        {"t": 1.0, "d": 0.5, "w": "test"},
    ]
    
    # Test _gap_after
    print("Testing _gap_after...")
    gap1 = _gap_after(words_whisper, 0)
    gap2 = _gap_after(words_whisperx, 0)
    gap3 = _gap_after(words_legacy, 0)
    print(f"  Whisper: {gap1:.3f}s")
    print(f"  WhisperX: {gap2:.3f}s")
    print(f"  Legacy: {gap3:.3f}s")
    assert gap1 == gap2 == gap3 == 0.0, "All should be 0.0s"
    
    # Test _gap_before
    print("Testing _gap_before...")
    gap1 = _gap_before(words_whisper, 1)
    gap2 = _gap_before(words_whisperx, 1)
    gap3 = _gap_before(words_legacy, 1)
    print(f"  Whisper: {gap1:.3f}s")
    print(f"  WhisperX: {gap2:.3f}s")
    print(f"  Legacy: {gap3:.3f}s")
    assert gap1 == gap2 == gap3 == 0.0, "All should be 0.0s"
    
    # Test _nearest_boundary_backward
    print("Testing _nearest_boundary_backward...")
    boundary1 = _nearest_boundary_backward(words_whisper, 1.2)
    boundary2 = _nearest_boundary_backward(words_whisperx, 1.2)
    boundary3 = _nearest_boundary_backward(words_legacy, 1.2)
    print(f"  Whisper: {boundary1:.3f}s")
    print(f"  WhisperX: {boundary2:.3f}s")
    print(f"  Legacy: {boundary3:.3f}s")
    
    # Test _nearest_boundary_forward
    print("Testing _nearest_boundary_forward...")
    boundary1 = _nearest_boundary_forward(words_whisper, 0.3)
    boundary2 = _nearest_boundary_forward(words_whisperx, 0.3)
    boundary3 = _nearest_boundary_forward(words_legacy, 0.3)
    print(f"  Whisper: {boundary1:.3f}s")
    print(f"  WhisperX: {boundary2:.3f}s")
    print(f"  Legacy: {boundary3:.3f}s")
    
    # Test edge cases
    print("Testing edge cases...")
    
    # Empty words
    assert _gap_after([], 0) == 9999.0, "Empty words should return 9999.0"
    assert _gap_before([], 0) == 9999.0, "Empty words should return 9999.0"
    
    # Out of bounds
    assert _gap_after(words_whisper, -1) == 9999.0, "Out of bounds should return 9999.0"
    assert _gap_after(words_whisper, 10) == 9999.0, "Out of bounds should return 9999.0"
    assert _gap_before(words_whisper, 0) == 9999.0, "Out of bounds should return 9999.0"
    assert _gap_before(words_whisper, 10) == 9999.0, "Out of bounds should return 9999.0"
    
    print("✅ All word adapter tests passed!")
    return True

if __name__ == "__main__":
    success = test_word_adapters()
    exit(0 if success else 1)
