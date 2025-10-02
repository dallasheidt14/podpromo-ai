#!/usr/bin/env python3
"""
Test suite for critical safety and correctness fixes.
"""

import os
import sys
import tempfile
import shutil
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.utils.io_safe import sanitize_seg, atomic_write_json
from services.utils.clip_validation import clamp_span, validate_clip_times
from services.utils.deterministic import seed_everything, get_deterministic_seed
from services.utils.clip_ids import generate_clip_id, assign_clip_ids
from services.utils.text_normalization import normalize_text, is_eos_char

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_path_sanitization():
    """Test path sanitization prevents directory traversal."""
    logger.info("Testing path sanitization...")
    
    # Test cases
    test_cases = [
        ("normal_episode", "normal_episode"),
        ("episode with spaces", "episode_with_spaces"),
        ("../../../etc/passwd", ".._.._.._etc_passwd"),
        ("episode/with/slashes", "episode_with_slashes"),
        ("", "untitled"),
        ("episode@#$%^&*()", "episode_"),
    ]
    
    for input_val, expected in test_cases:
        result = sanitize_seg(input_val)
        assert result == expected, f"Expected '{expected}', got '{result}'"
        logger.info(f"✓ sanitize_seg('{input_val}') = '{result}'")
    
    logger.info("✓ Path sanitization tests passed")

def test_atomic_writes():
    """Test atomic JSON writes."""
    logger.info("Testing atomic JSON writes...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test.json")
        test_data = {"test": "data", "number": 42}
        
        # Write atomically
        atomic_write_json(test_path, test_data)
        
        # Verify file exists and content is correct
        assert os.path.exists(test_path)
        
        import json
        with open(test_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
        logger.info("✓ Atomic JSON write test passed")

def test_clip_time_clamping():
    """Test clip time validation and clamping."""
    logger.info("Testing clip time clamping...")
    
    # Test cases: (start, end, episode_dur) -> (expected_start, expected_end)
    test_cases = [
        (10.0, 20.0, 100.0, 10.0, 20.0),  # Normal case
        (-5.0, 20.0, 100.0, 0.0, 20.0),   # Negative start
        (10.0, 150.0, 100.0, 10.0, 100.0), # End beyond episode
        (10.0, 10.5, 100.0, 10.0, 10.5),  # Minimum duration
        (10.0, 10.2, 100.0, 10.0, 10.5),  # Too short, gets clamped
        (95.0, 10.0, 100.0, 95.0, 95.5), # End before start, gets minimum duration
    ]
    
    for start, end, episode_dur, expected_start, expected_end in test_cases:
        result_start, result_end = clamp_span(start, end, episode_dur)
        assert abs(result_start - expected_start) < 0.001, f"Start mismatch: {result_start} vs {expected_start}"
        assert abs(result_end - expected_end) < 0.001, f"End mismatch: {result_end} vs {expected_end}"
        logger.info(f"✓ clamp_span({start}, {end}, {episode_dur}) = ({result_start}, {result_end})")
    
    logger.info("✓ Clip time clamping tests passed")

def test_deterministic_seeding():
    """Test deterministic RNG seeding."""
    logger.info("Testing deterministic seeding...")
    
    episode_id = "test_episode_123"
    
    # Test seed generation
    seed1 = get_deterministic_seed(episode_id)
    seed2 = get_deterministic_seed(episode_id)
    assert seed1 == seed2, "Seeds should be deterministic"
    
    # Test different episodes get different seeds
    seed3 = get_deterministic_seed("different_episode")
    assert seed1 != seed3, "Different episodes should get different seeds"
    
    logger.info(f"✓ Deterministic seeding: episode '{episode_id}' -> seed {seed1}")
    logger.info("✓ Deterministic seeding tests passed")

def test_clip_id_generation():
    """Test canonical clip ID generation."""
    logger.info("Testing clip ID generation...")
    
    episode_id = "test_episode"
    
    # Test deterministic IDs
    id1 = generate_clip_id(episode_id, 10.0, 20.0)
    id2 = generate_clip_id(episode_id, 10.0, 20.0)
    assert id1 == id2, "Same inputs should produce same ID"
    
    # Test different clips get different IDs
    id3 = generate_clip_id(episode_id, 10.0, 21.0)
    assert id1 != id3, "Different clips should get different IDs"
    
    # Test ID format
    assert len(id1) == 16, "ID should be 16 characters"
    assert all(c in "0123456789abcdef" for c in id1), "ID should be hexadecimal"
    
    logger.info(f"✓ Clip ID generation: {id1}")
    logger.info("✓ Clip ID generation tests passed")

def test_text_normalization():
    """Test Unicode text normalization."""
    logger.info("Testing text normalization...")
    
    # Test Unicode normalization
    test_cases = [
        ("Hello world.", "Hello world."),  # Normal text
        ("Hello… world", "Hello... world"),  # Ellipsis
        ("Hello – world", "Hello - world"),  # En dash
        ("Hello — world", "Hello - world"),  # Em dash
    ]
    
    for input_text, expected in test_cases:
        result = normalize_text(input_text)
        assert result == expected, f"Expected '{expected}', got '{result}'"
        logger.info(f"✓ normalize_text('{input_text}') = '{result}'")
    
    # Test EOS detection
    assert is_eos_char("."), "Period should be EOS"
    assert is_eos_char("?"), "Question mark should be EOS"
    assert is_eos_char("!"), "Exclamation should be EOS"
    assert is_eos_char("。"), "CJK period should be EOS"
    assert not is_eos_char("a"), "Letter should not be EOS"
    
    logger.info("✓ Text normalization tests passed")

def test_integration():
    """Test integration of all safety fixes."""
    logger.info("Testing integration of safety fixes...")
    
    # Create test clips
    test_clips = [
        {"start": 10.0, "end": 20.0, "text": "Hello world."},
        {"start": -5.0, "end": 25.0, "text": "This is a test."},
        {"start": 30.0, "end": 30.2, "text": "Too short"},
    ]
    
    episode_id = "integration_test"
    episode_duration = 100.0
    
    # Apply all safety fixes
    from services.utils.clip_validation import validate_all_clips
    from services.utils.clip_ids import assign_clip_ids
    from services.utils.text_normalization import normalize_all_segments
    
    # Validate times
    validated_clips = validate_all_clips(test_clips, episode_duration)
    
    # Assign IDs
    clips_with_ids = assign_clip_ids(validated_clips, episode_id)
    
    # Normalize text
    final_clips = normalize_all_segments(clips_with_ids)
    
    # Verify results
    assert len(final_clips) == 3, "Should have 3 clips"
    
    for clip in final_clips:
        assert "id" in clip, "Clip should have ID"
        assert len(clip["id"]) == 16, "ID should be 16 characters"
        assert clip["start"] >= 0.0, "Start should be non-negative"
        assert clip["end"] <= episode_duration, "End should not exceed episode duration"
        assert clip["duration"] >= 0.5, "Duration should be at least 0.5s"
    
    logger.info("✓ Integration test passed")

def main():
    """Run all safety tests."""
    logger.info("🔥 Starting critical safety and correctness tests...")
    
    try:
        test_path_sanitization()
        test_atomic_writes()
        test_clip_time_clamping()
        test_deterministic_seeding()
        test_clip_id_generation()
        test_text_normalization()
        test_integration()
        
        logger.info("🎉 ALL SAFETY TESTS PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Safety test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
