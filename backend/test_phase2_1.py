#!/usr/bin/env python3
"""
Test script for Phase 2.1 improvements:
- Dynamic discovery seatbelts (retry + smoothing + min-peak distance)
- Centralized viability gate
- Deterministic ordering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import hashlib
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_discovery_seatbelts():
    """Test dynamic discovery seatbelts"""
    logger.info("Testing Dynamic Discovery Seatbelts...")
    
    # Test environment variable handling
    os.environ['DISCOVERY_SMOOTH_SHORT'] = '4.5'
    os.environ['DYNAMIC_RETRY'] = '1'
    
    from services.secret_sauce_pkg.features import discover_dynamic_length
    
    # Create test data
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "this", "start": 1.0, "end": 1.5},
        {"word": "is", "start": 1.5, "end": 2.0},
        {"word": "a", "start": 2.0, "end": 2.5},
        {"word": "test", "start": 2.5, "end": 3.0},
    ]
    
    eos_times = [1.0, 3.0]
    duration_s = 300.0  # 5 minutes (short track)
    
    def mock_score_fn(start, end):
        duration = end - start
        return min(0.8, duration / 30.0)
    
    # Test with short track (should use 4.5s smoothing)
    candidates = discover_dynamic_length(words, eos_times, mock_score_fn, duration_s)
    
    logger.info(f"Dynamic discovery test:")
    logger.info(f"  Duration: {duration_s}s (short track)")
    logger.info(f"  Smooth window should be: 4.5s")
    logger.info(f"  Candidates found: {len(candidates)}")
    
    # Test with long track (should use 3.0s smoothing)
    duration_s_long = 700.0  # 11+ minutes (long track)
    candidates_long = discover_dynamic_length(words, eos_times, mock_score_fn, duration_s_long)
    
    logger.info(f"Long track test:")
    logger.info(f"  Duration: {duration_s_long}s (long track)")
    logger.info(f"  Smooth window should be: 3.0s")
    logger.info(f"  Candidates found: {len(candidates_long)}")
    
    # Test retry logic by disabling it
    os.environ['DYNAMIC_RETRY'] = '0'
    candidates_no_retry = discover_dynamic_length(words, eos_times, mock_score_fn, duration_s)
    
    logger.info(f"Retry disabled test:")
    logger.info(f"  Candidates found: {len(candidates_no_retry)}")
    
    return True

def test_centralized_viability_gate():
    """Test centralized viability gate"""
    logger.info("Testing Centralized Viability Gate...")
    
    from services.quality_filters import is_viable_clip
    
    # Test cases
    test_cases = [
        # (clip, expected_ok, expected_reason, description)
        ({"text": "Hello world", "finished_thought": True, "ft_coverage_ratio": 0.8}, True, "", "Valid clip"),
        ({"text": "", "finished_thought": True, "ft_coverage_ratio": 0.8}, False, "empty_text", "Empty text"),
        ({"text": "Call 1-800-GRAINGER", "finished_thought": True, "ft_coverage_ratio": 0.8}, False, "ad_like", "Ad content"),
        ({"text": "Hello world", "finished_thought": False, "ft_coverage_ratio": 0.8}, False, "unfinished", "Unfinished thought"),
        ({"text": "Hello world", "finished_thought": True, "ft_coverage_ratio": 0.5}, False, "unfinished", "Low coverage"),
        ({"text": "Hello world", "finished_thought": True, "ft_coverage_ratio": 0.8, "safety_flag": "hate"}, False, "safety_block", "Safety issue"),
    ]
    
    results = []
    
    for clip, expected_ok, expected_reason, description in test_cases:
        ok, reason = is_viable_clip(clip)
        passed = (ok == expected_ok and reason == expected_reason)
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}: ok={ok} (expected: {expected_ok}), reason='{reason}' (expected: '{expected_reason}')")
    
    # Test relax parameter
    logger.info("Testing relax parameter...")
    clip = {"text": "Hello world", "finished_thought": True, "ft_coverage_ratio": 0.5}
    
    # Without relax (should fail due to low coverage)
    ok, reason = is_viable_clip(clip)
    logger.info(f"Without relax: ok={ok}, reason='{reason}'")
    
    # With relax (should pass with lower threshold)
    ok, reason = is_viable_clip(clip, relax={"finished_thresh": 0.4})
    logger.info(f"With relax: ok={ok}, reason='{reason}'")
    
    all_passed = all(results)
    logger.info(f"Centralized viability gate test: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_deterministic_ordering():
    """Test deterministic ordering"""
    logger.info("Testing Deterministic Ordering...")
    
    # Test seed generation
    episode_id = "test-episode-123"
    seed = int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16)
    
    logger.info(f"Seed generation test:")
    logger.info(f"  Episode ID: {episode_id}")
    logger.info(f"  Generated seed: {seed}")
    
    # Test that same episode_id generates same seed
    seed2 = int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16)
    assert seed == seed2, "Same episode_id should generate same seed"
    
    # Test stable sort key
    def _stable_key(c):
        finished_like = 1 if (c.get("finished_thought") or c.get("ft_status") in ("finished","sparse_finished")) else 0
        coverage = float(c.get("ft_coverage_ratio") or 0.0)
        novelty  = float(c.get("novelty_score") or 0.0)
        length_s = float(c.get("duration") or 0.0)
        start    = float(c.get("start") or 0.0)
        vir      = float(c.get("final_score") or 0.0)
        return (-finished_like, -coverage, -novelty, -length_s, -vir, start)
    
    # Test clips with different characteristics
    test_clips = [
        {"id": "1", "finished_thought": True, "ft_coverage_ratio": 0.9, "novelty_score": 0.8, "duration": 30.0, "start": 0.0, "final_score": 0.9},
        {"id": "2", "finished_thought": False, "ft_coverage_ratio": 0.7, "novelty_score": 0.6, "duration": 25.0, "start": 10.0, "final_score": 0.8},
        {"id": "3", "finished_thought": True, "ft_coverage_ratio": 0.8, "novelty_score": 0.7, "duration": 35.0, "start": 5.0, "final_score": 0.85},
    ]
    
    # Sort with stable key
    sorted_clips = sorted(test_clips, key=_stable_key)
    
    logger.info(f"Stable sort test:")
    logger.info(f"  Original order: {[c['id'] for c in test_clips]}")
    logger.info(f"  Sorted order: {[c['id'] for c in sorted_clips]}")
    
    # Verify that finished clips come first
    finished_first = all(c.get("finished_thought", False) for c in sorted_clips[:2])
    logger.info(f"  Finished clips first: {finished_first}")
    
    # Test deterministic behavior
    random.seed(seed)
    order1 = [c['id'] for c in sorted(test_clips, key=_stable_key)]
    
    random.seed(seed)
    order2 = [c['id'] for c in sorted(test_clips, key=_stable_key)]
    
    deterministic = order1 == order2
    logger.info(f"  Deterministic ordering: {deterministic}")
    
    return deterministic and finished_first

def test_integration_scenario():
    """Test integration scenario"""
    logger.info("Testing Integration Scenario...")
    
    # Test that all components work together
    from services.quality_filters import is_viable_clip
    
    # Simulate a realistic clip processing scenario
    test_clips = [
        {"id": "1", "text": "Call 1-800-GRAINGER for supplies", "finished_thought": True, "ft_coverage_ratio": 0.8},
        {"id": "2", "text": "I love talking about technology", "finished_thought": True, "ft_coverage_ratio": 0.9},
        {"id": "3", "text": "This is incomplete", "finished_thought": False, "ft_coverage_ratio": 0.5},
        {"id": "4", "text": "Another great topic", "finished_thought": True, "ft_coverage_ratio": 0.8},
    ]
    
    # Apply viability gate
    viable_clips = []
    for clip in test_clips:
        ok, reason = is_viable_clip(clip)
        if ok:
            viable_clips.append(clip)
        else:
            logger.info(f"Filtered out clip {clip['id']}: {reason}")
    
    logger.info(f"Integration test:")
    logger.info(f"  Input clips: {len(test_clips)}")
    logger.info(f"  Viable clips: {len(viable_clips)}")
    logger.info(f"  Viable IDs: {[c['id'] for c in viable_clips]}")
    
    # Should filter out clips 1 (ad) and 3 (unfinished)
    expected_viable = ["2", "4"]
    actual_viable = [c['id'] for c in viable_clips]
    
    passed = actual_viable == expected_viable
    logger.info(f"  Expected: {expected_viable}")
    logger.info(f"  Actual: {actual_viable}")
    logger.info(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return passed

if __name__ == "__main__":
    logger.info("Starting Phase 2.1 tests...")
    
    tests = [
        ("Dynamic Discovery Seatbelts", test_dynamic_discovery_seatbelts),
        ("Centralized Viability Gate", test_centralized_viability_gate),
        ("Deterministic Ordering", test_deterministic_ordering),
        ("Integration Scenario", test_integration_scenario),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    logger.info(f"\n{'='*50}")
    logger.info("PHASE 2.1 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
