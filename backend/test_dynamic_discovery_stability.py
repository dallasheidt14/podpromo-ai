#!/usr/bin/env python3
"""
Test script for dynamic discovery stability improvements.
Tests adaptive smoothing and retry logic for short/structured tracks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.secret_sauce_pkg.features import discover_dynamic_length, compute_features_lite, build_hotness_curve
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def create_test_words_and_eos(duration_s, segments=5):
    """Create test words and EOS times for a short structured track"""
    words = []
    eos_times = []
    
    # Create segments with different characteristics
    segment_duration = duration_s / segments
    current_time = 0.0
    
    for i in range(segments):
        # Add some words for this segment
        segment_words = [
            f"segment_{i}_word_{j}" for j in range(10)
        ]
        words.extend(segment_words)
        
        # Add EOS times
        for j in range(5):
            eos_time = current_time + (j * segment_duration / 5)
            eos_times.append(eos_time)
        
        current_time += segment_duration
    
    return words, eos_times

def test_score_fn(segment_text, start_time, end_time):
    """Mock scoring function"""
    # Return a score based on segment position and content
    duration = end_time - start_time
    if "segment_2" in segment_text:  # Make segment 2 high-scoring
        return 0.8
    elif "segment_0" in segment_text:  # Make segment 0 low-scoring
        return 0.2
    else:
        return 0.5

def test_short_track_stability():
    """Test dynamic discovery on a short structured track"""
    log.info("Testing dynamic discovery stability on short track...")
    
    # Create a short track (5 minutes)
    duration_s = 300
    words, eos_times = create_test_words_and_eos(duration_s, segments=8)
    
    log.info(f"Created test track: {duration_s}s, {len(words)} words, {len(eos_times)} EOS points")
    
    # Test dynamic discovery
    try:
        candidates = discover_dynamic_length(words, eos_times, test_score_fn, duration_s)
        
        log.info(f"Dynamic discovery result: {len(candidates)} candidates found")
        
        if candidates:
            log.info("Sample candidates:")
            for i, candidate in enumerate(candidates[:3]):
                log.info(f"  {i+1}: start={candidate['start']:.1f}s, end={candidate['end']:.1f}s, score={candidate['score']:.3f}")
        
        return len(candidates) > 0
        
    except Exception as e:
        log.error(f"Dynamic discovery failed: {e}")
        return False

def test_long_track_stability():
    """Test dynamic discovery on a longer track"""
    log.info("Testing dynamic discovery stability on longer track...")
    
    # Create a longer track (15 minutes)
    duration_s = 900
    words, eos_times = create_test_words_and_eos(duration_s, segments=20)
    
    log.info(f"Created test track: {duration_s}s, {len(words)} words, {len(eos_times)} EOS points")
    
    # Test dynamic discovery
    try:
        candidates = discover_dynamic_length(words, eos_times, test_score_fn, duration_s)
        
        log.info(f"Dynamic discovery result: {len(candidates)} candidates found")
        
        if candidates:
            log.info("Sample candidates:")
            for i, candidate in enumerate(candidates[:3]):
                log.info(f"  {i+1}: start={candidate['start']:.1f}s, end={candidate['end']:.1f}s, score={candidate['score']:.3f}")
        
        return len(candidates) > 0
        
    except Exception as e:
        log.error(f"Dynamic discovery failed: {e}")
        return False

def test_very_short_track():
    """Test dynamic discovery on a very short track that might fail"""
    log.info("Testing dynamic discovery on very short track...")
    
    # Create a very short track (2 minutes)
    duration_s = 120
    words, eos_times = create_test_words_and_eos(duration_s, segments=3)
    
    log.info(f"Created test track: {duration_s}s, {len(words)} words, {len(eos_times)} EOS points")
    
    # Test dynamic discovery
    try:
        candidates = discover_dynamic_length(words, eos_times, test_score_fn, duration_s)
        
        log.info(f"Dynamic discovery result: {len(candidates)} candidates found")
        
        # Even if no candidates, the function should not crash
        return True
        
    except Exception as e:
        log.error(f"Dynamic discovery failed: {e}")
        return False

if __name__ == "__main__":
    log.info("Starting dynamic discovery stability tests...")
    
    tests = [
        ("Short Track (5min)", test_short_track_stability),
        ("Long Track (15min)", test_long_track_stability),
        ("Very Short Track (2min)", test_very_short_track),
    ]
    
    results = []
    for test_name, test_func in tests:
        log.info(f"\n{'='*50}")
        log.info(f"Running: {test_name}")
        log.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            log.info(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            log.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    log.info(f"\n{'='*50}")
    log.info("TEST SUMMARY")
    log.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        log.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    log.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
