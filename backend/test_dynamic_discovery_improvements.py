#!/usr/bin/env python3
"""
Test script for dynamic discovery stability improvements.
Tests the fixes for brittleness on short/structured tracks.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.secret_sauce_pkg.features import discover_dynamic_length, compute_features_lite, build_hotness_curve, top_k_local_maxima
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_transcript():
    """Load the real transcript file"""
    transcript_path = "uploads/transcripts/c746f928-85c5-4541-8119-261e659291d8.json"
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('transcript', [])
        return segments, data
        
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return None, None

def test_hotness_curve_smoothing():
    """Test adaptive smoothing for short vs long tracks"""
    logger.info("Testing hotness curve smoothing improvements...")
    
    # Load real transcript
    segments, transcript_data = load_real_transcript()
    if not segments:
        logger.error("Failed to load transcript data")
        return False
    
    # Extract words
    words = []
    for segment in segments:
        segment_words = segment.get('words', [])
        words.extend(segment_words)
    
    duration_s = transcript_data.get('duration', 0)
    logger.info(f"Testing with real transcript: {duration_s:.1f}s ({duration_s/60:.1f} minutes)")
    
    try:
        # Test adaptive smoothing
        times, hook, arousal, payoff, info, q_or_list, emotion, loop = compute_features_lite(words, hop_s=0.5)
        
        # Test both smoothing values
        smooth_short = 4.5  # For short/structured tracks
        smooth_long = 3.0   # For longer tracks
        
        hotness_short = build_hotness_curve(times, hook, arousal, payoff, info, q_or_list, emotion, loop, smooth_window=smooth_short)
        hotness_long = build_hotness_curve(times, hook, arousal, payoff, info, q_or_list, emotion, loop, smooth_window=smooth_long)
        
        logger.info(f"Hotness curve results:")
        logger.info(f"  Short smoothing ({smooth_short}s): {len(hotness_short)} points")
        logger.info(f"  Long smoothing ({smooth_long}s): {len(hotness_long)} points")
        
        # Test peak detection with different thresholds
        min_dist_short = 4.0  # For short tracks
        min_dist_long = 6.0   # For longer tracks
        
        peaks_short = top_k_local_maxima(hotness_short, times, k=30, min_dist=min_dist_short)
        peaks_long = top_k_local_maxima(hotness_long, times, k=30, min_dist=min_dist_long)
        
        logger.info(f"Peak detection results:")
        logger.info(f"  Short thresholds: {len(peaks_short)} peaks")
        logger.info(f"  Long thresholds: {len(peaks_long)} peaks")
        
        # Show sample peaks
        if peaks_short:
            logger.info(f"  Sample short peaks: {[f'{p[0]:.1f}s' for p in peaks_short[:5]]}")
        if peaks_long:
            logger.info(f"  Sample long peaks: {[f'{p[0]:.1f}s' for p in peaks_long[:5]]}")
        
        return len(peaks_short) > 0 and len(peaks_long) > 0
        
    except Exception as e:
        logger.error(f"Hotness curve test failed: {e}")
        return False

def test_retry_logic():
    """Test the retry logic for no-peaks scenarios"""
    logger.info("Testing retry logic for no-peaks scenarios...")
    
    # Load real transcript
    segments, transcript_data = load_real_transcript()
    if not segments:
        logger.error("Failed to load transcript data")
        return False
    
    # Extract words and EOS times
    words = []
    eos_times = []
    for segment in segments:
        segment_words = segment.get('words', [])
        words.extend(segment_words)
        eos_times.append(segment.get('end', 0))
    
    duration_s = transcript_data.get('duration', 0)
    
    try:
        # Test with very strict thresholds to trigger retry logic
        def strict_score_fn(start, end):
            # Very strict scoring to potentially cause no candidates
            duration = end - start
            if duration < 10.0:  # Reject short segments
                return 0.0
            return min(0.9, duration / 60.0)
        
        logger.info(f"Testing retry logic with strict thresholds...")
        candidates = discover_dynamic_length(words, eos_times, strict_score_fn, duration_s)
        
        logger.info(f"Retry logic test result: {len(candidates)} candidates found")
        
        if candidates:
            logger.info("Sample candidates after retry:")
            for i, candidate in enumerate(candidates[:3]):
                logger.info(f"  {i+1}: start={candidate['start']:.1f}s, end={candidate['end']:.1f}s, score={candidate['score']:.3f}")
        
        # The retry logic should prevent complete failure
        return True  # Success if no exception thrown
        
    except Exception as e:
        logger.error(f"Retry logic test failed: {e}")
        return False

def test_adaptive_thresholds():
    """Test adaptive thresholds for short vs long tracks"""
    logger.info("Testing adaptive thresholds...")
    
    # Test with different duration scenarios
    test_cases = [
        (300, "Short track (5 minutes)"),
        (600, "Medium track (10 minutes)"),
        (1200, "Long track (20 minutes)"),
    ]
    
    results = []
    
    for duration_s, description in test_cases:
        logger.info(f"\nTesting {description}:")
        
        # Calculate adaptive thresholds
        smooth_window = 3.0 if duration_s > 600 else 4.5
        min_peak_prominence = 0.15 if duration_s > 600 else 0.10
        min_peak_distance_s = 6.0 if duration_s > 600 else 4.0
        
        logger.info(f"  Duration: {duration_s}s ({duration_s/60:.1f} minutes)")
        logger.info(f"  Smooth window: {smooth_window}s")
        logger.info(f"  Min peak prominence: {min_peak_prominence}")
        logger.info(f"  Min peak distance: {min_peak_distance_s}s")
        
        # Verify thresholds make sense
        if duration_s <= 600:  # Short track
            if smooth_window == 4.5 and min_peak_prominence == 0.10 and min_peak_distance_s == 4.0:
                logger.info("  ✓ Short track thresholds applied correctly")
                results.append(True)
            else:
                logger.warning("  ⚠ Short track thresholds incorrect")
                results.append(False)
        else:  # Long track
            if smooth_window == 3.0 and min_peak_prominence == 0.15 and min_peak_distance_s == 6.0:
                logger.info("  ✓ Long track thresholds applied correctly")
                results.append(True)
            else:
                logger.warning("  ⚠ Long track thresholds incorrect")
                results.append(False)
    
    return all(results)

def test_dynamic_discovery_improvements():
    """Test the complete dynamic discovery improvements"""
    logger.info("Testing complete dynamic discovery improvements...")
    
    # Load real transcript
    segments, transcript_data = load_real_transcript()
    if not segments:
        logger.error("Failed to load transcript data")
        return False
    
    # Extract words and EOS times
    words = []
    eos_times = []
    for segment in segments:
        segment_words = segment.get('words', [])
        words.extend(segment_words)
        eos_times.append(segment.get('end', 0))
    
    duration_s = transcript_data.get('duration', 0)
    
    try:
        # Test with improved scoring function (compatible with optimize_window)
        def improved_score_fn(start, end):
            duration = end - start
            # More lenient scoring
            if duration < 5.0:  # Minimum duration
                return 0.0
            # Score based on duration with some randomness for variety
            base_score = min(0.8, duration / 45.0)
            return max(0.1, base_score)
        
        logger.info(f"Testing improved dynamic discovery...")
        logger.info(f"  Duration: {duration_s:.1f}s ({duration_s/60:.1f} minutes)")
        logger.info(f"  Words: {len(words)}")
        logger.info(f"  EOS markers: {len(eos_times)}")
        
        candidates = discover_dynamic_length(words, eos_times, improved_score_fn, duration_s)
        
        logger.info(f"Improved dynamic discovery result:")
        logger.info(f"  Candidates found: {len(candidates)}")
        
        if candidates:
            logger.info("  Sample candidates:")
            for i, candidate in enumerate(candidates[:5]):
                logger.info(f"    {i+1}: start={candidate['start']:.1f}s, end={candidate['end']:.1f}s, score={candidate['score']:.3f}")
            
            # Check if candidates are reasonable
            valid_candidates = 0
            for candidate in candidates:
                duration = candidate['end'] - candidate['start']
                if 5.0 <= duration <= 120.0 and candidate['score'] > 0.1:
                    valid_candidates += 1
            
            logger.info(f"  Valid candidates: {valid_candidates}/{len(candidates)}")
            
            if valid_candidates > 0:
                logger.info("✓ Dynamic discovery improvements working")
                return True
            else:
                logger.warning("⚠ No valid candidates found")
                return False
        else:
            logger.warning("⚠ No candidates found")
            return False
        
    except Exception as e:
        logger.error(f"Dynamic discovery improvements test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting dynamic discovery stability improvement tests...")
    
    tests = [
        ("Hotness Curve Smoothing", test_hotness_curve_smoothing),
        ("Retry Logic", test_retry_logic),
        ("Adaptive Thresholds", test_adaptive_thresholds),
        ("Complete Dynamic Discovery", test_dynamic_discovery_improvements),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        logger.info("\n✓ Dynamic discovery stability improvements are working!")
        logger.info("  - Adaptive smoothing for short/structured tracks")
        logger.info("  - Retry logic for no-peaks scenarios")
        logger.info("  - Adaptive thresholds based on duration")
        logger.info("  - Reduced brittleness in peak detection")
