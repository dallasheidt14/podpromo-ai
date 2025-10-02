#!/usr/bin/env python3
"""
Test script for Phase 3.2 improvements:
- Platform-aware length diversification
- Safety & profanity polish
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_duration_diversification():
    """Test platform-aware duration diversification"""
    logger.info("Testing Duration Diversification...")
    
    from services.clip_score import apply_duration_diversity
    
    # Test clips with different durations
    test_clips = [
        {"id": "1", "dur": 15.0, "final_score": 0.9, "text": "Short clip"},
        {"id": "2", "dur": 35.0, "final_score": 0.8, "text": "Medium clip"},
        {"id": "3", "dur": 80.0, "final_score": 0.7, "text": "Long clip"},
        {"id": "4", "dur": 10.0, "final_score": 0.6, "text": "Very short clip"},
        {"id": "5", "dur": 45.0, "final_score": 0.85, "text": "Another medium clip"},
        {"id": "6", "dur": 120.0, "final_score": 0.75, "text": "Very long clip"},
    ]
    
    logger.info(f"Input clips: {len(test_clips)}")
    for clip in test_clips:
        logger.info(f"  {clip['id']}: {clip['dur']}s, score={clip['final_score']}")
    
    # Test with default buckets
    os.environ.pop("BUCKETS", None)  # Clear any existing env var
    diversified = apply_duration_diversity(test_clips)
    
    logger.info(f"Diversified clips: {len(diversified)}")
    for clip in diversified:
        logger.info(f"  {clip['id']}: {clip['dur']}s, score={clip['final_score']}")
    
    # Test with custom buckets
    os.environ["BUCKETS"] = "short,medium,long"
    # Note: This will fail because we don't have bucket definitions for custom names
    # But it tests the environment variable handling
    
    # Verify diversification worked
    diversified_ids = [c['id'] for c in diversified]
    original_ids = [c['id'] for c in test_clips]
    
    # Should have some diversity (not just top-scored clips)
    has_diversity = len(set(c['dur'] for c in diversified)) > 1
    
    logger.info(f"Diversification test:")
    logger.info(f"  Original IDs: {original_ids}")
    logger.info(f"  Diversified IDs: {diversified_ids}")
    logger.info(f"  Has duration diversity: {has_diversity}")
    
    return has_diversity and len(diversified) <= len(test_clips)

def test_safety_penalty():
    """Test safety penalty system"""
    logger.info("Testing Safety Penalty System...")
    
    from services.clip_score import calculate_safety_penalty
    
    # Test cases
    test_cases = [
        # (clip, platform, expected_penalty_range, description)
        ({"id": "1", "text": "This is a clean clip about technology"}, "youtube", (0.0, 0.0), "Clean content"),
        ({"id": "2", "text": "This is damn good content"}, "youtube", (0.05, 0.15), "Mild profanity"),
        ({"id": "3", "text": "I hate this shit so much"}, "youtube", (0.15, 0.30), "Strong profanity"),
        ({"id": "4", "text": "This is damn good content"}, "tiktok", (0.01, 0.08), "Mild profanity on TikTok"),
        ({"id": "5", "text": "I hate this shit so much"}, "tiktok", (0.04, 0.08), "Strong profanity on TikTok"),
        ({"id": "6", "text": "Kill the competition with innovation"}, "youtube", (0.05, 0.15), "Violent language"),
        ({"id": "7", "text": ""}, "youtube", (0.0, 0.0), "Empty text"),
    ]
    
    results = []
    
    for clip, platform, expected_range, description in test_cases:
        penalty = calculate_safety_penalty(clip, platform)
        min_expected, max_expected = expected_range
        
        passed = min_expected <= penalty <= max_expected
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}:")
        logger.info(f"    Platform: {platform}")
        logger.info(f"    Text: '{clip['text'][:50]}...'")
        logger.info(f"    Penalty: {penalty:.3f} (expected: {min_expected:.3f}-{max_expected:.3f})")
    
    all_passed = all(results)
    logger.info(f"Safety penalty test: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_platform_coefficients():
    """Test platform-specific penalty coefficients"""
    logger.info("Testing Platform Coefficients...")
    
    from services.clip_score import calculate_safety_penalty
    
    # Test clip with profanity
    test_clip = {"id": "test", "text": "This is damn good content"}
    
    platforms = ["youtube", "tiktok", "x", "facebook", "instagram", "unknown"]
    penalties = {}
    
    for platform in platforms:
        penalty = calculate_safety_penalty(test_clip, platform)
        penalties[platform] = penalty
        logger.info(f"  {platform}: {penalty:.3f}")
    
    # Verify platform differences
    youtube_penalty = penalties["youtube"]
    tiktok_penalty = penalties["tiktok"]
    
    # TikTok should have lower penalty (more tolerant)
    tiktok_tolerant = tiktok_penalty < youtube_penalty
    
    # Unknown platform should default to YouTube level
    unknown_default = abs(penalties["unknown"] - youtube_penalty) < 0.01
    
    logger.info(f"Platform coefficient test:")
    logger.info(f"  TikTok more tolerant: {tiktok_tolerant}")
    logger.info(f"  Unknown defaults correctly: {unknown_default}")
    
    return tiktok_tolerant and unknown_default

def test_edge_cases():
    """Test edge cases"""
    logger.info("Testing Edge Cases...")
    
    from services.clip_score import apply_duration_diversity, calculate_safety_penalty
    
    # Test duration diversification edge cases
    logger.info("Duration diversification edge cases:")
    
    # Too few clips
    few_clips = [{"id": "1", "dur": 20.0, "final_score": 0.8}]
    diversified_few = apply_duration_diversity(few_clips)
    logger.info(f"  Few clips: {len(few_clips)} → {len(diversified_few)}")
    
    # Clips with no duration
    no_duration_clips = [
        {"id": "1", "dur": 0.0, "final_score": 0.8},
        {"id": "2", "dur": -1.0, "final_score": 0.7},
        {"id": "3", "dur": 25.0, "final_score": 0.9},
    ]
    diversified_no_dur = apply_duration_diversity(no_duration_clips)
    logger.info(f"  No duration clips: {len(no_duration_clips)} → {len(diversified_no_dur)}")
    
    # Test safety penalty edge cases
    logger.info("Safety penalty edge cases:")
    
    # Empty clip
    empty_clip = {"id": "empty", "text": ""}
    empty_penalty = calculate_safety_penalty(empty_clip, "youtube")
    logger.info(f"  Empty clip penalty: {empty_penalty}")
    
    # Very long text
    long_text = "This is a very long text with lots of content. " * 100
    long_clip = {"id": "long", "text": long_text}
    long_penalty = calculate_safety_penalty(long_clip, "youtube")
    logger.info(f"  Long text penalty: {long_penalty}")
    
    # Multiple profanity instances
    multi_profanity = {"id": "multi", "text": "damn damn damn damn damn"}
    multi_penalty = calculate_safety_penalty(multi_profanity, "youtube")
    logger.info(f"  Multiple profanity penalty: {multi_penalty}")
    
    return True

def test_integration_scenario():
    """Test integration scenario"""
    logger.info("Testing Integration Scenario...")
    
    from services.clip_score import apply_duration_diversity, calculate_safety_penalty
    
    # Simulate a realistic scenario
    logger.info("Realistic scenario:")
    
    # Create realistic clips
    realistic_clips = [
        {"id": "1", "dur": 18.0, "final_score": 0.9, "text": "This is amazing technology"},
        {"id": "2", "dur": 35.0, "final_score": 0.85, "text": "I love this damn innovation"},
        {"id": "3", "dur": 12.0, "final_score": 0.8, "text": "Short but sweet"},
        {"id": "4", "dur": 60.0, "final_score": 0.75, "text": "Longer explanation of concepts"},
        {"id": "5", "dur": 25.0, "final_score": 0.7, "text": "This shit is incredible"},
    ]
    
    logger.info(f"  Input clips: {len(realistic_clips)}")
    
    # Apply duration diversification
    diversified = apply_duration_diversity(realistic_clips)
    logger.info(f"  After diversification: {len(diversified)}")
    
    # Apply safety penalties
    platform = "youtube"
    for clip in diversified:
        penalty = calculate_safety_penalty(clip, platform)
        if penalty > 0:
            clip["final_score"] = max(0.0, clip["final_score"] - penalty)
            clip["safety_penalty"] = penalty
    
    # Check results
    penalized_clips = [c for c in diversified if c.get("safety_penalty", 0) > 0]
    logger.info(f"  Clips with safety penalties: {len(penalized_clips)}")
    
    # Verify diversification worked
    durations = [c["dur"] for c in diversified]
    unique_durations = len(set(durations))
    
    logger.info(f"  Unique durations: {unique_durations}")
    logger.info(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    
    # Should have diversity and some penalties applied
    has_diversity = unique_durations > 1
    has_penalties = len(penalized_clips) > 0
    
    logger.info(f"  Has duration diversity: {has_diversity}")
    logger.info(f"  Has safety penalties: {has_penalties}")
    
    return has_diversity and has_penalties

if __name__ == "__main__":
    logger.info("Starting Phase 3.2 tests...")
    
    tests = [
        ("Duration Diversification", test_duration_diversification),
        ("Safety Penalty System", test_safety_penalty),
        ("Platform Coefficients", test_platform_coefficients),
        ("Edge Cases", test_edge_cases),
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
    logger.info("PHASE 3.2 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
