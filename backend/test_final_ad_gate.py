#!/usr/bin/env python3
"""
Test script for Final No-Ad Gate implementation.
Tests that ads are filtered out at the final stage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.title_service import _looks_like_ad
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_final_ad_gate():
    """Test the final ad gate with known ad content"""
    logger.info("Testing Final No-Ad Gate...")
    
    # Test cases with known ad content
    test_cases = [
        ("Call 1-800-GRAINGER for all your industrial supplies", True, "Phone number ad"),
        ("Visit our website at www.example.com", True, "Website URL ad"),
        ("Use code SAVE20 at checkout", True, "Promo code ad"),
        ("This episode is sponsored by Squarespace", True, "Sponsor mention"),
        ("I love talking about technology and innovation", False, "Normal content"),
        ("The weather today is really nice", False, "Normal content"),
        ("Order now and get free shipping", True, "CTA ad"),
        ("Let me tell you about my recent vacation", False, "Normal content"),
    ]
    
    results = []
    
    for text, expected_ad, description in test_cases:
        is_ad = _looks_like_ad(text)
        passed = is_ad == expected_ad
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}: '{text[:50]}...' -> {'AD' if is_ad else 'NOT AD'} (expected: {'AD' if expected_ad else 'NOT AD'})")
    
    all_passed = all(results)
    logger.info(f"\nFinal Ad Gate Test: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        logger.info("✅ Final No-Ad Gate is working correctly!")
        logger.info("  - Ads are properly detected and filtered")
        logger.info("  - Normal content passes through")
    else:
        logger.info("❌ Final No-Ad Gate needs adjustment")
    
    return all_passed

def test_ad_gate_integration():
    """Test the ad gate logic as it would be used in clip_score.py"""
    logger.info("Testing ad gate integration logic...")
    
    # Simulate clips with different content
    test_clips = [
        {"id": "1", "text": "Call 1-800-GRAINGER for supplies", "is_advertisement": False},
        {"id": "2", "text": "I love talking about technology", "is_advertisement": False},
        {"id": "3", "text": "Visit www.example.com today", "is_advertisement": True},
        {"id": "4", "text": "Use code SAVE20 at checkout", "is_advertisement": False},
        {"id": "5", "text": "The weather is nice today", "is_advertisement": False},
    ]
    
    # Apply the final ad gate logic
    pre_ad_gate_count = len(test_clips)
    filtered_clips = [c for c in test_clips if not (
        c.get("is_advertisement", False) or 
        _looks_like_ad(c.get("text", ""))
    )]
    ad_gate_filtered = pre_ad_gate_count - len(filtered_clips)
    
    logger.info(f"Ad Gate Integration Results:")
    logger.info(f"  Input clips: {pre_ad_gate_count}")
    logger.info(f"  Filtered clips: {ad_gate_filtered}")
    logger.info(f"  Output clips: {len(filtered_clips)}")
    
    # Check results
    expected_filtered = 3  # clips 1, 3, and 4 should be filtered
    expected_remaining = 2  # clips 2 and 5 should remain
    
    if ad_gate_filtered == expected_filtered and len(filtered_clips) == expected_remaining:
        logger.info("✅ Ad gate integration working correctly!")
        logger.info(f"  Remaining clips: {[c['id'] for c in filtered_clips]}")
        return True
    else:
        logger.error(f"❌ Expected {expected_filtered} filtered, {expected_remaining} remaining")
        logger.error(f"   Got {ad_gate_filtered} filtered, {len(filtered_clips)} remaining")
        return False

if __name__ == "__main__":
    logger.info("Starting Final No-Ad Gate tests...")
    
    tests = [
        ("Final Ad Gate Detection", test_final_ad_gate),
        ("Ad Gate Integration", test_ad_gate_integration),
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
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
