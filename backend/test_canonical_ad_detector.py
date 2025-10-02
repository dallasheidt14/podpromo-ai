#!/usr/bin/env python3
"""
Test script for Canonical Ad Detector implementation.
Tests that all modules use the same centralized ad detection logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_canonical_ad_detector():
    """Test the canonical ad detector functions"""
    logger.info("Testing Canonical Ad Detector...")
    
    from services.ads import looks_like_ad, ad_like_score, ad_penalty
    
    # Test cases
    test_cases = [
        ("Call 1-800-GRAINGER for supplies", True, "Phone number ad"),
        ("Visit www.example.com today", True, "Website URL ad"),
        ("Use code SAVE20 at checkout", True, "Promo code ad"),
        ("This episode is sponsored by Squarespace", True, "Sponsor mention"),
        ("I love talking about technology", False, "Normal content"),
        ("The weather is nice today", False, "Normal content"),
    ]
    
    results = []
    
    for text, expected_ad, description in test_cases:
        # Test looks_like_ad
        is_ad = looks_like_ad(text)
        passed_looks = is_ad == expected_ad
        
        # Test ad_like_score
        score = ad_like_score(text)
        expected_score_range = (0.6, 1.0) if expected_ad else (0.0, 0.3)
        passed_score = expected_score_range[0] <= score <= expected_score_range[1]
        
        # Test ad_penalty
        penalty_result = ad_penalty(text)
        passed_penalty = penalty_result["is_advertisement"] == expected_ad
        
        passed = passed_looks and passed_score and passed_penalty
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}:")
        logger.info(f"    looks_like_ad: {is_ad} (expected: {expected_ad})")
        logger.info(f"    ad_like_score: {score:.2f} (expected: {expected_score_range})")
        logger.info(f"    ad_penalty: {penalty_result['is_advertisement']} (expected: {expected_ad})")
    
    all_passed = all(results)
    logger.info(f"\nCanonical ad detector test: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_module_consistency():
    """Test that all modules use the same ad detection logic"""
    logger.info("Testing module consistency...")
    
    # Test text that should be detected as ad
    test_text = "Call 1-800-GRAINGER for all your industrial supplies"
    
    # Test centralized detector
    from services.ads import looks_like_ad
    centralized_result = looks_like_ad(test_text)
    
    # Test title_service delegation
    from services.title_service import _looks_like_ad
    title_service_result = _looks_like_ad(test_text)
    
    # Test quality_filters delegation
    from services.quality_filters import _ad_like_score
    quality_filters_result = _ad_like_score(test_text)
    
    logger.info(f"Module consistency test:")
    logger.info(f"  Centralized detector: {centralized_result}")
    logger.info(f"  Title service: {title_service_result}")
    logger.info(f"  Quality filters score: {quality_filters_result:.2f}")
    
    # All should agree that this is an ad
    passed = (centralized_result == title_service_result == True and 
              quality_filters_result >= 0.6)
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Module consistency: {'PASS' if passed else 'FAIL'}")
    
    return passed

def test_environment_configuration():
    """Test environment configuration functions"""
    logger.info("Testing environment configuration...")
    
    from services.ads import (
        is_enhanced_ad_filtering_enabled,
        get_preroll_seconds,
        is_strict_mode,
        is_log_only_mode
    )
    
    # Test default values
    enhanced = is_enhanced_ad_filtering_enabled()
    preroll = get_preroll_seconds()
    strict = is_strict_mode()
    log_only = is_log_only_mode()
    
    logger.info(f"Environment configuration:")
    logger.info(f"  Enhanced filtering: {enhanced}")
    logger.info(f"  Pre-roll seconds: {preroll}")
    logger.info(f"  Strict mode: {strict}")
    logger.info(f"  Log only mode: {log_only}")
    
    # Test that we get reasonable defaults
    passed = (enhanced == True and  # Should be enabled by default
              preroll == 35 and     # Should be 35 seconds
              strict == False and   # Should be non-strict by default
              log_only == False)    # Should be active by default
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Environment configuration: {'PASS' if passed else 'FAIL'}")
    
    return passed

def test_filter_function():
    """Test the filter_ads_from_features function"""
    logger.info("Testing filter_ads_from_features function...")
    
    from services.ads import filter_ads_from_features
    
    # Test features with mixed content
    test_features = [
        {"id": "1", "text": "Call 1-800-GRAINGER for supplies", "start": 0, "end": 5},
        {"id": "2", "text": "I love talking about technology", "start": 5, "end": 10},
        {"id": "3", "text": "Visit www.example.com today", "start": 10, "end": 15},
        {"id": "4", "text": "The weather is nice today", "start": 15, "end": 20},
    ]
    
    filtered_features = filter_ads_from_features(test_features)
    
    logger.info(f"Filter test:")
    logger.info(f"  Input features: {len(test_features)}")
    logger.info(f"  Filtered features: {len(filtered_features)}")
    logger.info(f"  Remaining IDs: {[f['id'] for f in filtered_features]}")
    
    # Should filter out features 1 and 3 (ads), keep 2 and 4 (normal content)
    expected_ids = ["2", "4"]
    actual_ids = [f["id"] for f in filtered_features]
    
    passed = actual_ids == expected_ids
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Filter function: {'PASS' if passed else 'FAIL'}")
    
    return passed

def test_edge_cases():
    """Test edge cases for the canonical ad detector"""
    logger.info("Testing edge cases...")
    
    from services.ads import looks_like_ad, ad_like_score, ad_penalty
    
    # Edge cases
    edge_cases = [
        ("", False, "Empty string"),
        (None, False, "None input"),
        ("   ", False, "Whitespace only"),
        ("Call 1-800-GRAINGER", True, "Phone number only"),
        ("Visit www.example.com", True, "URL only"),
        ("Use code SAVE20", True, "Promo code only"),
    ]
    
    results = []
    
    for text, expected, description in edge_cases:
        try:
            is_ad = looks_like_ad(text)
            score = ad_like_score(text)
            penalty = ad_penalty(text)
            
            passed = (is_ad == expected and 
                     penalty["is_advertisement"] == expected)
            
            status = "✓" if passed else "✗"
            logger.info(f"{status} {description}: {is_ad} (expected: {expected})")
            
            results.append(passed)
        except Exception as e:
            logger.error(f"✗ {description}: Exception {e}")
            results.append(False)
    
    all_passed = all(results)
    logger.info(f"Edge cases test: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

if __name__ == "__main__":
    logger.info("Starting Canonical Ad Detector tests...")
    
    tests = [
        ("Canonical Ad Detector", test_canonical_ad_detector),
        ("Module Consistency", test_module_consistency),
        ("Environment Configuration", test_environment_configuration),
        ("Filter Function", test_filter_function),
        ("Edge Cases", test_edge_cases),
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
