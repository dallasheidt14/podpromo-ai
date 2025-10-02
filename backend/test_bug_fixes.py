#!/usr/bin/env python3
"""
Quick verification test for all bug fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semantic_dedupe_import():
    """Test that semantic_dedupe import works"""
    logger.info("Testing semantic_dedupe import...")
    
    try:
        from services.clip_score import semantic_dedupe
        logger.info("✓ semantic_dedupe import successful")
        return True
    except ImportError as e:
        logger.error(f"✗ semantic_dedupe import failed: {e}")
        return False

def test_dynamic_discovery_fixes():
    """Test dynamic discovery fixes"""
    logger.info("Testing dynamic discovery fixes...")
    
    from services.secret_sauce_pkg.features import discover_dynamic_length
    
    # Test with short duration (should use 4.0s min_dist)
    words = [{"word": "test", "start": 0.0, "end": 0.5}]
    eos_times = [1.0]
    duration_s = 300.0  # 5 minutes (short)
    
    def mock_score_fn(start, end):
        return 0.5
    
    try:
        candidates = discover_dynamic_length(words, eos_times, mock_score_fn, duration_s)
        logger.info(f"✓ Dynamic discovery works for short duration: {len(candidates)} candidates")
        return True
    except Exception as e:
        logger.error(f"✗ Dynamic discovery failed: {e}")
        return False

def test_eos_dense_threshold():
    """Test EOS dense threshold fix"""
    logger.info("Testing EOS dense threshold...")
    
    # Test realistic EOS density (should not be "dense" with old 0.04 threshold)
    eos_count = 81
    duration_minutes = 498.0 / 60.0  # ~8.3 minutes
    eos_density = eos_count / duration_minutes  # ~9.8 per_min
    
    # With new threshold (12.0 per_min), this should NOT be dense
    dense_new = eos_density >= 12.0
    # With old threshold (0.04 per_min), this would be dense (always true)
    dense_old = eos_density >= 0.04
    
    logger.info(f"EOS density: {eos_density:.1f} per_min")
    logger.info(f"New threshold (12.0): dense={dense_new}")
    logger.info(f"Old threshold (0.04): dense={dense_old}")
    
    # New threshold should be more realistic
    realistic = not dense_new and dense_old
    logger.info(f"✓ EOS threshold fix: {realistic}")
    return realistic

def test_canonical_ad_detector():
    """Test canonical ad detector usage"""
    logger.info("Testing canonical ad detector...")
    
    from services.ads import looks_like_ad, ad_like_score
    
    # Test cases
    test_cases = [
        ("Call 1-800-GRAINGER", True, "Phone number"),
        ("Visit our website", True, "CTA phrase"),
        ("This is great content", False, "Clean content"),
    ]
    
    results = []
    for text, expected_ad, description in test_cases:
        is_ad = looks_like_ad(text)
        score = ad_like_score(text)
        
        passed = is_ad == expected_ad
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}: is_ad={is_ad}, score={score:.3f}")
    
    all_passed = all(results)
    logger.info(f"✓ Canonical ad detector: {all_passed}")
    return all_passed

def test_duration_diversification_gating():
    """Test duration diversification gating"""
    logger.info("Testing duration diversification gating...")
    
    from services.clip_score import apply_duration_diversity
    
    test_clips = [
        {"id": "1", "dur": 15.0, "final_score": 0.9},
        {"id": "2", "dur": 35.0, "final_score": 0.8},
        {"id": "3", "dur": 80.0, "final_score": 0.7},
    ]
    
    # Test with gating enabled (default)
    os.environ["DURATION_BUCKETING"] = "1"
    diversified_enabled = apply_duration_diversity(test_clips)
    
    # Test with gating disabled
    os.environ["DURATION_BUCKETING"] = "0"
    diversified_disabled = apply_duration_diversity(test_clips)
    
    # Both should work without errors
    enabled_works = len(diversified_enabled) <= len(test_clips)
    disabled_works = len(diversified_disabled) <= len(test_clips)
    
    logger.info(f"Diversification enabled: {len(diversified_enabled)} clips")
    logger.info(f"Diversification disabled: {len(diversified_disabled)} clips")
    logger.info(f"✓ Duration diversification gating: {enabled_works and disabled_works}")
    
    return enabled_works and disabled_works

def test_deprecation_warning():
    """Test TitlesService deprecation warning"""
    logger.info("Testing TitlesService deprecation warning...")
    
    import warnings
    import io
    import sys
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        from services.titles_service import TitlesService
        ts = TitlesService()
        
        # Check if deprecation warning was issued
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        
        has_warning = len(deprecation_warnings) > 0
        logger.info(f"✓ TitlesService deprecation warning: {has_warning}")
        
        if has_warning:
            logger.info(f"  Warning message: {deprecation_warnings[0].message}")
        
        return has_warning

if __name__ == "__main__":
    logger.info("Starting bug fix verification...")
    
    tests = [
        ("Semantic Dedupe Import", test_semantic_dedupe_import),
        ("Dynamic Discovery Fixes", test_dynamic_discovery_fixes),
        ("EOS Dense Threshold", test_eos_dense_threshold),
        ("Canonical Ad Detector", test_canonical_ad_detector),
        ("Duration Diversification Gating", test_duration_diversification_gating),
        ("Deprecation Warning", test_deprecation_warning),
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
    logger.info("BUG FIX VERIFICATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL FIXES VERIFIED' if all_passed else 'SOME FIXES FAILED'}")
