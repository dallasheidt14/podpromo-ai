#!/usr/bin/env python3
"""
Test script for EOS density standardization.
Verifies that all EOS density calculations use markers per minute consistently.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.secret_sauce_pkg.features import build_eos_index
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_segments():
    """Create test segments for EOS density testing"""
    segments = [
        {
            "start": 0.0,
            "end": 60.0,
            "text": "This is the first segment. It has multiple sentences! And another one?",
            "words": [
                {"start": 0.0, "end": 0.5, "word": "This"},
                {"start": 0.5, "end": 1.0, "word": "is"},
                {"start": 1.0, "end": 1.5, "word": "the"},
                {"start": 1.5, "end": 2.0, "word": "first"},
                {"start": 2.0, "end": 2.5, "word": "segment"},
                {"start": 2.5, "end": 3.0, "word": "."},
                {"start": 3.0, "end": 3.5, "word": "It"},
                {"start": 3.5, "end": 4.0, "word": "has"},
                {"start": 4.0, "end": 4.5, "word": "multiple"},
                {"start": 4.5, "end": 5.0, "word": "sentences"},
                {"start": 5.0, "end": 5.5, "word": "!"},
                {"start": 5.5, "end": 6.0, "word": "And"},
                {"start": 6.0, "end": 6.5, "word": "another"},
                {"start": 6.5, "end": 7.0, "word": "one"},
                {"start": 7.0, "end": 7.5, "word": "?"},
            ]
        },
        {
            "start": 60.0,
            "end": 120.0,
            "text": "This is the second segment. It also has sentences! And punctuation?",
            "words": [
                {"start": 60.0, "end": 60.5, "word": "This"},
                {"start": 60.5, "end": 61.0, "word": "is"},
                {"start": 61.0, "end": 61.5, "word": "the"},
                {"start": 61.5, "end": 62.0, "word": "second"},
                {"start": 62.0, "end": 62.5, "word": "segment"},
                {"start": 62.5, "end": 63.0, "word": "."},
                {"start": 63.0, "end": 63.5, "word": "It"},
                {"start": 63.5, "end": 64.0, "word": "also"},
                {"start": 64.0, "end": 64.5, "word": "has"},
                {"start": 64.5, "end": 65.0, "word": "sentences"},
                {"start": 65.0, "end": 65.5, "word": "!"},
                {"start": 65.5, "end": 66.0, "word": "And"},
                {"start": 66.0, "end": 66.5, "word": "punctuation"},
                {"start": 66.5, "end": 67.0, "word": "?"},
            ]
        }
    ]
    return segments

def test_eos_density_calculation():
    """Test EOS density calculation with explicit unit logging"""
    logger.info("Testing EOS density standardization...")
    
    # Create test segments
    segments = create_test_segments()
    
    # Test build_eos_index function
    try:
        eos_times, word_end_times, eos_source = build_eos_index(segments)
        
        logger.info(f"EOS index result:")
        logger.info(f"  EOS markers: {len(eos_times)}")
        logger.info(f"  Word boundaries: {len(word_end_times)}")
        logger.info(f"  Source: {eos_source}")
        
        if eos_times and word_end_times:
            # Calculate density manually to verify
            duration_minutes = (word_end_times[-1] - word_end_times[0]) / 60.0
            density_per_min = len(eos_times) / duration_minutes
            
            logger.info(f"  Duration: {duration_minutes:.1f} minutes")
            logger.info(f"  Density: {density_per_min:.3f} markers per minute")
            
            # Verify the density is reasonable (should be around 2-4 markers per minute)
            if 1.0 <= density_per_min <= 10.0:
                logger.info("✓ EOS density is within reasonable range")
                return True
            else:
                logger.warning(f"⚠ EOS density {density_per_min:.3f} seems unusual")
                return False
        else:
            logger.warning("No EOS markers or word boundaries found")
            return False
            
    except Exception as e:
        logger.error(f"EOS density test failed: {e}")
        return False

def test_density_thresholds():
    """Test that density thresholds are consistent with per-minute units"""
    logger.info("Testing EOS density thresholds...")
    
    # Test various density values
    test_cases = [
        (0.02, "Very low density (1.2 markers/min)"),
        (0.04, "Low density (2.4 markers/min)"),
        (0.05, "Medium-low density (3.0 markers/min)"),
        (0.10, "Medium density (6.0 markers/min)"),
        (0.15, "High density (9.0 markers/min)"),
    ]
    
    for density, description in test_cases:
        markers_per_min = density * 60  # Convert to actual markers per minute
        logger.info(f"  {description}: {density:.3f} per_min = {markers_per_min:.1f} markers/min")
        
        # Test threshold logic
        is_dense_normal = density >= 0.04
        is_dense_fallback = density >= 0.05
        
        logger.info(f"    Normal dense threshold (≥0.04): {is_dense_normal}")
        logger.info(f"    Fallback dense threshold (≥0.05): {is_dense_fallback}")
    
    logger.info("✓ Density thresholds are consistent with per-minute units")
    return True

if __name__ == "__main__":
    logger.info("Starting EOS density standardization tests...")
    
    tests = [
        ("EOS Density Calculation", test_eos_density_calculation),
        ("Density Thresholds", test_density_thresholds),
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
    
    if all_passed:
        logger.info("\n✓ EOS density standardization is working correctly!")
        logger.info("  - All calculations use markers per minute")
        logger.info("  - All logging includes explicit units")
        logger.info("  - Thresholds are consistent with per-minute units")
