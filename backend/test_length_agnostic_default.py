#!/usr/bin/env python3
"""
Test script for Length-agnostic default OFF change.
Tests that LENGTH_AGNOSTIC defaults to OFF (0) instead of ON (1).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_length_agnostic_default():
    """Test that LENGTH_AGNOSTIC defaults to OFF"""
    logger.info("Testing Length-agnostic default OFF...")
    
    # Test the _get_platform_mode function
    from services.clip_score import _get_platform_mode
    
    # Clear any existing LENGTH_AGNOSTIC env var
    if 'LENGTH_AGNOSTIC' in os.environ:
        del os.environ['LENGTH_AGNOSTIC']
    
    # Test default behavior (should be OFF/False)
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"Default behavior:")
    logger.info(f"  pl_v2_weight: {pl_v2_weight}")
    logger.info(f"  platform_protect: {platform_protect}")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    # Should be False (OFF) by default
    expected = False
    passed = length_agnostic == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Default length_agnostic: {length_agnostic} (expected: {expected})")
    
    return passed

def test_env_override():
    """Test that LENGTH_AGNOSTIC can still be overridden by env"""
    logger.info("Testing LENGTH_AGNOSTIC env override...")
    
    from services.clip_score import _get_platform_mode
    
    # Test with LENGTH_AGNOSTIC=1
    os.environ['LENGTH_AGNOSTIC'] = '1'
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"With LENGTH_AGNOSTIC=1:")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    expected = True
    passed = length_agnostic == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} LENGTH_AGNOSTIC=1: {length_agnostic} (expected: {expected})")
    
    # Test with LENGTH_AGNOSTIC=0
    os.environ['LENGTH_AGNOSTIC'] = '0'
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"With LENGTH_AGNOSTIC=0:")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    expected = False
    passed = length_agnostic == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} LENGTH_AGNOSTIC=0: {length_agnostic} (expected: {expected})")
    
    # Test with LENGTH_AGNOSTIC=true
    os.environ['LENGTH_AGNOSTIC'] = 'true'
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"With LENGTH_AGNOSTIC=true:")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    expected = True
    passed = length_agnostic == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} LENGTH_AGNOSTIC=true: {length_agnostic} (expected: {expected})")
    
    # Clean up
    if 'LENGTH_AGNOSTIC' in os.environ:
        del os.environ['LENGTH_AGNOSTIC']
    
    return passed

def test_edge_cases():
    """Test edge cases for LENGTH_AGNOSTIC"""
    logger.info("Testing edge cases...")
    
    from services.clip_score import _get_platform_mode
    
    # Test with empty string
    os.environ['LENGTH_AGNOSTIC'] = ''
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"With LENGTH_AGNOSTIC='':")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    # Empty string should be False
    expected = False
    passed = length_agnostic == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} LENGTH_AGNOSTIC='': {length_agnostic} (expected: {expected})")
    
    # Test with invalid value
    os.environ['LENGTH_AGNOSTIC'] = 'invalid'
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"With LENGTH_AGNOSTIC='invalid':")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    # Invalid value should be False
    expected = False
    passed = length_agnostic == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} LENGTH_AGNOSTIC='invalid': {length_agnostic} (expected: {expected})")
    
    # Clean up
    if 'LENGTH_AGNOSTIC' in os.environ:
        del os.environ['LENGTH_AGNOSTIC']
    
    return passed

def test_integration_scenario():
    """Test a realistic integration scenario"""
    logger.info("Testing realistic integration scenario...")
    
    from services.clip_score import _get_platform_mode
    
    # Simulate a production environment without LENGTH_AGNOSTIC set
    if 'LENGTH_AGNOSTIC' in os.environ:
        del os.environ['LENGTH_AGNOSTIC']
    
    pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
    
    logger.info(f"Production scenario (no LENGTH_AGNOSTIC env var):")
    logger.info(f"  pl_v2_weight: {pl_v2_weight}")
    logger.info(f"  platform_protect: {platform_protect}")
    logger.info(f"  length_agnostic: {length_agnostic}")
    
    # In production, length_agnostic should be False (OFF) by default
    expected_length_agnostic = False
    expected_platform_protect = True  # Should still be True by default
    
    passed = (length_agnostic == expected_length_agnostic and 
              platform_protect == expected_platform_protect)
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Production scenario result: {'PASS' if passed else 'FAIL'}")
    
    if passed:
        logger.info("✅ Length-agnostic defaults to OFF in production")
        logger.info("✅ Platform protection remains enabled")
    else:
        logger.info("❌ Unexpected behavior in production scenario")
    
    return passed

if __name__ == "__main__":
    logger.info("Starting Length-agnostic default OFF tests...")
    
    tests = [
        ("Length-agnostic Default OFF", test_length_agnostic_default),
        ("Environment Override", test_env_override),
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
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
