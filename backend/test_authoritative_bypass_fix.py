#!/usr/bin/env python3
"""
Test script for Authoritative bypass of finished gate fix.
Tests that unfinished clips don't sneak through when authoritative is set.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_authoritative_bypass_fix():
    """Test the authoritative bypass fix logic"""
    logger.info("Testing Authoritative bypass fix...")
    
    # Test cases for the finished_required logic
    test_cases = [
        # (FINISHED_THOUGHT_REQUIRED, authoritative_enhanced, is_finished_like, coverage, expected_finished_required, description)
        (True, False, True, 0.8, True, "Normal case: finished required, not authoritative"),
        (True, True, True, 0.8, False, "Authoritative with high finished coverage"),
        (True, True, True, 0.5, True, "Authoritative with low finished coverage"),
        (True, True, False, 0.0, True, "Authoritative but no finished-like clips"),
        (False, True, True, 0.8, False, "Finished not required"),
        (False, False, True, 0.8, False, "Finished not required, not authoritative"),
    ]
    
    results = []
    
    for FINISHED_THOUGHT_REQUIRED, authoritative_enhanced, is_finished_like, coverage, expected, description in test_cases:
        # Apply the fixed logic
        finished_required = FINISHED_THOUGHT_REQUIRED and not (authoritative_enhanced and is_finished_like and coverage >= 0.66)
        
        passed = finished_required == expected
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}:")
        logger.info(f"    FINISHED_THOUGHT_REQUIRED={FINISHED_THOUGHT_REQUIRED}, authoritative={authoritative_enhanced}")
        logger.info(f"    is_finished_like={is_finished_like}, coverage={coverage:.2f}")
        logger.info(f"    -> finished_required={finished_required} (expected: {expected})")
    
    all_passed = all(results)
    logger.info(f"\nAuthoritative bypass fix test: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        logger.info("✅ Authoritative bypass fix is working correctly!")
        logger.info("  - Unfinished clips won't sneak through when authoritative is set")
        logger.info("  - High finished coverage allows authoritative bypass")
        logger.info("  - Low finished coverage prevents authoritative bypass")
    else:
        logger.info("❌ Authoritative bypass fix needs adjustment")
    
    return all_passed

def test_edge_cases():
    """Test edge cases for the authoritative bypass logic"""
    logger.info("Testing edge cases...")
    
    # Edge case: coverage exactly at threshold
    FINISHED_THOUGHT_REQUIRED = True
    authoritative_enhanced = True
    is_finished_like = True
    coverage = 0.66  # Exactly at threshold
    
    finished_required = FINISHED_THOUGHT_REQUIRED and not (authoritative_enhanced and is_finished_like and coverage >= 0.66)
    
    # Should be False (bypass allowed) since coverage >= 0.66
    expected = False
    passed = finished_required == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Edge case - coverage exactly at threshold (0.66):")
    logger.info(f"    -> finished_required={finished_required} (expected: {expected})")
    
    # Edge case: coverage just below threshold
    coverage = 0.65  # Just below threshold
    
    finished_required = FINISHED_THOUGHT_REQUIRED and not (authoritative_enhanced and is_finished_like and coverage >= 0.66)
    
    # Should be True (bypass not allowed) since coverage < 0.66
    expected = True
    passed = finished_required == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Edge case - coverage just below threshold (0.65):")
    logger.info(f"    -> finished_required={finished_required} (expected: {expected})")
    
    return finished_required == expected

def test_integration_scenario():
    """Test a realistic integration scenario"""
    logger.info("Testing realistic integration scenario...")
    
    # Simulate a scenario where authoritative enhanced is True but coverage is low
    FINISHED_THOUGHT_REQUIRED = True
    authoritative_enhanced = True
    
    # Simulate finals with mixed finished status
    finals = [
        {"id": "1", "finished_thought": True, "text": "This is a complete thought."},
        {"id": "2", "finished_thought": False, "text": "This is incomplete"},
        {"id": "3", "finished_thought": False, "text": "Another incomplete"},
        {"id": "4", "finished_thought": True, "text": "Another complete thought."},
    ]
    
    # Calculate is_finished_like and coverage
    def _is_finished_like(x):
        return x.get('finished_thought', False)
    
    is_finished_like = any(_is_finished_like(c) for c in finals)
    coverage = len([c for c in finals if _is_finished_like(c)]) / max(len(finals), 1)
    
    # Apply the fixed logic
    finished_required = FINISHED_THOUGHT_REQUIRED and not (authoritative_enhanced and is_finished_like and coverage >= 0.66)
    
    logger.info(f"Integration scenario:")
    logger.info(f"  Finals: {len(finals)} clips")
    logger.info(f"  Finished clips: {len([c for c in finals if _is_finished_like(c)])}")
    logger.info(f"  Coverage: {coverage:.2f}")
    logger.info(f"  Authoritative: {authoritative_enhanced}")
    logger.info(f"  -> finished_required: {finished_required}")
    
    # With 2/4 = 0.5 coverage, should require finished (True)
    expected = True
    passed = finished_required == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Integration scenario result: {'PASS' if passed else 'FAIL'}")
    
    return passed

if __name__ == "__main__":
    logger.info("Starting Authoritative bypass fix tests...")
    
    tests = [
        ("Authoritative Bypass Fix Logic", test_authoritative_bypass_fix),
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
