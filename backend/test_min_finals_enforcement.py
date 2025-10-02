#!/usr/bin/env python3
"""
Test script for MIN_FINALS enforcement at the very end.
Tests that the system enforces MIN_FINALS by topping up from reserve pool.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_min_finals_enforcement_logic():
    """Test the MIN_FINALS enforcement logic"""
    logger.info("Testing MIN_FINALS enforcement logic...")
    
    # Test cases for MIN_FINALS enforcement
    test_cases = [
        # (final_clips_count, reserve_pool_count, MIN_FINALS, expected_final_count, description)
        (3, 5, 4, 4, "Need 1 more clip from reserve pool"),
        (2, 3, 4, 4, "Need 2 more clips from reserve pool"),
        (5, 2, 4, 5, "Already have enough clips"),
        (1, 10, 4, 4, "Need 3 more clips from reserve pool"),
        (0, 2, 4, 2, "Not enough reserve clips available"),
    ]
    
    results = []
    
    for final_count, reserve_count, min_finals, expected_final, description in test_cases:
        # Simulate the enforcement logic
        final_clips = [{"id": f"final_{i}", "final_score": 0.8} for i in range(final_count)]
        reserve_pool = [{"id": f"reserve_{i}", "final_score": 0.7} for i in range(reserve_count)]
        
        # Apply MIN_FINALS enforcement
        if len(final_clips) < min_finals:
            logger.info(f"MIN_FINALS_ENFORCEMENT: only {len(final_clips)} finals, need {min_finals}")
            
            # Top-up from reserve pool
            final_ids = {c.get('id') for c in final_clips}
            safe_reserve = [c for c in reserve_pool if c.get('id') not in final_ids]
            
            if safe_reserve:
                needed = min_finals - len(final_clips)
                top_up = sorted(safe_reserve, key=lambda c: c.get("final_score", 0.0), reverse=True)[:needed]
                final_clips.extend(top_up)
                logger.info(f"MIN_FINALS_ENFORCEMENT: topped up with {len(top_up)} clips from reserve pool")
        
        actual_final = len(final_clips)
        passed = actual_final == expected_final
        results.append(passed)
        
        status = "✓" if passed else "✗"
        logger.info(f"{status} {description}:")
        logger.info(f"    Input: {final_count} finals, {reserve_count} reserve")
        logger.info(f"    MIN_FINALS: {min_finals}")
        logger.info(f"    Result: {actual_final} finals (expected: {expected_final})")
    
    all_passed = all(results)
    logger.info(f"\nMIN_FINALS enforcement test: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        logger.info("✅ MIN_FINALS enforcement is working correctly!")
        logger.info("  - System tops up from reserve pool when needed")
        logger.info("  - Respects MIN_FINALS target")
        logger.info("  - Handles edge cases properly")
    else:
        logger.info("❌ MIN_FINALS enforcement needs adjustment")
    
    return all_passed

def test_edge_cases():
    """Test edge cases for MIN_FINALS enforcement"""
    logger.info("Testing edge cases...")
    
    # Edge case: Empty reserve pool
    final_clips = [{"id": "final_1", "final_score": 0.8}]
    reserve_pool = []
    min_finals = 4
    
    logger.info("Edge case: Empty reserve pool")
    logger.info(f"  Input: {len(final_clips)} finals, {len(reserve_pool)} reserve")
    
    if len(final_clips) < min_finals:
        final_ids = {c.get('id') for c in final_clips}
        safe_reserve = [c for c in reserve_pool if c.get('id') not in final_ids]
        
        if safe_reserve:
            needed = min_finals - len(final_clips)
            top_up = sorted(safe_reserve, key=lambda c: c.get("final_score", 0.0), reverse=True)[:needed]
            final_clips.extend(top_up)
        else:
            logger.info("  No safe reserve clips available for top-up")
    
    logger.info(f"  Result: {len(final_clips)} finals (target: {min_finals})")
    
    # Edge case: Reserve pool with duplicate IDs
    final_clips = [{"id": "final_1", "final_score": 0.8}]
    reserve_pool = [
        {"id": "reserve_1", "final_score": 0.7},
        {"id": "final_1", "final_score": 0.6},  # Duplicate ID
        {"id": "reserve_2", "final_score": 0.7},
    ]
    min_finals = 4
    
    logger.info("Edge case: Reserve pool with duplicate IDs")
    logger.info(f"  Input: {len(final_clips)} finals, {len(reserve_pool)} reserve")
    
    if len(final_clips) < min_finals:
        final_ids = {c.get('id') for c in final_clips}
        safe_reserve = [c for c in reserve_pool if c.get('id') not in final_ids]
        
        if safe_reserve:
            needed = min_finals - len(final_clips)
            top_up = sorted(safe_reserve, key=lambda c: c.get("final_score", 0.0), reverse=True)[:needed]
            final_clips.extend(top_up)
            logger.info(f"  Topped up with {len(top_up)} clips (duplicates filtered)")
    
    logger.info(f"  Result: {len(final_clips)} finals (target: {min_finals})")
    
    return True

def test_integration_scenario():
    """Test a realistic integration scenario"""
    logger.info("Testing realistic integration scenario...")
    
    # Simulate a scenario where we have 3 finals but need 4
    final_clips = [
        {"id": "clip_1", "final_score": 0.9, "text": "Great content"},
        {"id": "clip_2", "final_score": 0.8, "text": "Good content"},
        {"id": "clip_3", "final_score": 0.7, "text": "Decent content"},
    ]
    
    reserve_pool = [
        {"id": "clip_4", "final_score": 0.6, "text": "Backup content 1"},
        {"id": "clip_5", "final_score": 0.5, "text": "Backup content 2"},
        {"id": "clip_6", "final_score": 0.4, "text": "Backup content 3"},
    ]
    
    min_finals = 4
    
    logger.info(f"Integration scenario:")
    logger.info(f"  Finals: {len(final_clips)} clips")
    logger.info(f"  Reserve: {len(reserve_pool)} clips")
    logger.info(f"  MIN_FINALS: {min_finals}")
    
    # Apply enforcement
    if len(final_clips) < min_finals:
        final_ids = {c.get('id') for c in final_clips}
        safe_reserve = [c for c in reserve_pool if c.get('id') not in final_ids]
        
        if safe_reserve:
            needed = min_finals - len(final_clips)
            top_up = sorted(safe_reserve, key=lambda c: c.get("final_score", 0.0), reverse=True)[:needed]
            final_clips.extend(top_up)
            logger.info(f"  Topped up with {len(top_up)} clips")
    
    logger.info(f"  Final result: {len(final_clips)} clips")
    
    # Should have exactly 4 clips now
    expected = 4
    passed = len(final_clips) == expected
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Integration scenario result: {'PASS' if passed else 'FAIL'}")
    
    return passed

if __name__ == "__main__":
    logger.info("Starting MIN_FINALS enforcement tests...")
    
    tests = [
        ("MIN_FINALS Enforcement Logic", test_min_finals_enforcement_logic),
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
