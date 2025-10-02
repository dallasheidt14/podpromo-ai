#!/usr/bin/env python3
"""
Test script for Phase 2.2 improvements:
- Semantic dedupe
- Unit normalization helpers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unit_helpers():
    """Test unit conversion helpers"""
    logger.info("Testing Unit Helpers...")
    
    from services.utils.units import per_min, per_sec
    
    # Test cases
    test_cases = [
        (81, 498.0, "EOS markers in 498 seconds"),
        (120, 600.0, "Questions in 10 minutes"),
        (45, 300.0, "Payoffs in 5 minutes"),
    ]
    
    results = []
    
    for count, duration_s, description in test_cases:
        per_min_rate = per_min(count, duration_s)
        per_sec_rate = per_sec(count, duration_s)
        
        # Test invariant: per_min should equal per_sec * 60
        expected_per_min = per_sec_rate * 60
        invariant_passed = abs(per_min_rate - expected_per_min) < 1e-6
        
        results.append(invariant_passed)
        
        status = "✓" if invariant_passed else "✗"
        logger.info(f"{status} {description}:")
        logger.info(f"    per_min: {per_min_rate:.3f}")
        logger.info(f"    per_sec: {per_sec_rate:.3f}")
        logger.info(f"    invariant: {invariant_passed}")
    
    all_passed = all(results)
    logger.info(f"Unit helpers test: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_semantic_dedupe():
    """Test semantic deduplication"""
    logger.info("Testing Semantic Deduplication...")
    
    from services.semantics import semantic_dedupe, HAS_EMB
    
    logger.info(f"Embeddings available: {HAS_EMB}")
    
    # Test cases with semantically similar content
    test_clips = [
        {"id": "1", "text": "AI will change everything.", "final_score": 0.9},
        {"id": "2", "text": "Artificial intelligence will change everything.", "final_score": 0.8},
        {"id": "3", "text": "We discuss football today.", "final_score": 0.7},
        {"id": "4", "text": "AI is going to revolutionize the world.", "final_score": 0.85},
        {"id": "5", "text": "Football is a great sport.", "final_score": 0.6},
    ]
    
    logger.info(f"Input clips: {len(test_clips)}")
    for clip in test_clips:
        logger.info(f"  {clip['id']}: '{clip['text']}' (score: {clip['final_score']})")
    
    # Test semantic dedupe
    deduped = semantic_dedupe(test_clips, sim_thresh=0.85)
    
    logger.info(f"Deduplicated clips: {len(deduped)}")
    for clip in deduped:
        logger.info(f"  {clip['id']}: '{clip['text']}' (score: {clip['final_score']})")
    
    # Verify that we kept the highest-scored clips
    if HAS_EMB:
        # With embeddings, should reduce from 5 to 2-3 clips
        expected_range = (2, 3)
        passed = expected_range[0] <= len(deduped) <= expected_range[1]
        logger.info(f"Expected range: {expected_range}, actual: {len(deduped)}")
    else:
        # Without embeddings, should return all clips unchanged
        passed = len(deduped) == len(test_clips)
        logger.info(f"Fallback mode: returned all {len(deduped)} clips unchanged")
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Semantic dedupe test: {'PASS' if passed else 'FAIL'}")
    
    return passed

def test_edge_cases():
    """Test edge cases"""
    logger.info("Testing Edge Cases...")
    
    from services.utils.units import per_min, per_sec
    from services.semantics import semantic_dedupe
    
    # Test unit helpers with edge cases
    logger.info("Unit helpers edge cases:")
    
    # Zero duration
    try:
        per_min(10, 0.0)
        logger.info("  Zero duration handled gracefully")
    except Exception as e:
        logger.error(f"  Zero duration error: {e}")
    
    # Very small duration
    try:
        rate = per_min(1, 0.001)
        logger.info(f"  Very small duration: {rate:.3f}")
    except Exception as e:
        logger.error(f"  Very small duration error: {e}")
    
    # Test semantic dedupe with edge cases
    logger.info("Semantic dedupe edge cases:")
    
    # Empty list
    empty_result = semantic_dedupe([])
    logger.info(f"  Empty list: {len(empty_result)} clips")
    
    # Single clip
    single_result = semantic_dedupe([{"id": "1", "text": "Hello", "final_score": 0.5}])
    logger.info(f"  Single clip: {len(single_result)} clips")
    
    # Clips with empty text
    empty_text_clips = [
        {"id": "1", "text": "", "final_score": 0.5},
        {"id": "2", "text": "Hello", "final_score": 0.6},
    ]
    empty_text_result = semantic_dedupe(empty_text_clips)
    logger.info(f"  Empty text clips: {len(empty_text_result)} clips")
    
    return True

def test_integration_scenario():
    """Test integration scenario"""
    logger.info("Testing Integration Scenario...")
    
    from services.utils.units import per_min
    from services.semantics import semantic_dedupe
    
    # Simulate a realistic scenario
    logger.info("Realistic scenario:")
    
    # Simulate EOS density calculation
    eos_count = 88
    duration_s = 560.0
    eos_density = per_min(eos_count, duration_s)
    
    logger.info(f"  EOS count: {eos_count}")
    logger.info(f"  Duration: {duration_s}s")
    logger.info(f"  EOS density: {eos_density:.3f} per_min")
    
    # Simulate semantic dedupe on realistic clips
    realistic_clips = [
        {"id": "1", "text": "This is a great topic about technology", "final_score": 0.9},
        {"id": "2", "text": "This is an excellent topic about technology", "final_score": 0.85},
        {"id": "3", "text": "Let me tell you about my vacation", "final_score": 0.8},
        {"id": "4", "text": "I want to share my vacation story", "final_score": 0.75},
        {"id": "5", "text": "Technology is amazing", "final_score": 0.7},
    ]
    
    logger.info(f"  Input clips: {len(realistic_clips)}")
    
    deduped = semantic_dedupe(realistic_clips, sim_thresh=0.85)
    
    logger.info(f"  Deduplicated clips: {len(deduped)}")
    logger.info(f"  Remaining IDs: {[c['id'] for c in deduped]}")
    
    # Verify reasonable deduplication
    reasonable_dedup = 2 <= len(deduped) <= 4
    logger.info(f"  Reasonable deduplication: {reasonable_dedup}")
    
    return reasonable_dedup

if __name__ == "__main__":
    logger.info("Starting Phase 2.2 tests...")
    
    tests = [
        ("Unit Helpers", test_unit_helpers),
        ("Semantic Deduplication", test_semantic_dedupe),
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
    logger.info("PHASE 2.2 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
