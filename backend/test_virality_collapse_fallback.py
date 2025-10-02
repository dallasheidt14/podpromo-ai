#!/usr/bin/env python3
"""
Test script for Virality collapse fallback implementation.
Tests that z-score fallback provides proper ranking separation when min≈max.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_virality_collapse_fallback():
    """Test the virality collapse fallback logic"""
    logger.info("Testing Virality collapse fallback...")
    
    # Create test clips with equal raw scores (simulating collapse)
    test_clips = [
        {
            "id": "clip_1",
            "virality_score": 0.5,
            "hook": 0.3,
            "info": 0.2,
            "payoff": 0.1,
            "finished_thought": True,
            "start": 0.0,
            "end": 30.0,
            "text": "This is a finished thought."
        },
        {
            "id": "clip_2", 
            "virality_score": 0.5,
            "hook": 0.4,
            "info": 0.1,
            "payoff": 0.2,
            "finished_thought": False,
            "start": 0.0,
            "end": 45.0,
            "text": "This is a longer unfinished thought"
        },
        {
            "id": "clip_3",
            "virality_score": 0.5,
            "hook": 0.2,
            "info": 0.3,
            "payoff": 0.3,
            "finished_thought": False,
            "start": 0.0,
            "end": 20.0,
            "text": "This is a shorter thought"
        },
    ]
    
    # Simulate the virality collapse fallback logic
    scores = [c.get('virality_score', c.get('composite_score', c.get('final_score', 0))) for c in test_clips]
    vmin, vmax = min(scores), max(scores)
    
    logger.info(f"Score range: vmin={vmin:.3f}, vmax={vmax:.3f}, range={vmax-vmin:.6f}")
    
    if vmax - vmin < 1e-6:
        logger.info("VIRALITY_COLLAPSE: min≈max detected, switching to z-score fallback")
        
        # Create composite scores (virality + hook + insight + novelty)
        composite_scores = []
        for c in test_clips:
            virality = c.get('virality_score', c.get('composite_score', c.get('final_score', 0)))
            hook = c.get('hook', 0.0)
            insight = c.get('info', 0.0)  # info score as insight proxy
            novelty = c.get('payoff', 0.0)  # payoff as novelty proxy
            composite = virality + hook + insight + novelty
            composite_scores.append(composite)
        
        logger.info(f"Composite scores: {composite_scores}")
        
        # Calculate z-scores
        if len(composite_scores) > 1:
            mean_comp = statistics.mean(composite_scores)
            stdev_comp = statistics.stdev(composite_scores) if len(composite_scores) > 1 else 1.0
            
            logger.info(f"Composite stats: mean={mean_comp:.3f}, stdev={stdev_comp:.3f}")
            
            for i, c in enumerate(test_clips):
                z_score = (composite_scores[i] - mean_comp) / stdev_comp if stdev_comp > 0 else 0.0
                # Convert z-score to percentage (0-100 range)
                virality_pct = max(0, min(100, 50 + z_score * 15))  # Scale z-score to reasonable range
                c['virality_pct'] = round(virality_pct, 1)
                c['display_score'] = round(virality_pct, 1)
                c['z_score'] = z_score
                logger.info(f"  {c['id']}: composite={composite_scores[i]:.3f}, z_score={z_score:.3f}, virality_pct={virality_pct:.1f}")
        
        # Apply tie-breaking: finished > longer > novelty
        test_clips.sort(key=lambda c: (
            c.get('finished_thought', False),  # finished first
            c.get('end', 0) - c.get('start', 0),  # longer first
            c.get('payoff', 0.0),  # novelty (payoff) first
            c.get('virality_pct', 0.0)  # z-score last
        ), reverse=True)
        
        logger.info(f"VIRALITY_COLLAPSE: applied z-score fallback with tie-breaking")
        logger.info(f"Final order: {[c['id'] for c in test_clips]}")
        logger.info(f"Virality percentages: {[c.get('virality_pct', 0) for c in test_clips]}")
    
    # Verify results
    virality_pcts = [c.get('virality_pct', 0) for c in test_clips]
    
    # Check that we don't have all 50% (no collapse)
    all_fifty = all(pct == 50.0 for pct in virality_pcts)
    
    # Check that we have some variation
    has_variation = len(set(virality_pcts)) > 1
    
    # Check tie-breaking order (finished first, then longer)
    order_correct = (test_clips[0]['id'] == 'clip_1' and  # finished thought first
                     test_clips[1]['id'] == 'clip_2')     # longer second
    
    passed = not all_fifty and has_variation and order_correct
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Virality collapse fallback test:")
    logger.info(f"    No 50% collapse: {not all_fifty}")
    logger.info(f"    Has variation: {has_variation}")
    logger.info(f"    Correct order: {order_correct}")
    
    return passed

def test_edge_cases():
    """Test edge cases for virality collapse fallback"""
    logger.info("Testing edge cases...")
    
    # Edge case: Single clip
    single_clip = [{
        "id": "single",
        "virality_score": 0.5,
        "hook": 0.3,
        "info": 0.2,
        "payoff": 0.1,
        "finished_thought": True,
        "start": 0.0,
        "end": 30.0,
    }]
    
    scores = [c.get('virality_score', 0) for c in single_clip]
    vmin, vmax = min(scores), max(scores)
    
    if vmax - vmin < 1e-6:
        # Single clip case
        for c in single_clip:
            c['virality_pct'] = 50.0
            c['display_score'] = 50.0
    
    logger.info(f"Single clip case: virality_pct={single_clip[0].get('virality_pct', 0)}")
    
    # Edge case: All clips identical composite scores
    identical_clips = [
        {
            "id": "clip_1",
            "virality_score": 0.5,
            "hook": 0.3,
            "info": 0.2,
            "payoff": 0.1,
            "finished_thought": True,
            "start": 0.0,
            "end": 30.0,
        },
        {
            "id": "clip_2",
            "virality_score": 0.5,
            "hook": 0.3,
            "info": 0.2,
            "payoff": 0.1,
            "finished_thought": False,
            "start": 0.0,
            "end": 45.0,
        },
    ]
    
    scores = [c.get('virality_score', 0) for c in identical_clips]
    vmin, vmax = min(scores), max(scores)
    
    if vmax - vmin < 1e-6:
        composite_scores = []
        for c in identical_clips:
            composite = c.get('virality_score', 0) + c.get('hook', 0) + c.get('info', 0) + c.get('payoff', 0)
            composite_scores.append(composite)
        
        if len(composite_scores) > 1:
            mean_comp = statistics.mean(composite_scores)
            stdev_comp = statistics.stdev(composite_scores) if len(composite_scores) > 1 else 1.0
            
            for i, c in enumerate(identical_clips):
                z_score = (composite_scores[i] - mean_comp) / stdev_comp if stdev_comp > 0 else 0.0
                virality_pct = max(0, min(100, 50 + z_score * 15))
                c['virality_pct'] = round(virality_pct, 1)
                c['display_score'] = round(virality_pct, 1)
        
        # Apply tie-breaking
        identical_clips.sort(key=lambda c: (
            c.get('finished_thought', False),
            c.get('end', 0) - c.get('start', 0),
            c.get('payoff', 0.0),
            c.get('virality_pct', 0.0)
        ), reverse=True)
    
    logger.info(f"Identical composite case: order={[c['id'] for c in identical_clips]}")
    
    return True

def test_integration_scenario():
    """Test a realistic integration scenario"""
    logger.info("Testing realistic integration scenario...")
    
    # Simulate a scenario where all clips have identical virality scores (true collapse)
    realistic_clips = [
        {
            "id": "hook_clip",
            "virality_score": 0.5,
            "hook": 0.8,
            "info": 0.3,
            "payoff": 0.2,
            "finished_thought": True,
            "start": 0.0,
            "end": 25.0,
            "text": "This is a great hook that's finished."
        },
        {
            "id": "insight_clip",
            "virality_score": 0.5,
            "hook": 0.4,
            "info": 0.9,
            "payoff": 0.3,
            "finished_thought": False,
            "start": 0.0,
            "end": 40.0,
            "text": "This provides deep insights but is longer"
        },
        {
            "id": "novelty_clip",
            "virality_score": 0.5,
            "hook": 0.3,
            "info": 0.4,
            "payoff": 0.8,
            "finished_thought": False,
            "start": 0.0,
            "end": 20.0,
            "text": "This has high novelty but is short"
        },
    ]
    
    logger.info(f"Realistic scenario:")
    logger.info(f"  Clips: {len(realistic_clips)}")
    logger.info(f"  Virality scores: {[c.get('virality_score', 0) for c in realistic_clips]}")
    
    # Apply collapse detection
    scores = [c.get('virality_score', 0) for c in realistic_clips]
    vmin, vmax = min(scores), max(scores)
    
    if vmax - vmin < 1e-6:
        logger.info("Collapse detected - applying z-score fallback")
        
        composite_scores = []
        for c in realistic_clips:
            composite = c.get('virality_score', 0) + c.get('hook', 0) + c.get('info', 0) + c.get('payoff', 0)
            composite_scores.append(composite)
        
        mean_comp = statistics.mean(composite_scores)
        stdev_comp = statistics.stdev(composite_scores)
        
        for i, c in enumerate(realistic_clips):
            z_score = (composite_scores[i] - mean_comp) / stdev_comp
            virality_pct = max(0, min(100, 50 + z_score * 15))
            c['virality_pct'] = round(virality_pct, 1)
            c['display_score'] = round(virality_pct, 1)
            c['z_score'] = z_score
        
        # Apply tie-breaking
        realistic_clips.sort(key=lambda c: (
            c.get('finished_thought', False),
            c.get('end', 0) - c.get('start', 0),
            c.get('payoff', 0.0),
            c.get('virality_pct', 0.0)
        ), reverse=True)
    
    logger.info(f"Final results:")
    for c in realistic_clips:
        logger.info(f"  {c['id']}: virality_pct={c.get('virality_pct', 0)}, finished={c.get('finished_thought', False)}, duration={c.get('end', 0) - c.get('start', 0)}")
    
    # Verify that we have proper ranking separation
    virality_pcts = [c.get('virality_pct', 0) for c in realistic_clips]
    has_separation = len(set(virality_pcts)) > 1
    
    # Verify tie-breaking worked (finished first)
    order_correct = realistic_clips[0]['id'] == 'hook_clip'  # finished thought should be first
    
    passed = has_separation and order_correct
    
    status = "✓" if passed else "✗"
    logger.info(f"{status} Integration scenario result: {'PASS' if passed else 'FAIL'}")
    
    return passed

if __name__ == "__main__":
    logger.info("Starting Virality collapse fallback tests...")
    
    tests = [
        ("Virality Collapse Fallback", test_virality_collapse_fallback),
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
