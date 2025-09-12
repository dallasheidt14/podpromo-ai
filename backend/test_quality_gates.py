#!/usr/bin/env python3
"""
Test script to verify quality gate improvements
"""

import os
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.clip_score import apply_quality_gate, adaptive_gate, enforce_soft_floor, THRESHOLDS

def create_test_candidates():
    """Create test candidates with various scores"""
    return [
        {
            "id": "clip_1",
            "final_score": 0.92,
            "payoff_score": 0.20,
            "hook_score": 0.15,
            "arousal_score": 0.45,
            "info_density": 0.70,
            "platform_length_score_v2": 0.85,
            "text": "Last year, Salem... major restructuring...",
            "start": 10.0,
            "end": 25.0,
            "quality_ok": True
        },
        {
            "id": "clip_2", 
            "final_score": 0.91,
            "payoff_score": 0.15,
            "hook_score": 0.12,
            "arousal_score": 0.40,
            "info_density": 0.65,
            "platform_length_score_v2": 0.80,
            "text": "I'll ever say we're going to be a digital-first company...",
            "start": 30.0,
            "end": 45.0,
            "quality_ok": True
        },
        {
            "id": "clip_3",
            "final_score": 0.72,
            "payoff_score": 0.10,
            "hook_score": 0.08,
            "arousal_score": 0.35,
            "info_density": 0.60,
            "platform_length_score_v2": 0.44,
            "text": "What's your biggest challenge right now?",
            "start": 50.0,
            "end": 65.0,
            "is_question": True
        },
        {
            "id": "clip_4",
            "final_score": 0.68,
            "payoff_score": 0.08,
            "hook_score": 0.06,
            "arousal_score": 0.30,
            "info_density": 0.55,
            "platform_length_score_v2": 0.65,
            "text": "We're seeing great results with our new approach...",
            "start": 70.0,
            "end": 85.0
        },
        {
            "id": "clip_5",
            "final_score": 0.45,
            "payoff_score": 0.05,
            "hook_score": 0.04,
            "arousal_score": 0.25,
            "info_density": 0.40,
            "platform_length_score_v2": 0.50,
            "text": "Thank you for having me on the show...",
            "start": 90.0,
            "end": 105.0,
            "is_ad": True
        }
    ]

def test_quality_gate_auto_relax():
    """Test that strict gate auto-relaxes when too few clips remain"""
    print("🔧 Testing Quality Gate Auto-Relaxation")
    print("=" * 50)
    
    candidates = create_test_candidates()
    
    # Test strict mode - should auto-relax since only 2 pass strict thresholds
    result = apply_quality_gate(candidates, mode="strict")
    
    print(f"Strict mode result: {len(result)} clips")
    for i, clip in enumerate(result, 1):
        print(f"  {i}. {clip['id']} (score: {clip['final_score']:.2f}, payoff: {clip['payoff_score']:.2f})")
    
    # Should have at least 3 clips due to auto-relaxation
    assert len(result) >= 3, f"Expected >= 3 clips, got {len(result)}"
    print("✅ Auto-relaxation working correctly")

def test_adaptive_gate():
    """Test adaptive gate adds clips from remaining pool"""
    print("\n🎯 Testing Adaptive Gate")
    print("=" * 40)
    
    candidates = create_test_candidates()
    
    # Mark some as quality_ok
    candidates[0]["quality_ok"] = True
    candidates[1]["quality_ok"] = True
    
    result = adaptive_gate(candidates, min_count=3)
    
    print(f"Adaptive gate result: {len(result)} clips")
    for i, clip in enumerate(result, 1):
        print(f"  {i}. {clip['id']} (score: {clip['final_score']:.2f}, quality_ok: {clip.get('quality_ok', False)})")
    
    # Should have at least 3 clips
    assert len(result) >= 3, f"Expected >= 3 clips, got {len(result)}"
    print("✅ Adaptive gate working correctly")

def test_soft_floor():
    """Test soft floor enforcement"""
    print("\n🛡️ Testing Soft Floor Enforcement")
    print("=" * 40)
    
    # Start with only 2 clips
    candidates = create_test_candidates()[:2]
    
    result = enforce_soft_floor(candidates, min_count=3)
    
    print(f"Soft floor result: {len(result)} clips")
    for i, clip in enumerate(result, 1):
        print(f"  {i}. {clip['id']} (score: {clip['final_score']:.2f})")
    
    # Should have at least 3 clips
    assert len(result) >= 3, f"Expected >= 3 clips, got {len(result)}"
    print("✅ Soft floor working correctly")

def test_thresholds():
    """Test that thresholds are properly loosened"""
    print("\n📊 Testing Threshold Values")
    print("=" * 40)
    
    print("Strict thresholds:")
    for key, value in THRESHOLDS["strict"].items():
        print(f"  {key}: {value}")
    
    print("\nBalanced thresholds:")
    for key, value in THRESHOLDS["balanced"].items():
        print(f"  {key}: {value}")
    
    # Verify payoff_min is loosened
    assert THRESHOLDS["strict"]["payoff_min"] == 0.12, "payoff_min should be 0.12"
    assert THRESHOLDS["balanced"]["payoff_min"] == 0.08, "balanced payoff_min should be 0.08"
    
    # Verify ql_max is loosened
    assert THRESHOLDS["strict"]["ql_max"] == 0.78, "ql_max should be 0.78"
    assert THRESHOLDS["balanced"]["ql_max"] == 0.85, "balanced ql_max should be 0.85"
    
    print("✅ Thresholds properly loosened")

def test_integration():
    """Test the full pipeline integration"""
    print("\n🚀 Testing Full Pipeline Integration")
    print("=" * 50)
    
    candidates = create_test_candidates()
    
    # Simulate the pipeline
    print("1. Apply quality gate (strict)")
    after_gate = apply_quality_gate(candidates, mode="strict")
    print(f"   After gate: {len(after_gate)} clips")
    
    print("2. Apply adaptive gate if needed")
    if len(after_gate) < 3:
        after_adaptive = adaptive_gate(after_gate, min_count=3)
        print(f"   After adaptive: {len(after_adaptive)} clips")
    else:
        after_adaptive = after_gate
        print("   Adaptive gate skipped (enough clips)")
    
    print("3. Enforce soft floor")
    final = enforce_soft_floor(after_adaptive, min_count=3)
    print(f"   Final result: {len(final)} clips")
    
    print("\nFinal clips:")
    for i, clip in enumerate(final, 1):
        print(f"  {i}. {clip['id']} (score: {clip['final_score']:.2f}, payoff: {clip['payoff_score']:.2f})")
    
    # Should have 3-4 clips
    assert 3 <= len(final) <= 5, f"Expected 3-5 clips, got {len(final)}"
    print("✅ Full pipeline working correctly")

if __name__ == "__main__":
    print("🧪 Quality Gate Improvements Test Suite")
    print("=" * 60)
    
    test_thresholds()
    test_quality_gate_auto_relax()
    test_adaptive_gate()
    test_soft_floor()
    test_integration()
    
    print("\n🎉 All tests passed! Quality gates are working correctly.")
