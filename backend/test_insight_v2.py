#!/usr/bin/env python3
"""
Test suite for Insight V2 system
Tests both ViralMomentDetector V2 and _detect_insight_content_v2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.viral_moment_detector import ViralMomentDetector
from services.secret_sauce_pkg import _detect_insight_content_v2, _detect_insight_content
from config_loader import load_config

def test_insight_v2_config():
    """Test that Insight V2 configuration is properly loaded"""
    print("=== Testing Insight V2 Configuration ===")
    
    config = load_config()
    insight_v2_config = config.get("insight_v2", {})
    
    assert "enabled" in insight_v2_config, "insight_v2.enabled should be present"
    assert "confidence_multiplier" in insight_v2_config, "confidence_multiplier should be present"
    assert "evidence_weights" in insight_v2_config, "evidence_weights should be present"
    assert "flexibility" in insight_v2_config, "flexibility should be present"
    
    print("✅ Insight V2 configuration loaded correctly")

def test_insight_content_v2_evidence_detection():
    """Test evidence-based insight detection"""
    print("=== Testing Insight Content V2 Evidence Detection ===")
    
    # Test contrast detection
    text1 = "Most people think fixed rates are better, but actually ARMs can save you money"
    score1, reasons1 = _detect_insight_content_v2(text1, "general")
    assert score1 > 0.5, f"Contrast should score high, got {score1}"
    assert "contrast" in reasons1, "Should detect contrast"
    print(f"✅ Contrast detection: {score1:.3f} - {reasons1}")
    
    # Test number detection
    text2 = "Here's the thing: refinancing can save you 2.5% on your rate"
    score2, reasons2 = _detect_insight_content_v2(text2, "general")
    assert score2 > 0.4, f"Number should score well, got {score2}"
    assert "number" in reasons2, "Should detect number"
    print(f"✅ Number detection: {score2:.3f} - {reasons2}")
    
    # Test comparison detection
    text3 = "The key is choosing between fixed vs adjustable rates"
    score3, reasons3 = _detect_insight_content_v2(text3, "general")
    assert score3 > 0.3, f"Comparison should score well, got {score3}"
    assert "comparison" in reasons3, "Should detect comparison"
    print(f"✅ Comparison detection: {score3:.3f} - {reasons3}")
    
    # Test imperative detection
    text4 = "Here's what you need to do: start by checking your credit score"
    score4, reasons4 = _detect_insight_content_v2(text4, "general")
    assert score4 > 0.3, f"Imperative should score well, got {score4}"
    assert "imperative" in reasons4, "Should detect imperative"
    print(f"✅ Imperative detection: {score4:.3f} - {reasons4}")
    
    # Test hedge penalty
    text5 = "I think maybe you should probably consider refinancing"
    score5, reasons5 = _detect_insight_content_v2(text5, "general")
    assert "hedge_penalty" in reasons5, "Should detect hedge penalty"
    print(f"✅ Hedge penalty: {score5:.3f} - {reasons5}")

def test_insight_content_v2_saturating_combiner():
    """Test saturating combiner for multiple evidence types"""
    print("=== Testing Insight Content V2 Saturating Combiner ===")
    
    # Test multiple evidence types
    text = "Most people think fixed rates are better, but actually ARMs can save you 2.5% vs fixed rates. Here's what you need to do: start by checking your credit score."
    score, reasons = _detect_insight_content_v2(text, "general")
    
    # Should have multiple evidence types
    evidence_types = ["contrast", "number", "comparison", "imperative"]
    found_types = [t for t in evidence_types if t in reasons]
    assert len(found_types) >= 3, f"Should find multiple evidence types, found: {found_types}"
    
    # Score should be high due to multiple evidence
    assert score > 0.7, f"Multiple evidence should score high, got {score}"
    print(f"✅ Multiple evidence: {score:.3f} - {reasons}")

def test_insight_content_v2_genre_specific():
    """Test genre-specific insight patterns"""
    print("=== Testing Insight Content V2 Genre-Specific Patterns ===")
    
    # Test fantasy sports patterns
    text = "Here's my observation: casual drafters are way better than serious players this season"
    score, reasons = _detect_insight_content_v2(text, "fantasy_sports")
    assert score > 0.4, f"Fantasy sports should score well, got {score}"
    assert "fantasy_insight" in reasons, "Should detect fantasy insight"
    print(f"✅ Fantasy sports: {score:.3f} - {reasons}")
    
    # Test specific insight boost
    text2 = "The key insight is that casual drafters are way better at picking sleepers"
    score2, reasons2 = _detect_insight_content_v2(text2, "fantasy_sports")
    assert "specific_insight_boost" in reasons2, "Should detect specific insight boost"
    print(f"✅ Specific insight boost: {score2:.3f} - {reasons2}")

def test_insight_content_v2_penalties():
    """Test penalty system"""
    print("=== Testing Insight Content V2 Penalties ===")
    
    # Test filler penalty
    text = "Hey guys, hope you're doing well today"
    score, reasons = _detect_insight_content_v2(text, "general")
    assert "filler_penalty" in reasons, "Should detect filler penalty"
    assert score < 0.3, f"Filler should score low, got {score}"
    print(f"✅ Filler penalty: {score:.3f} - {reasons}")
    
    # Test hedge penalty
    text2 = "I think maybe this could probably work"
    score2, reasons2 = _detect_insight_content_v2(text2, "general")
    assert "hedge_penalty" in reasons2, "Should detect hedge penalty"
    print(f"✅ Hedge penalty: {score2:.3f} - {reasons2}")

def test_viral_moment_detector_v2_evidence_requirements():
    """Test ViralMomentDetector V2 evidence requirements"""
    print("=== Testing ViralMomentDetector V2 Evidence Requirements ===")
    
    detector = ViralMomentDetector("general")
    
    # Test contrast insight (should be accepted)
    transcript1 = [
        {"start": 0.0, "end": 3.0, "text": "here's the thing"},
        {"start": 3.0, "end": 12.0, "text": "Most people think fixed is always better, but actually ARMs can save 1-2% early."},
    ]
    insights1 = detector._find_insights_v2(transcript1)
    assert len(insights1) > 0, "Should detect insight with contrast"
    assert insights1[0]["confidence"] >= 0.7, f"Contrast should have high confidence, got {insights1[0]['confidence']}"
    print(f"✅ Contrast insight: confidence {insights1[0]['confidence']:.3f}")
    
    # Test numeric claim (should be accepted)
    transcript2 = [
        {"start": 0.0, "end": 4.0, "text": "the truth is"},
        {"start": 4.0, "end": 12.0, "text": "Refinancing helps if you plan ahead and can save 2.5%."},
    ]
    insights2 = detector._find_insights_v2(transcript2)
    assert len(insights2) > 0, "Should detect insight with number"
    assert insights2[0]["confidence"] >= 0.7, f"Number should have high confidence, got {insights2[0]['confidence']}"
    print(f"✅ Numeric insight: confidence {insights2[0]['confidence']:.3f}")
    
    # Test long low-evidence (should be rejected)
    transcript3 = [
        {"start": 0.0, "end": 4.0, "text": "the truth is"},
        {"start": 4.0, "end": 22.0, "text": "Refinancing helps if you plan ahead and think about the future, you know."},
    ]
    insights3 = detector._find_insights_v2(transcript3)
    assert len(insights3) == 0, "Low-evidence long insight should be rejected"
    print("✅ Long low-evidence rejected correctly")
    
    # Test short low-evidence (should be accepted with lower confidence)
    transcript4 = [
        {"start": 0.0, "end": 2.5, "text": "here's the thing"},
        {"start": 2.5, "end": 12.0, "text": "Most people ignore their LTV before rate shopping, so you should check it first."},
    ]
    insights4 = detector._find_insights_v2(transcript4)
    assert len(insights4) > 0, "Short low-evidence should be accepted"
    assert insights4[0]["confidence"] < 0.8, f"Short low-evidence should have lower confidence, got {insights4[0]['confidence']}"
    print(f"✅ Short low-evidence: confidence {insights4[0]['confidence']:.3f}")

def test_viral_moment_detector_v2_window_expansion():
    """Test clause-aware window expansion"""
    print("=== Testing ViralMomentDetector V2 Window Expansion ===")
    
    detector = ViralMomentDetector("general")
    
    # Test window expansion across clauses with hard evidence
    transcript = [
        {"start": 0.0, "end": 3.0, "text": "Here's what matters."},
        {"start": 3.0, "end": 12.0, "text": "Drop PMI when LTV < 80% and you'll save 2.5%."},
        {"start": 12.0, "end": 15.0, "text": "This can save you thousands."},
    ]
    insights = detector._find_insights_v2(transcript)
    assert len(insights) > 0, "Should detect insight across expanded window with hard evidence"
    
    # Check that the window includes multiple segments and has evidence
    insight = insights[0]
    duration = insight["end"] - insight["start"]
    assert duration > 8.0, f"Window should expand beyond single segment, duration: {duration}"
    assert "80%" in insight["text"], "Window should include the numeric evidence"
    assert insight["confidence"] >= 0.70, f"Should have high confidence with hard evidence, got {insight['confidence']}"
    print(f"✅ Window expansion: duration {duration:.1f}s, confidence {insight['confidence']:.3f}")

def test_viral_moment_detector_v2_long_no_evidence_rejection():
    """Test that long insights without hard evidence are rejected"""
    print("=== Testing ViralMomentDetector V2 Long No-Evidence Rejection ===")
    
    detector = ViralMomentDetector("general")
    
    # Test long insight without hard evidence (should be rejected)
    transcript = [
        {"start": 0.0, "end": 4.0, "text": "Here's what matters."},
        {"start": 4.0, "end": 18.0, "text": "Refinancing helps if you plan ahead and think about the future."},
    ]
    insights = detector._find_insights_v2(transcript)
    assert len(insights) == 0, "Long insight without hard evidence should be rejected"
    print("✅ Long no-evidence insight correctly rejected")

def test_viral_moment_detector_v2_confidence_scoring():
    """Test confidence scoring system"""
    print("=== Testing ViralMomentDetector V2 Confidence Scoring ===")
    
    detector = ViralMomentDetector("general")
    
    # Test high confidence with multiple evidence
    transcript1 = [
        {"start": 0.0, "end": 3.0, "text": "here's the thing"},
        {"start": 3.0, "end": 12.0, "text": "Most people think fixed rates are better, but actually ARMs can save you 2.5% vs fixed rates."},
    ]
    insights1 = detector._find_insights_v2(transcript1)
    assert len(insights1) > 0, "Should detect high-confidence insight"
    assert insights1[0]["confidence"] >= 0.8, f"Multiple evidence should have high confidence, got {insights1[0]['confidence']}"
    print(f"✅ High confidence: {insights1[0]['confidence']:.3f}")
    
    # Test hedge penalty
    transcript2 = [
        {"start": 0.0, "end": 3.0, "text": "here's what I learned"},
        {"start": 3.0, "end": 12.0, "text": "Maybe you should probably consider refinancing, I guess."},
    ]
    insights2 = detector._find_insights_v2(transcript2)
    if len(insights2) > 0:  # May be rejected due to low evidence
        assert insights2[0]["confidence"] < 0.7, f"Hedge penalty should reduce confidence, got {insights2[0]['confidence']}"
        print(f"✅ Hedge penalty: {insights2[0]['confidence']:.3f}")

def test_insight_v2_vs_v1_comparison():
    """Test V2 vs V1 behavior differences"""
    print("=== Testing Insight V2 vs V1 Comparison ===")
    
    # Test same text with both versions
    text = "Most people think fixed rates are better, but actually ARMs can save you 2.5% vs fixed rates"
    
    score_v1, reasons_v1 = _detect_insight_content(text, "general")
    score_v2, reasons_v2 = _detect_insight_content_v2(text, "general")
    
    print(f"V1 Score: {score_v1:.3f} - {reasons_v1}")
    print(f"V2 Score: {score_v2:.3f} - {reasons_v2}")
    
    # V2 should generally score differently due to saturating combiner
    assert score_v1 != score_v2, "V1 and V2 should produce different scores"
    print("✅ V2 produces different scoring than V1")

def test_insight_confidence_multiplier():
    """Test confidence multiplier functionality"""
    print("=== Testing Insight Confidence Multiplier ===")
    
    from services.secret_sauce_pkg import _apply_insight_confidence_multiplier
    
    # Test with different confidence levels
    base_score = 0.5
    
    # Low confidence should reduce score
    low_conf_score = _apply_insight_confidence_multiplier(base_score, 0.5)
    assert low_conf_score < base_score, f"Low confidence should reduce score, got {low_conf_score}"
    
    # High confidence should increase score
    high_conf_score = _apply_insight_confidence_multiplier(base_score, 0.9)
    assert high_conf_score > base_score, f"High confidence should increase score, got {high_conf_score}"
    
    # Test capping at 1.0
    capped_score = _apply_insight_confidence_multiplier(0.9, 0.9)
    assert capped_score <= 1.0, f"Score should be capped at 1.0, got {capped_score}"
    
    print(f"✅ Confidence multiplier: low={low_conf_score:.3f}, high={high_conf_score:.3f}, capped={capped_score:.3f}")

def run_all_tests():
    """Run all Insight V2 tests"""
    print("🚀 Starting Insight V2 Test Suite\n")
    
    try:
        test_insight_v2_config()
        print()
        
        test_insight_content_v2_evidence_detection()
        print()
        
        test_insight_content_v2_saturating_combiner()
        print()
        
        test_insight_content_v2_genre_specific()
        print()
        
        test_insight_content_v2_penalties()
        print()
        
        test_viral_moment_detector_v2_evidence_requirements()
        print()
        
        test_viral_moment_detector_v2_window_expansion()
        print()
        
        test_viral_moment_detector_v2_long_no_evidence_rejection()
        print()
        
        test_viral_moment_detector_v2_confidence_scoring()
        print()
        
        test_insight_v2_vs_v1_comparison()
        print()
        
        test_insight_confidence_multiplier()
        print()
        
        print("🎉 All Insight V2 tests passed!")
        print("\nKey improvements verified:")
        print("- Evidence-based detection (contrast, numbers, comparisons, imperatives)")
        print("- Saturating combiner for multiple evidence types")
        print("- Clause-aware window expansion")
        print("- Meaningful confidence scoring (0.5-0.9)")
        print("- Short insight flexibility (≤12s)")
        print("- Hedge and filler penalties")
        print("- Confidence-based score multiplier")
        print("- Genre-specific pattern support")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
