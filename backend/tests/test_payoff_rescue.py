"""
Tests for the payoff rescue system.
"""

import pytest
from services.payoff_rescue import apply_payoff_rescue, _eligible_for_payoff_rescue, _bump_amount, is_length_agnostic_mode


class TestPayoffRescue:
    """Test payoff rescue functionality."""
    
    def test_eligible_for_payoff_rescue_with_payoff_label(self):
        """Test eligibility with explicit payoff label."""
        features = {
            "payoff_label": "cta",
            "insight_conf": 0.7,
            "payoff_score": 0.8
        }
        assert _eligible_for_payoff_rescue(features) == True
    
    def test_eligible_for_payoff_rescue_with_insight(self):
        """Test eligibility with strong insight score."""
        features = {
            "insight_score": 0.25,
            "insight_conf": 0.65
        }
        assert _eligible_for_payoff_rescue(features) == True
    
    def test_not_eligible_low_confidence(self):
        """Test not eligible with low confidence."""
        features = {
            "payoff_label": "cta",
            "insight_conf": 0.5,  # Below threshold
            "payoff_score": 0.8
        }
        assert _eligible_for_payoff_rescue(features) == False
    
    def test_not_eligible_weak_insight(self):
        """Test not eligible with weak insight."""
        features = {
            "insight_score": 0.15,  # Below threshold
            "insight_conf": 0.7
        }
        assert _eligible_for_payoff_rescue(features) == False
    
    def test_bump_amount_high_strength(self):
        """Test high bump for strong payoff/insight."""
        features = {
            "payoff_score": 0.4,
            "insight_score": 0.3
        }
        assert _bump_amount(features) == 0.05  # BUMP_HIGH
    
    def test_bump_amount_low_strength(self):
        """Test low bump for moderate payoff/insight."""
        features = {
            "payoff_score": 0.2,
            "insight_score": 0.1
        }
        assert _bump_amount(features) == 0.03  # BUMP_LOW
    
    def test_apply_payoff_rescue_basic(self):
        """Test basic payoff rescue application."""
        items = [
            {
                "calibrated": 0.8,
                "features": {"payoff_label": "cta", "insight_conf": 0.7, "payoff_score": 0.4},
                "clip": {"id": "clip1", "start": 0, "end": 10}
            },
            {
                "calibrated": 0.6,
                "features": {"insight_score": 0.1, "insight_conf": 0.5},
                "clip": {"id": "clip2", "start": 0, "end": 15}
            }
        ]
        
        result = apply_payoff_rescue(items)
        
        # First item should be rescued
        assert "calibrated_rescued" in result[0]
        assert result[0]["calibrated_rescued"] > result[0]["calibrated"]
        assert result[0]["rescue_reason"] == "payoff_rescue(+5.0%)"
        
        # Second item should not be rescued
        assert result[1]["calibrated_rescued"] == result[1]["calibrated"]
        assert result[1]["rescue_reason"] is None
    
    def test_apply_payoff_rescue_no_leapfrog(self):
        """Test that rescue doesn't allow unfair leapfrogging."""
        items = [
            {
                "calibrated": 0.9,
                "features": {"insight_score": 0.1, "insight_conf": 0.5},
                "clip": {"id": "clip1", "start": 0, "end": 10}
            },
            {
                "calibrated": 0.6,
                "features": {"payoff_label": "cta", "insight_conf": 0.7, "payoff_score": 0.4},
                "clip": {"id": "clip2", "start": 0, "end": 15}
            }
        ]
        
        result = apply_payoff_rescue(items)
        
        # First item should remain on top
        assert result[0]["clip"]["id"] == "clip1"
        # Second item can be rescued but not leapfrog
        assert result[1]["calibrated_rescued"] <= result[0]["calibrated_rescued"] - 0.02  # TOP_MARGIN
    
    def test_apply_payoff_rescue_hard_cap(self):
        """Test that rescue respects hard cap."""
        items = [
            {
                "calibrated": 0.1,  # Very low score
                "features": {"payoff_label": "cta", "insight_conf": 0.7, "payoff_score": 0.4},
                "clip": {"id": "clip1", "start": 0, "end": 10}
            }
        ]
        
        result = apply_payoff_rescue(items)
        
        # Should be rescued but capped at +0.08
        assert result[0]["calibrated_rescued"] <= 0.1 + 0.08
        assert result[0]["calibrated_rescued"] > result[0]["calibrated"]
    
    def test_is_length_agnostic_mode(self, monkeypatch):
        """Test length-agnostic mode detection."""
        # Test enabled
        monkeypatch.setenv("PLATFORM_PROTECT", "false")
        monkeypatch.setenv("PL_V2_WEIGHT", "0.0")
        monkeypatch.setenv("LENGTH_CAP_ENABLED", "false")
        assert is_length_agnostic_mode() == True
        
        # Test disabled (default)
        monkeypatch.setenv("PLATFORM_PROTECT", "true")
        monkeypatch.setenv("PL_V2_WEIGHT", "0.5")
        monkeypatch.setenv("LENGTH_CAP_ENABLED", "true")
        assert is_length_agnostic_mode() == False
        
        # Test partial (should be disabled)
        monkeypatch.setenv("PLATFORM_PROTECT", "false")
        monkeypatch.setenv("PL_V2_WEIGHT", "0.5")  # Not 0.0
        monkeypatch.setenv("LENGTH_CAP_ENABLED", "false")
        assert is_length_agnostic_mode() == False
