"""
Tests for length cap safety to prevent regressions.
"""

import pytest
from services.clip_score import apply_length_bucket_cap


class TestLengthCapSafety:
    """Test that length cap never returns empty list."""
    
    def test_length_cap_never_empties(self):
        """Test that limiter never returns empty list."""
        finals = [
            {"id": "a", "duration": 47.5, "score": 0.6},
            {"id": "b", "duration": 16.0, "score": 0.58},
            {"id": "c", "duration": 27.5, "score": 0.57},
        ]
        out = apply_length_bucket_cap(finals, length_agnostic=True)
        assert out, "Limiter must not return empty"
        assert any(c["duration"] > 30 for c in out), "Should keep at least one long or leave list unmodified"
    
    def test_length_cap_preserves_short_clips(self):
        """Test that short clips are preserved."""
        finals = [
            {"id": "a", "duration": 8.0, "score": 0.7},
            {"id": "b", "duration": 12.0, "score": 0.6},
        ]
        out = apply_length_bucket_cap(finals, length_agnostic=True)
        assert len(out) == 2, "Should preserve all short clips"
        assert all(c["duration"] < 13 for c in out), "All clips should be short"
    
    def test_length_cap_keeps_best_long(self):
        """Test that only the best long clip is kept when length-agnostic."""
        finals = [
            {"id": "a", "duration": 45.0, "score": 0.8},  # Best long
            {"id": "b", "duration": 50.0, "score": 0.6},  # Worse long
            {"id": "c", "duration": 55.0, "score": 0.5},  # Worst long
            {"id": "d", "duration": 15.0, "score": 0.7},  # Short
        ]
        out = apply_length_bucket_cap(finals, length_agnostic=True)
        assert len(out) == 2, "Should keep 1 long + 1 short"
        long_clips = [c for c in out if c["duration"] > 30]
        assert len(long_clips) == 1, "Should keep exactly 1 long clip"
        assert long_clips[0]["id"] == "a", "Should keep the best long clip"
    
    def test_length_cap_ignores_when_not_agnostic(self):
        """Test that length cap is ignored when not length-agnostic."""
        finals = [
            {"id": "a", "duration": 45.0, "score": 0.6},
            {"id": "b", "duration": 50.0, "score": 0.8},  # Better score
            {"id": "c", "duration": 55.0, "score": 0.5},
        ]
        out = apply_length_bucket_cap(finals, length_agnostic=False)
        assert len(out) == 3, "Should keep all clips when not length-agnostic"
        assert all(c["id"] in ["a", "b", "c"] for c in out), "Should preserve all original clips"
    
    def test_length_cap_handles_empty_input(self):
        """Test that empty input returns empty output."""
        out = apply_length_bucket_cap([], length_agnostic=True)
        assert out == [], "Empty input should return empty output"
    
    def test_length_cap_handles_missing_duration(self):
        """Test that missing duration field is handled gracefully."""
        finals = [
            {"id": "a", "score": 0.6},  # No duration
            {"id": "b", "duration": 15.0, "score": 0.7},
        ]
        out = apply_length_bucket_cap(finals, length_agnostic=True)
        assert len(out) == 2, "Should handle missing duration gracefully"
    
    def test_length_cap_handles_missing_score(self):
        """Test that missing score field is handled gracefully."""
        finals = [
            {"id": "a", "duration": 45.0},  # No score
            {"id": "b", "duration": 50.0, "score": 0.8},
        ]
        out = apply_length_bucket_cap(finals, length_agnostic=True)
        assert len(out) == 1, "Should keep the clip with score"
        assert out[0]["id"] == "b", "Should keep the clip with the higher score"
