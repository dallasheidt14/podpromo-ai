"""
Tests for EOS improvements and finish ratio enhancements.
"""

import pytest
from services.util import (
    detect_sentence_endings_from_words,
    unify_eos_markers,
    extend_to_natural_end,
    calculate_finish_confidence,
    finish_threshold_for
)


class TestEOSImprovements:
    """Test enhanced EOS detection and boundary refinement."""
    
    def test_detect_sentence_endings_from_words(self):
        """Test punctuation-based EOS detection."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world.", "start": 0.5, "end": 1.0},
            {"word": "How", "start": 1.2, "end": 1.5},
            {"word": "are", "start": 1.5, "end": 1.7},
            {"word": "you?", "start": 1.7, "end": 2.0},
            {"word": "I", "start": 2.2, "end": 2.3},
            {"word": "am", "start": 2.3, "end": 2.5},
            {"word": "fine.", "start": 2.5, "end": 3.0},
        ]
        
        eos = detect_sentence_endings_from_words(words)
        
        # Should detect sentence endings at periods and question marks
        assert 1.0 in eos  # "world."
        assert 2.0 in eos  # "you?"
        assert 3.0 in eos  # "fine."
        assert len(eos) == 3
    
    def test_unify_eos_markers(self):
        """Test EOS marker unification."""
        existing = [1.0, 2.5, 4.0]
        words_based = [1.1, 2.6, 3.8]
        
        unified = unify_eos_markers(existing, words_based, tol=0.25)
        
        # Should prefer words-based markers when within tolerance
        assert len(unified) == 3
        assert 1.1 in unified  # Words-based preferred
        assert 2.6 in unified  # Words-based preferred
        assert 3.8 in unified  # Words-based preferred
    
    def test_extend_to_natural_end(self):
        """Test smart boundary extension."""
        words = [
            {"word": "This", "start": 0.0, "end": 0.3},
            {"word": "is", "start": 0.3, "end": 0.5},
            {"word": "a", "start": 0.5, "end": 0.6},
            {"word": "test.", "start": 0.6, "end": 1.0},
            {"word": "Another", "start": 1.2, "end": 1.5},
            {"word": "sentence.", "start": 1.5, "end": 2.0},
        ]
        
        clip = {"start": 0.0, "end": 0.8, "text": "This is a test"}
        
        extended = extend_to_natural_end(clip, words, max_extend_sec=3.0)
        
        # Should extend to the period at 1.0
        assert extended["end"] == 1.0
        assert extended["extended"] is True
        assert extended["extension_delta"] == 0.2
    
    def test_calculate_finish_confidence(self):
        """Test finish confidence calculation."""
        # High confidence: ends with period
        clip1 = {"text": "This is a complete sentence.", "end": 5.0}
        conf1 = calculate_finish_confidence(clip1)
        assert conf1 >= 0.6
        
        # Medium confidence: discourse closer
        clip2 = {"text": "And that's why we do this", "end": 5.0}
        conf2 = calculate_finish_confidence(clip2)
        assert conf2 >= 0.4
        
        # Low confidence: no punctuation
        clip3 = {"text": "This is incomplete", "end": 5.0}
        conf3 = calculate_finish_confidence(clip3)
        assert conf3 < 0.3
    
    def test_finish_threshold_for(self):
        """Test adaptive threshold calculation."""
        # Conversational content should have lower threshold
        threshold1 = finish_threshold_for("general", {"conversational_ratio": 0.8})
        assert threshold1 < 0.60
        
        # Technical content should have higher threshold
        threshold2 = finish_threshold_for("technical", {"conversational_ratio": 0.3})
        assert threshold2 > 0.60
        
        # Interview format should have lower threshold
        threshold3 = finish_threshold_for("general", {"interview_format": True})
        assert threshold3 < 0.60


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_eos_detection_integration(self):
        """Test that EOS detection works with real word data."""
        # Sample words from a real episode
        words = [
            {"word": "So", "start": 0.0, "end": 0.2},
            {"word": "today", "start": 0.2, "end": 0.5},
            {"word": "we're", "start": 0.5, "end": 0.7},
            {"word": "going", "start": 0.7, "end": 0.9},
            {"word": "to", "start": 0.9, "end": 1.0},
            {"word": "talk", "start": 1.0, "end": 1.2},
            {"word": "about", "start": 1.2, "end": 1.4},
            {"word": "AI.", "start": 1.4, "end": 1.7},
            {"word": "This", "start": 1.9, "end": 2.0},
            {"word": "is", "start": 2.0, "end": 2.1},
            {"word": "important.", "start": 2.1, "end": 2.5},
        ]
        
        eos = detect_sentence_endings_from_words(words)
        
        # Should detect sentence endings
        assert len(eos) >= 1
        assert 1.7 in eos  # "AI."
        assert 2.5 in eos  # "important."
    
    def test_boundary_extension_integration(self):
        """Test boundary extension with realistic clip data."""
        words = [
            {"word": "The", "start": 0.0, "end": 0.2},
            {"word": "key", "start": 0.2, "end": 0.4},
            {"word": "point", "start": 0.4, "end": 0.6},
            {"word": "is", "start": 0.6, "end": 0.7},
            {"word": "this.", "start": 0.7, "end": 1.0},
            {"word": "Now", "start": 1.2, "end": 1.4},
            {"word": "let's", "start": 1.4, "end": 1.6},
            {"word": "continue.", "start": 1.6, "end": 2.0},
        ]
        
        clip = {
            "start": 0.0,
            "end": 0.8,  # Ends before the period
            "text": "The key point is this",
            "id": "test_clip_1"
        }
        
        extended = extend_to_natural_end(clip, words, max_extend_sec=2.0)
        
        # Should extend to the period
        assert extended["end"] == 1.0
        assert extended["extended"] is True
        assert extended["extension_delta"] == 0.2


if __name__ == "__main__":
    pytest.main([__file__])
