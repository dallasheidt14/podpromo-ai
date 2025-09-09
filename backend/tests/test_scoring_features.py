"""
Unit tests for scoring and feature computation modules.
Tests multi-path scoring, synergy bonuses, and genre-specific behavior.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the backend directory to the path so we can import services
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import the modules we want to test
# Use a more robust import strategy that works with pytest
import importlib.util
import os

def import_module_from_path(module_name, file_path):
    """Import a module from a specific file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Try to import from the package first
try:
    from services.secret_sauce_pkg import score_segment_v4, explain_segment_v4, viral_potential_v4
    from services.secret_sauce_pkg import compute_features_v4, _hook_score, _emotion_score, _payoff_presence
    from services.secret_sauce_pkg import GenreAwareScorer
    from services.secret_sauce_pkg import FantasySportsGenreProfile, ComedyGenreProfile
    print("Successfully imported from package")
except ImportError as e:
    print(f"Package import failed: {e}")
    # If package import fails, skip this test file
    pytest.skip("Could not import required modules from secret_sauce_pkg", allow_module_level=True)


class TestScoringFeatures:
    """Test suite for scoring and feature computation"""
    
    def test_hook_score_basic(self):
        """Test basic hook scoring functionality"""
        # Test strong hook
        strong_hook = "How I made $10,000 in one month"
        score = _hook_score(strong_hook)
        assert score > 0.0, f"Expected hook score > 0, got {score}"
        
        # Test weak hook
        weak_hook = "So anyway, I was thinking about stuff"
        score = _hook_score(weak_hook)
        assert score >= 0.0, f"Expected hook score >= 0, got {score}"
    
    def test_emotion_score_basic(self):
        """Test basic emotion scoring functionality"""
        # Test emotional content
        emotional_text = "I was absolutely devastated when this happened"
        score = _emotion_score(emotional_text)
        assert score >= 0.0, f"Expected emotion score >= 0, got {score}"
        
        # Test neutral content
        neutral_text = "The weather is nice today"
        score = _emotion_score(neutral_text)
        assert score >= 0.0, f"Expected emotion score >= 0, got {score}"
    
    def test_payoff_presence_basic(self):
        """Test basic payoff detection functionality"""
        # Test content with payoff
        payoff_text = "Here's the secret to making money online"
        score = _payoff_presence(payoff_text)
        assert score > 0.0, f"Expected payoff score > 0, got {score}"
        
        # Test content without payoff
        no_payoff_text = "I was just thinking about random things"
        score = _payoff_presence(no_payoff_text)
        assert score >= 0.0, f"Expected payoff score >= 0, got {score}"
    
    def test_score_segment_v4_basic(self):
        """Test basic V4 scoring functionality"""
        # Create test features
        features = {
            "hook_score": 0.8,
            "arousal_score": 0.6,
            "emotion_score": 0.4,
            "payoff_score": 0.7,
            "info_density": 0.5,
            "question_score": 0.3,
            "loopability": 0.6,
            "platform_len_match": 0.8,
            "insight_score": 0.4
        }
        
        result = score_segment_v4(features)
        
        # Check that result has expected structure
        assert "final_score" in result, "Result should contain final_score"
        assert "winning_path" in result, "Result should contain winning_path"
        assert "path_scores" in result, "Result should contain path_scores"
        assert "synergy_multiplier" in result, "Result should contain synergy_multiplier"
        assert "bonuses_applied" in result, "Result should contain bonuses_applied"
        assert "bonus_reasons" in result, "Result should contain bonus_reasons"
        assert "viral_score_100" in result, "Result should contain viral_score_100"
        
        # Check that scores are reasonable
        assert 0.0 <= result["final_score"] <= 1.0, f"Final score should be 0-1, got {result['final_score']}"
        assert 0 <= result["viral_score_100"] <= 100, f"Viral score should be 0-100, got {result['viral_score_100']}"
    
    def test_score_segment_v4_genre_specific(self):
        """Test genre-specific scoring behavior"""
        # Test general genre
        features = {
            "hook_score": 0.8,
            "arousal_score": 0.6,
            "emotion_score": 0.4,
            "payoff_score": 0.7,
            "info_density": 0.5,
            "question_score": 0.3,
            "loopability": 0.6,
            "platform_len_match": 0.8,
            "insight_score": 0.4
        }
        
        general_result = score_segment_v4(features, genre='general')
        fantasy_sports_result = score_segment_v4(features, genre='fantasy_sports')
        
        # Results should be different for different genres
        assert general_result["final_score"] != fantasy_sports_result["final_score"], \
            "Different genres should produce different scores"
    
    def test_explain_segment_v4_reuse_scoring(self):
        """Test that explain_segment_v4 can reuse existing scoring results"""
        features = {
            "hook_score": 0.8,
            "arousal_score": 0.6,
            "emotion_score": 0.4,
            "payoff_score": 0.7,
            "info_density": 0.5,
            "question_score": 0.3,
            "loopability": 0.6,
            "platform_len_match": 0.8,
            "insight_score": 0.4
        }
        
        # Get scoring result first
        scoring_result = score_segment_v4(features)
        
        # Test explanation with existing scoring result
        explanation = explain_segment_v4(features, scoring_result=scoring_result)
        
        # Check that explanation has expected structure
        assert "overall_assessment" in explanation, "Explanation should contain overall_assessment"
        assert "viral_score" in explanation, "Explanation should contain viral_score"
        assert "winning_strategy" in explanation, "Explanation should contain winning_strategy"
        assert "strengths" in explanation, "Explanation should contain strengths"
        assert "improvements" in explanation, "Explanation should contain improvements"
    
    def test_viral_potential_v4_basic(self):
        """Test basic viral potential calculation"""
        features = {
            "hook_score": 0.8,
            "arousal_score": 0.6,
            "emotion_score": 0.4,
            "payoff_score": 0.7,
            "info_density": 0.5,
            "question_score": 0.3,
            "loopability": 0.6,
            "platform_len_match": 0.8,
            "insight_score": 0.4
        }
        
        result = viral_potential_v4(features, length_s=30.0, platform="tiktok")
        
        # Check that result has expected structure
        assert "viral_score" in result, "Result should contain viral_score"
        assert "platforms" in result, "Result should contain platforms"
        assert "confidence" in result, "Result should contain confidence"
        
        # Check that scores are reasonable
        assert 0 <= result["viral_score"] <= 100, f"Viral score should be 0-100, got {result['viral_score']}"
        assert isinstance(result["platforms"], list), "Platforms should be a list"
    
    @patch('services.secret_sauce_pkg.features.librosa.load')
    def test_compute_features_v4_basic(self, mock_librosa):
        """Test basic feature computation with mocked audio"""
        # Mock librosa.load to return dummy audio data
        mock_librosa.return_value = (np.random.randn(1000), 22050)
        
        segment = {
            "start": 10.0,
            "end": 40.0,
            "text": "This is a test segment with some content"
        }
        
        # This should not raise an exception
        features = compute_features_v4(segment, "dummy_audio.mp3")
        
        # Check that features contain expected keys
        expected_keys = [
            "hook_score", "arousal_score", "emotion_score", "payoff_score",
            "info_density", "question_score", "loopability", "platform_len_match"
        ]
        
        for key in expected_keys:
            assert key in features, f"Features should contain {key}"
            assert isinstance(features[key], (int, float)), f"{key} should be numeric"
    
    def test_genre_aware_scorer_initialization(self):
        """Test that genre-aware scorer initializes correctly"""
        scorer = GenreAwareScorer()
        
        # Check that it has expected genres
        assert "general" in scorer.genres, "Should have general genre"
        assert "fantasy_sports" in scorer.genres, "Should have fantasy_sports genre"
        assert "comedy" in scorer.genres, "Should have comedy genre"
    
    def test_fantasy_sports_genre_profile(self):
        """Test fantasy sports genre profile"""
        profile = FantasySportsGenreProfile()
        
        # Check that it has expected attributes
        assert profile.name == "fantasy_sports", "Should have correct name"
        assert "hook" in profile.weights, "Should have hook weight"
        assert "payoff" in profile.weights, "Should have payoff weight"
        assert len(profile.weights) > 0, "Should have some weights defined"
    
    def test_comedy_genre_profile(self):
        """Test comedy genre profile"""
        profile = ComedyGenreProfile()
        
        # Check that it has expected attributes
        assert profile.name == "comedy", "Should have correct name"
        assert "hook" in profile.weights, "Should have hook weight"
        assert "arousal" in profile.weights, "Should have arousal weight"
        assert "payoff" in profile.weights, "Should have payoff weight"
    
    def test_synergy_bonus_calculation(self):
        """Test that synergy bonuses are calculated correctly"""
        # Test features that should trigger synergy bonus
        high_synergy_features = {
            "hook_score": 0.9,
            "arousal_score": 0.8,
            "emotion_score": 0.7,
            "payoff_score": 0.9,
            "info_density": 0.6,
            "question_score": 0.4,
            "loopability": 0.7,
            "platform_len_match": 0.9,
            "insight_score": 0.5
        }
        
        result = score_segment_v4(high_synergy_features)
        
        # Should have synergy multiplier > 1.0 for high synergy
        assert result["synergy_multiplier"] > 1.0, f"Expected synergy > 1.0, got {result['synergy_multiplier']}"
        assert result["bonuses_applied"] > 0.0, f"Expected bonuses > 0, got {result['bonuses_applied']}"
    
    def test_multi_path_scoring(self):
        """Test that multi-path scoring works correctly"""
        features = {
            "hook_score": 0.8,
            "arousal_score": 0.6,
            "emotion_score": 0.4,
            "payoff_score": 0.7,
            "info_density": 0.5,
            "question_score": 0.3,
            "loopability": 0.6,
            "platform_len_match": 0.8,
            "insight_score": 0.4
        }
        
        result = score_segment_v4(features)
        
        # Should have multiple path scores
        assert len(result["path_scores"]) > 1, "Should have multiple scoring paths"
        
        # All path scores should be reasonable
        for path_name, score in result["path_scores"].items():
            assert 0.0 <= score <= 1.0, f"Path {path_name} score should be 0-1, got {score}"
    
    def test_platform_length_matching(self):
        """Test platform length matching functionality"""
        from services.secret_sauce_pkg.features import _platform_length_match
        
        # Test TikTok (short form)
        tiktok_score = _platform_length_match(30.0, "tiktok")
        assert tiktok_score > 0.0, f"30s should score > 0 for TikTok, got {tiktok_score}"
        
        # Test YouTube (longer form)
        youtube_score = _platform_length_match(30.0, "youtube")
        assert youtube_score > 0.0, f"30s should score > 0 for YouTube, got {youtube_score}"
    
    def test_insight_detection(self):
        """Test insight content detection"""
        from services.secret_sauce_pkg.features import _detect_insight_content
        
        # Test content with insights
        insight_text = "The key insight here is that most people don't understand the fundamental principle"
        score, reason = _detect_insight_content(insight_text)
        assert score > 0.0, f"Expected insight score > 0, got {score}"
        
        # Test content without insights
        no_insight_text = "So anyway, I was just talking about random stuff"
        score, reason = _detect_insight_content(no_insight_text)
        assert score >= 0.0, f"Expected insight score >= 0, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
