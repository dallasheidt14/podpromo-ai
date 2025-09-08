from typing import Dict
import re

# Genre-Aware Scoring System
class GenreProfile:
    """Base class for genre-specific scoring profiles"""
    def __init__(self):
        self.name = "general"
        self.weights = {
            'hook': 0.45, 'arousal': 0.25, 'payoff': 0.15, 
            'info_density': 0.10, 'loopability': 0.05
        }
        self.viral_triggers = []
        self.hook_patterns = []
        self.payoff_patterns = []
        self.optimal_length = (15, 45)  # seconds
        self.min_viral_score = 50  # Genre-specific threshold
        self.viral_threshold = 50  # Genre-specific viral threshold
        self.penalty_config = {
            'context_penalty': 0.25,  # "clearly", "obviously" etc.
            'repetition_penalty': 0.15,
            'filler_penalty': 0.10
        }
        
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect genre-specific viral patterns in text"""
        return {}
        
    def adjust_features(self, features: Dict) -> Dict:
        """Apply genre-specific adjustments to raw features"""
        return features
        
    def get_scoring_paths(self, features: Dict) -> Dict:
        """Get genre-specific scoring paths"""
        f = features
        # Default 4-path system
        path_a = (0.45 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                  0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
                  0.05 * f.get("loopability", 0.0))
        
        path_b = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                  0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                  0.05 * f.get("arousal_score", 0.0))
        
        path_c = (0.40 * f.get("arousal_score", 0.0) + 0.25 * f.get("emotion_score", 0.0) + 
                  0.20 * f.get("hook_score", 0.0) + 0.10 * f.get("loopability", 0.0) + 
                  0.05 * f.get("question_score", 0.0))
        
        path_d = (0.25 * f.get("question_score", 0.0) + 0.25 * f.get("info_density", 0.0) + 
                  0.20 * f.get("payoff_score", 0.0) + 0.20 * f.get("hook_score", 0.0) + 
                  0.10 * f.get("arousal_score", 0.0))
        
        return {"hook": path_a, "payoff": path_b, "energy": path_c, "structured": path_d}
    
    def apply_quality_gate(self, features: Dict) -> float:
        """Genre-specific quality gate logic with insight/question awareness"""
        payoff = features.get("payoff_score", 0.0)
        hook = features.get("hook_score", 0.0)
        insight = features.get("insight_score", 0.0)
        qlist = features.get("question_score", 0.0)
        
        # Allow insight/question to substitute for payoff
        if payoff >= 0.3 or insight >= 0.7 or qlist >= 0.5:
            return 1.0  # No penalty
        if hook >= 0.4:
            return 0.95  # Very mild penalty
        return 0.85  # Mild penalty
    
    def score(self, features: Dict) -> Dict:
        """Score features using genre-specific weights"""
        score = 0.0
        for feature, weight in self.weights.items():
            score += weight * features.get(feature, 0.0)
        return {"score": score, "weighted_features": self.weights}

class GenreAwareScorer:
    """Main genre-aware scoring system"""
    def __init__(self):
        from .genre_profiles import FantasySportsGenreProfile, ComedyGenreProfile
        
        self.genres = {
            'fantasy_sports': FantasySportsGenreProfile(),
            'sports': FantasySportsGenreProfile(),  # Use same profile for now
            'comedy': ComedyGenreProfile(),
            'general': GenreProfile()  # Fallback/default
        }
    
    def auto_detect_genre(self, text: str) -> str:
        """Auto-detect genre from content keywords with weighted scoring"""
        genre_keywords = {
            'fantasy_sports': ['fantasy', 'waiver', 'roster', 'matchup', 'dfs', 'sleeper', 'start', 'sit', 'league', 'draft', 'trade', 'pickup', 'chalk', 'gpp', 'value', 'buy low', 'sell high'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'league', 'season', 'touchdown', 'yards', 'points', 'stats', 'performance'],
            'comedy': ['funny', 'joke', 'laugh', 'hilarious', 'story', 'happened', 'crazy', 'weird', 'unbelievable', 'off guard', 'caught me', 'actually happened'],
            'general': []
        }
        
        # Weighted scoring system
        scores = {}
        for genre, keywords in genre_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Multi-word keywords get higher weight
                    if ' ' in keyword:
                        score += 2
                    else:
                        score += 1
            scores[genre] = score
        
        # Return highest scoring genre or 'general'
        best_genre = max(scores, key=scores.get)
        if scores[best_genre] >= 2:  # Lower threshold for better detection
            return best_genre
        return 'general'
    
    def detect_genre_with_confidence(self, text: str) -> tuple[str, float]:
        """Returns (genre, confidence_score) with improved detection"""
        genre_keywords = {
            'fantasy_sports': ['fantasy', 'waiver', 'roster', 'matchup', 'dfs', 'sleeper', 'start', 'sit', 'league', 'draft', 'trade', 'pickup', 'chalk', 'gpp', 'value', 'buy low', 'sell high'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'league', 'season', 'touchdown', 'yards', 'points', 'stats', 'performance'],
            'comedy': ['funny', 'joke', 'laugh', 'hilarious', 'story', 'happened', 'crazy', 'weird', 'unbelievable', 'off guard', 'caught me', 'actually happened'],
            'general': []
        }
        
        # Calculate genre scores
        scores = {}
        for genre, keywords in genre_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Multi-word keywords get higher weight
                    if ' ' in keyword:
                        score += 2
                    else:
                        score += 1
            scores[genre] = score
        
        if not scores or max(scores.values()) == 0:
            return 'general', 0.0
        
        # Find best genre
        best_genre = max(scores, key=scores.get)
        best_score = scores[best_genre]
        
        # Calculate confidence based on score distribution
        if len(scores) > 1:
            sorted_scores = sorted(scores.values(), reverse=True)
            second_best = sorted_scores[1]
            confidence = (best_score - second_best) / best_score if best_score > 0 else 0.0
        else:
            confidence = 1.0 if best_score > 0 else 0.0
        
        # Normalize confidence to 0-1 range
        confidence = min(max(confidence, 0.0), 1.0)
        
        return best_genre, confidence
    
    def score_for_genre(self, features: Dict, genre: str = 'general') -> Dict:
        """Score content using genre-specific weights and patterns"""
        profile = self.genres.get(genre, self.genres['general'])
        return profile.score(features)
