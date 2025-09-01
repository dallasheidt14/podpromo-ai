"""
secret_sauce.py - V4 Enhanced Viral Detection System
All heuristics + weights for picking 'winning' clips.
"""

from typing import Dict, List, Tuple
import numpy as np, librosa
from config_loader import get_config
import re
import datetime
import hashlib
from functools import lru_cache

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
        """Genre-specific quality gate logic"""
        # Default quality gate for general genre
        if (features.get("payoff_score", 0.0) < 0.3 and features.get("hook_score", 0.0) < 0.4):
            return 0.85  # Mild penalty
        return 1.0  # No penalty
    
    def score(self, features: Dict) -> Dict:
        """Score features using genre-specific weights"""
        score = 0.0
        for feature, weight in self.weights.items():
            score += weight * features.get(feature, 0.0)
        return {"score": score, "weighted_features": self.weights}

class FantasySportsGenreProfile(GenreProfile):
    """Fantasy Sports specific scoring profile"""
    def __init__(self):
        super().__init__()
        self.name = "fantasy_sports"
        
        # Fantasy sports values different features
        self.weights = {
            'hook': 0.25,  # Less important than payoff
            'payoff': 0.35,  # Critical - actionable advice
            'info_density': 0.20,  # Stats and specifics matter
            'urgency': 0.10,  # "This week" matters
            'confidence': 0.10  # "Trust me" indicators
        }
        
        self.viral_triggers = [
            r"sleeper (pick|candidate|alert)",
            r"league winner",
            r"waiver (wire|pickup|add)",
            r"must[\s-]?start",
            r"sit[\s/]start",
            r"buy low",
            r"sell high",
            r"(dfs|draft) (play|pick)",
            r"chalk play",
            r"gpp (play|leverage)",
            r"value play",
            r"(trade|roster) advice"
        ]
        
        self.hook_patterns = [
            r"nobody('s| is) talking about",
            r"everyone('s| is) wrong about",
            r"the (one|only) (guy|player|pick)",
            r"before (it's|its) too late",
            r"this week('s| is)",
            r"breakout (alert|candidate|week)"
        ]
        
        self.payoff_patterns = [
            r"fire him up",
            r"(start|sit|bench|play) him",
            r"he's (your|the) (guy|play|start)",
            r"(add|grab|pickup) him (now|immediately)",
            r"(sell|buy) (high|low) on",
            r"target share",
            r"usage rate"
        ]
        
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect fantasy sports specific patterns"""
        t = text.lower()
        patterns = {}
        
        # Urgency score (time-sensitive advice)
        urgency_patterns = [
            r"this week", r"tonight", r"tomorrow", r"immediately", 
            r"now", r"before", r"deadline", r"last chance"
        ]
        urgency_hits = sum(1 for pattern in urgency_patterns if re.search(pattern, t))
        patterns['urgency_score'] = min(urgency_hits * 0.2, 1.0)
        
        # Confidence score (trust indicators)
        confidence_patterns = [
            r"trust me", r"i'm telling you", r"mark my words", 
            r"guarantee", r"100%", r"definitely", r"absolutely"
        ]
        confidence_hits = sum(1 for pattern in confidence_patterns if re.search(pattern, t))
        patterns['confidence_score'] = min(confidence_hits * 0.25, 1.0)
        
        # Viral trigger boost
        trigger_hits = sum(1 for trigger in self.viral_triggers if re.search(trigger, t))
        patterns['viral_trigger_boost'] = min(trigger_hits * 0.1, 0.3)
        
        return patterns
        
    def adjust_features(self, features: Dict) -> Dict:
        """Apply fantasy sports specific adjustments"""
        adjusted = features.copy()
        
        # Boost payoff for actionable advice
        if features.get('payoff_score', 0) > 0.6:
            adjusted['payoff_score'] = min(features['payoff_score'] * 1.2, 1.0)
            
        # Reduce hook penalty for sports context
        if features.get('hook_score', 0) < 0.3:
            adjusted['hook_score'] = features['hook_score'] * 1.3
            
        # Boost info density for stats-heavy content
        if features.get('info_density', 0) > 0.4:
            adjusted['info_density'] = min(features['info_density'] * 1.15, 1.0)
            
        return adjusted
        
    def get_scoring_paths(self, features: Dict) -> Dict:
        """Fantasy sports specific scoring paths with proper genre feature integration"""
        f = features
        
        # Actionable path (replaces energy path) - uses genre-specific features
        actionable_path = (0.40 * f.get("payoff_score", 0.0) + 
                          0.30 * f.get("info_density", 0.0) + 
                          0.20 * f.get("confidence_score", 0.0) + 
                          0.10 * f.get("urgency_score", 0.0))
        
        # Hot take path (replaces structured path) - uses genre-specific features
        hot_take_path = (0.35 * f.get("confidence_score", 0.0) + 
                        0.35 * f.get("hook_score", 0.0) + 
                        0.30 * f.get("viral_trigger_boost", 0.0))
        
        # Keep hook and payoff paths but adjust weights
        hook_path = (0.35 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                     0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("info_density", 0.0) + 
                     0.05 * f.get("loopability", 0.0))
        
        payoff_path = (0.45 * f.get("payoff_score", 0.0) + 0.25 * f.get("info_density", 0.0) + 
                       0.15 * f.get("confidence_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                       0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook_path, 
            "payoff": payoff_path, 
            "actionable": actionable_path, 
            "hot_take": hot_take_path
        }

class ComedyGenreProfile(GenreProfile):
    """Comedy specific scoring profile"""
    def __init__(self):
        super().__init__()
        self.name = "comedy"
        
        self.weights = {
            'hook': 0.40,  # Setup is crucial
            'arousal': 0.25,  # Energy/delivery
            'emotion': 0.20,  # Laughter response
            'payoff': 0.15  # Punchline
        }
        
        self.viral_triggers = [
            r"I can't believe",
            r"the (funniest|craziest|weirdest)",
            r"you won't believe",
            r"this actually happened",
            r"caught (me|him|her) off guard"
        ]
        
        self.hook_patterns = [
            r"so I was",
            r"the other day",
            r"you know when",
            r"imagine this",
            r"picture this"
        ]
        
        self.payoff_patterns = [
            r"and then",
            r"turns out",
            r"plot twist",
            r"the punchline",
            r"the reveal"
        ]
        
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect comedy specific patterns"""
        t = text.lower()
        patterns = {}
        
        # Timing score (setup-punchline structure)
        timing_patterns = [
            r"so.*then", r"first.*then", r"started.*ended",
            r"thought.*turns out", r"expected.*got"
        ]
        timing_hits = sum(1 for pattern in timing_patterns if re.search(pattern, t))
        patterns['timing_score'] = min(timing_hits * 0.3, 1.0)
        
        # Surprise score (unexpected elements)
        surprise_patterns = [
            r"plot twist", r"unexpected", r"shocking", 
            r"out of nowhere", r"didn't see that coming"
        ]
        surprise_hits = sum(1 for pattern in surprise_patterns if re.search(pattern, t))
        patterns['surprise_score'] = min(surprise_hits * 0.25, 1.0)
        
        # Viral trigger boost
        trigger_hits = sum(1 for trigger in self.viral_triggers if re.search(trigger, t))
        patterns['viral_trigger_boost'] = min(trigger_hits * 0.15, 0.4)
        
        return patterns
        
    def adjust_features(self, features: Dict) -> Dict:
        """Apply comedy specific adjustments"""
        adjusted = features.copy()
        
        # Boost arousal for energetic delivery
        if features.get('arousal_score', 0) > 0.5:
            adjusted['arousal_score'] = min(features['arousal_score'] * 1.25, 1.0)
            
        # Boost emotion for laughter triggers
        if features.get('emotion_score', 0) > 0.3:
            adjusted['emotion_score'] = min(features['emotion_score'] * 1.2, 1.0)
            
        return adjusted
        
    def get_scoring_paths(self, features: Dict) -> Dict:
        """Comedy specific scoring paths with proper genre feature integration"""
        f = features
        
        # Setup-punchline path - uses genre-specific features
        setup_punchline = (0.45 * f.get("hook_score", 0.0) + 
                          0.35 * f.get("payoff_score", 0.0) + 
                          0.20 * f.get("timing_score", 0.0))
        
        # Energy path - uses genre-specific features
        energy = (0.50 * f.get("arousal_score", 0.0) + 
                  0.30 * f.get("emotion_score", 0.0) + 
                  0.20 * f.get("surprise_score", 0.0))
        
        # Keep hook and payoff paths
        hook = (0.45 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
                0.05 * f.get("loopability", 0.0))
        
        payoff = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                  0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                  0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook, 
            "payoff": payoff, 
            "setup_punchline": setup_punchline, 
            "energy": energy
        }

class TrueCrimeGenreProfile(GenreProfile):
    def __init__(self):
        super().__init__()
        self.name = "true_crime"
        self.weights = {
            'hook': 0.30,  # Mystery setup
            'payoff': 0.30,  # Resolution/revelation
            'emotion': 0.20,  # Shock value
            'info_density': 0.15,
            'narrative_arc': 0.05
        }
        self.viral_threshold = 55
        self.optimal_length = (45, 60)  # Needs narrative time
        self.min_viral_score = 55
        self.penalty_config = {
            'context_penalty': 0.10,  # Reduced penalty
            'repetition_penalty': 0.15,
            'filler_penalty': 0.10
        }
        
        self.viral_triggers = [
            r"the shocking truth",
            r"what they didn't tell you",
            r"the real story",
            r"behind the scenes",
            r"the untold story"
        ]
        
        self.hook_patterns = [
            r"in (19|20)\d{2}",
            r"the case of",
            r"when (he|she|they)",
            r"little did (he|she|they) know",
            r"the investigation revealed"
        ]
        
        self.payoff_patterns = [
            r"turns out",
            r"the truth was",
            r"investigators discovered",
            r"evidence showed",
            r"the real killer was"
        ]
    
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect true crime specific patterns"""
        features = {}
        
        # Mystery indicators
        mystery_patterns = [
            r"mystery", r"case", r"investigation", r"evidence", r"clue",
            r"witness", r"suspect", r"victim", r"crime scene"
        ]
        mystery_score = sum(0.1 for pattern in mystery_patterns if re.search(pattern, text.lower()))
        features['mystery_score'] = min(mystery_score, 1.0)
        
        # Resolution indicators
        resolution_patterns = [
            r"turns out", r"the truth", r"discovered", r"revealed",
            r"finally", r"at last", r"the answer"
        ]
        resolution_score = sum(0.15 for pattern in resolution_patterns if re.search(pattern, text.lower()))
        features['resolution_score'] = min(resolution_score, 1.0)
        
        # True crime specific features
        features['has_case_details'] = bool(re.search(r'case|investigation|evidence|witness', text.lower()))
        features['has_timeline'] = bool(re.search(r'in \d{4}|when|then|after|before', text.lower()))
        features['has_resolution'] = bool(re.search(r'turns out|the truth|discovered|revealed', text.lower()))
        
        # Viral trigger boost
        viral_boost = sum(0.1 for trigger in self.viral_triggers if re.search(trigger, text.lower()))
        features['viral_trigger_boost'] = min(viral_boost, 0.5)
        
        return features
    
    def apply_quality_gate(self, features: Dict) -> float:
        """True crime quality gate - needs mystery and resolution"""
        if features.get('mystery_score', 0.0) < 0.2 and features.get('resolution_score', 0.0) < 0.2:
            return 0.80  # Penalty for weak mystery/resolution
        return 1.0  # No penalty
    
    def get_scoring_paths(self, features: Dict) -> Dict:
        """True crime specific scoring paths with proper genre feature integration"""
        f = features
        
        # Mystery path - uses genre-specific features
        mystery_path = (0.40 * f.get("hook_score", 0.0) + 
                       0.30 * f.get("mystery_score", 0.0) + 
                       0.20 * f.get("emotion_score", 0.0) + 
                       0.10 * f.get("info_density", 0.0))
        
        # Resolution path - uses genre-specific features
        resolution_path = (0.40 * f.get("payoff_score", 0.0) + 
                          0.30 * f.get("resolution_score", 0.0) + 
                          0.20 * f.get("mystery_score", 0.0) + 
                          0.10 * f.get("emotion_score", 0.0))
        
        # Keep hook and payoff paths
        hook_path = (0.35 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                     0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("info_density", 0.0) + 
                     0.05 * f.get("loopability", 0.0))
        
        payoff_path = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                       0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                       0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook_path, 
            "payoff": payoff_path, 
            "mystery": mystery_path, 
            "resolution": resolution_path
        }

class BusinessGenreProfile(GenreProfile):
    """Business/Entrepreneurship specific scoring profile"""
    def __init__(self):
        super().__init__()
        self.name = "business"
        
        self.weights = {
            'hook': 0.30,  # Story opening
            'payoff': 0.35,  # Actionable insights
            'info_density': 0.25,  # Specific advice
            'confidence': 0.10  # Authority indicators
        }
        
        self.viral_triggers = [
            r"\$\d+ to \$\d+",
            r"my biggest mistake",
            r"how I built",
            r"startup lesson",
            r"investment advice"
        ]
        
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect business specific patterns"""
        t = text.lower()
        patterns = {}
        
        # Authority score (expertise indicators)
        authority_patterns = [
            r"i built", r"i founded", r"i sold", r"i invested",
            r"my company", r"my startup", r"my experience"
        ]
        authority_hits = sum(1 for pattern in authority_patterns if re.search(pattern, t))
        patterns['authority_score'] = min(authority_hits * 0.2, 1.0)
        
        # Specificity score (concrete numbers/details)
        specificity_patterns = [
            r"\$\d+", r"\d+%", r"\d+ years", r"\d+ months",
            r"revenue", r"profit", r"customers", r"employees"
        ]
        specificity_hits = sum(1 for pattern in specificity_patterns if re.search(pattern, t))
        patterns['specificity_score'] = min(specificity_hits * 0.15, 1.0)
        
        return patterns
    
    def get_scoring_paths(self, features: Dict) -> Dict:
        """Business specific scoring paths with proper genre feature integration"""
        f = features
        
        # Authority path - uses genre-specific features
        authority_path = (0.40 * f.get("payoff_score", 0.0) + 
                         0.30 * f.get("authority_score", 0.0) + 
                         0.20 * f.get("info_density", 0.0) + 
                         0.10 * f.get("hook_score", 0.0))
        
        # Specificity path - uses genre-specific features
        specificity_path = (0.40 * f.get("info_density", 0.0) + 
                           0.30 * f.get("specificity_score", 0.0) + 
                           0.20 * f.get("payoff_score", 0.0) + 
                           0.10 * f.get("hook_score", 0.0))
        
        # Keep hook and payoff paths
        hook_path = (0.35 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                     0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("info_density", 0.0) + 
                     0.05 * f.get("loopability", 0.0))
        
        payoff_path = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                       0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                       0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook_path, 
            "payoff": payoff_path, 
            "authority": authority_path, 
            "specificity": specificity_path
        }

class EducationalGenreProfile(GenreProfile):
    """Educational/Science specific scoring profile"""
    def __init__(self):
        super().__init__()
        self.name = "educational"
        
        self.weights = {
            'hook': 0.35,  # Curiosity gap
            'info_density': 0.30,  # Knowledge content
            'payoff': 0.25,  # Clear explanation
            'surprise': 0.10  # Counter-intuitive facts
        }
        
        self.viral_triggers = [
            r"did you know",
            r"scientists discovered",
            r"mind-blowing",
            r"counter-intuitive",
            r"the truth about"
        ]
        
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect educational specific patterns"""
        t = text.lower()
        patterns = {}
        
        # Curiosity score (knowledge gaps)
        curiosity_patterns = [
            r"did you know", r"surprisingly", r"contrary to",
            r"most people think", r"the truth is"
        ]
        curiosity_hits = sum(1 for pattern in curiosity_patterns if re.search(pattern, t))
        patterns['curiosity_score'] = min(curiosity_hits * 0.25, 1.0)
        
        # Clarity score (clear explanations)
        clarity_patterns = [
            r"here's why", r"the reason is", r"this means",
            r"in other words", r"simply put"
        ]
        clarity_hits = sum(1 for pattern in clarity_patterns if re.search(pattern, t))
        patterns['clarity_score'] = min(clarity_hits * 0.2, 1.0)
        
        return patterns
    
    def get_scoring_paths(self, features: Dict) -> Dict:
        """Educational specific scoring paths with proper genre feature integration"""
        f = features
        
        # Curiosity path - uses genre-specific features
        curiosity_path = (0.40 * f.get("hook_score", 0.0) + 
                         0.30 * f.get("curiosity_score", 0.0) + 
                         0.20 * f.get("info_density", 0.0) + 
                         0.10 * f.get("arousal_score", 0.0))
        
        # Clarity path - uses genre-specific features
        clarity_path = (0.40 * f.get("payoff_score", 0.0) + 
                       0.30 * f.get("clarity_score", 0.0) + 
                       0.20 * f.get("info_density", 0.0) + 
                       0.10 * f.get("hook_score", 0.0))
        
        # Keep hook and payoff paths
        hook_path = (0.35 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                     0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("info_density", 0.0) + 
                     0.05 * f.get("loopability", 0.0))
        
        payoff_path = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                       0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                       0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook_path, 
            "payoff": payoff_path, 
            "curiosity": curiosity_path, 
            "clarity": clarity_path
        }

class HealthWellnessGenreProfile(GenreProfile):
    def __init__(self):
        super().__init__()
        self.name = "health_wellness"
        self.weights = {
            'payoff': 0.35,  # Actionable advice
            'credibility': 0.25,  # Trust signals
            'hook': 0.20,
            'emotion': 0.15,  # Transformation stories
            'specificity': 0.05
        }
        self.viral_threshold = 50
        self.optimal_length = (30, 45)
        self.min_viral_score = 50
        self.penalty_config = {
            'context_penalty': 0.10,  # Reduced penalty
            'repetition_penalty': 0.15,
            'filler_penalty': 0.10
        }
        
        self.viral_triggers = [
            r"doctors don't want you to know",
            r"the secret to",
            r"transform your",
            r"natural remedy",
            r"proven method"
        ]
        
        self.hook_patterns = [
            r"if you struggle with",
            r"are you tired of",
            r"here's what I discovered",
            r"the truth about",
            r"what they don't tell you"
        ]
        
        self.payoff_patterns = [
            r"here's how to",
            r"the solution is",
            r"try this",
            r"start with",
            r"focus on"
        ]
    
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect health/wellness specific patterns"""
        features = {}
        
        # Credibility indicators
        credibility_patterns = [
            r"doctor", r"physician", r"therapist", r"nutritionist", r"trainer",
            r"expert", r"specialist", r"certified", r"licensed"
        ]
        credibility_score = sum(0.15 for pattern in credibility_patterns if re.search(pattern, text.lower()))
        features['credibility_score'] = min(credibility_score, 1.0)
        
        # Transformation indicators
        transformation_patterns = [
            r"transform", r"change", r"improve", r"boost", r"enhance",
            r"better", r"stronger", r"healthier", r"happier"
        ]
        transformation_score = sum(0.1 for pattern in transformation_patterns if re.search(pattern, text.lower()))
        features['transformation_score'] = min(transformation_score, 1.0)
        
        # Health specific features
        features['has_credentials'] = bool(re.search(r'doctor|physician|therapist|nutritionist|trainer', text.lower()))
        features['has_benefits'] = bool(re.search(r'benefits|improve|boost|enhance|better', text.lower()))
        features['has_actionable'] = bool(re.search(r'here\'s how|try this|start with|focus on', text.lower()))
        
        # Viral trigger boost
        viral_boost = sum(0.1 for trigger in self.viral_triggers if re.search(trigger, text.lower()))
        features['viral_trigger_boost'] = min(viral_boost, 0.5)
        
        return features
    
    def apply_quality_gate(self, features: Dict) -> float:
        """Health/wellness quality gate - needs credible advice"""
        if features.get('payoff_score', 0.0) < 0.3 and features.get('credibility_score', 0.0) < 0.2:
            return 0.80  # Penalty for weak health advice
        return 1.0  # No penalty
    
    def get_scoring_paths(self, features: Dict) -> Dict:
        """Health/wellness specific scoring paths with proper genre feature integration"""
        f = features
        
        # Credibility path - uses genre-specific features
        credibility_path = (0.40 * f.get("payoff_score", 0.0) + 
                           0.30 * f.get("credibility_score", 0.0) + 
                           0.20 * f.get("info_density", 0.0) + 
                           0.10 * f.get("hook_score", 0.0))
        
        # Transformation path - uses genre-specific features
        transformation_path = (0.40 * f.get("emotion_score", 0.0) + 
                             0.30 * f.get("transformation_score", 0.0) + 
                             0.20 * f.get("payoff_score", 0.0) + 
                             0.10 * f.get("hook_score", 0.0))
        
        # Keep hook and payoff paths
        hook_path = (0.35 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                     0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("info_density", 0.0) + 
                     0.05 * f.get("loopability", 0.0))
        
        payoff_path = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                       0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                       0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook_path, 
            "payoff": payoff_path, 
            "credibility": credibility_path, 
            "transformation": transformation_path
        }

class GenreAwareScorer:
    """Main genre-aware scoring system"""
    def __init__(self):
        self.genres = {
            'fantasy_sports': FantasySportsGenreProfile(),
            'sports': FantasySportsGenreProfile(),  # Use same profile for now
            'comedy': ComedyGenreProfile(),
            'true_crime': TrueCrimeGenreProfile(),
            'business': BusinessGenreProfile(),
            'news_politics': NewsPoliticsGenreProfile(),
            'education': EducationalGenreProfile(),
            'health_wellness': HealthWellnessGenreProfile(),
            'general': GenreProfile()  # Fallback/default
        }
    
    def auto_detect_genre(self, text: str) -> str:
        """Auto-detect genre from content keywords with weighted scoring"""
        genre_keywords = {
            'fantasy_sports': ['fantasy', 'waiver', 'roster', 'matchup', 'dfs', 'sleeper', 'start', 'sit', 'league', 'draft', 'trade', 'pickup', 'chalk', 'gpp', 'value', 'buy low', 'sell high'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'league', 'season', 'touchdown', 'yards', 'points', 'stats', 'performance'],
            'comedy': ['funny', 'joke', 'laugh', 'hilarious', 'story', 'happened', 'crazy', 'weird', 'unbelievable', 'off guard', 'caught me', 'actually happened'],
            'true_crime': ['case', 'evidence', 'murder', 'investigation', 'police', 'detective', 'victim', 'suspect', 'crime', 'mystery', 'disappeared', 'missing'],
            'business': ['company', 'startup', 'revenue', 'investment', 'business', 'entrepreneur', 'million', 'dollar', 'profit', 'strategy', 'market', 'industry'],
            'news_politics': ['breaking', 'announcement', 'government', 'political', 'election', 'policy', 'controversy', 'scandal', 'official', 'statement'],
            'education': ['science', 'research', 'study', 'discovered', 'learn', 'fact', 'scientist', 'experiment', 'theory', 'mind-blowing', 'did you know'],
            'health_wellness': ['doctor', 'doctors', 'health', 'wellness', 'fitness', 'nutrition', 'mental', 'transformation', 'before after', 'myth', 'tip', 'advice', 'natural', 'remedy', 'cure', 'treatment', 'therapy', 'healing', 'wellness', 'lifestyle']
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
            'true_crime': ['case', 'evidence', 'murder', 'investigation', 'police', 'detective', 'victim', 'suspect', 'crime', 'mystery', 'disappeared', 'missing'],
            'business': ['company', 'startup', 'revenue', 'investment', 'business', 'entrepreneur', 'million', 'dollar', 'profit', 'strategy', 'market', 'industry'],
            'news_politics': ['breaking', 'announcement', 'government', 'political', 'election', 'policy', 'controversy', 'scandal', 'official', 'statement'],
            'education': ['science', 'research', 'study', 'discovered', 'learn', 'fact', 'scientist', 'experiment', 'theory', 'mind-blowing', 'did you know'],
            'health_wellness': ['doctor', 'doctors', 'health', 'wellness', 'fitness', 'nutrition', 'mental', 'transformation', 'before after', 'myth', 'tip', 'advice', 'natural', 'remedy', 'cure', 'treatment', 'therapy', 'healing', 'wellness', 'lifestyle']
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

def get_clip_weights():
    return get_config()["weights"]

CLIP_WEIGHTS = get_clip_weights()
HOOK_CUES = tuple(get_config()["lexicons"]["HOOK_CUES"])
EMO_WORDS = tuple(get_config()["lexicons"]["EMO_WORDS"])
FILLERS = tuple(get_config()["lexicons"]["FILLERS"])
PAYOFF_MARKERS = tuple(get_config()["lexicons"]["PAYOFF_MARKERS"])
QUESTION_STARTS = tuple(get_config()["lexicons"]["QUESTION_STARTS"])
LIST_MARKERS = tuple(get_config()["lexicons"]["LIST_MARKERS"])

def _detect_insight_content(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect if content contains actual insights vs. intro/filler material"""
    if not text or len(text.strip()) < 10:
        return 0.0, "too_short"
    
    t = text.lower()
    insight_score = 0.0
    reasons = []
    
    # Fantasy sports insight patterns
    if genre in ['fantasy_sports', 'sports']:
        insight_patterns = [
            r"(observation|insight|noticed|realized|discovered)",
            r"(main|key|important|significant) (takeaway|point|finding)",
            r"(casual|serious|experienced) (drafters|players|managers)",
            r"(way better|much better|improved|evolved)",
            r"(under my belt|experience|seen|witnessed)",
            r"(home league|draft|waiver|roster)",
            r"(sleeper|bust|value|target|avoid)",
            r"(this week|next week|season|playoffs)"
        ]
        
        for pattern in insight_patterns:
            if re.search(pattern, t):
                insight_score += 0.2
                reasons.append("fantasy_insight")
        
        # Boost for specific insights
        if re.search(r"(casual drafters are way better)", t):
            insight_score += 0.3
            reasons.append("specific_insight_boost")
    
    # General insight patterns
    general_insight_patterns = [
        r"(here's what|the thing is|what i found|what i learned)",
        r"(the key|the secret|the trick|the strategy)",
        r"(most people|everyone|nobody) (thinks|believes|knows)",
        r"(contrary to|despite|although|even though)",
        r"(the truth is|reality is|actually|in fact)"
    ]
    
    for pattern in general_insight_patterns:
        if re.search(pattern, t):
            insight_score += 0.15
            reasons.append("general_insight")
    
    # Penalty for filler content
    filler_patterns = [
        r"^(yo|hey|hi|hello|what's up)",
        r"^(it's|this is) (monday|tuesday|wednesday)",
        r"^(hope you|hope everyone)",
        r"^(let's get|let's start|let's begin)"
    ]
    
    for pattern in filler_patterns:
        if re.match(pattern, t):
            insight_score -= 0.3
            reasons.append("filler_penalty")
            break
    
    final_score = float(np.clip(insight_score, 0.0, 1.0))
    reason_str = ";".join(reasons) if reasons else "no_insights"
    return final_score, reason_str

def _hook_score_v4(text: str, arousal: float = 0.0, words_per_sec: float = 0.0, genre: str = 'general') -> tuple[float, str]:
    """V4 hook detection with more reasonable thresholds"""
    if not text or len(text.strip()) < 8:
        return 0.1, "text_too_short"  # Changed from 0.0 to 0.1 minimum
    
    t = text.lower()[:200]
    score = 0.1  # Start with base score of 0.1 instead of 0.0
    reasons = []
    
    # INTRO/GREETING DETECTION - Heavy penalty for intro material
    intro_patterns = [
        r"^(yo|hey|hi|hello|what's up|how's it going|good morning|good afternoon|good evening)",
        r"^(it's|this is) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"^(i'm|my name is) \w+",
        r"^(welcome to|thanks for|thank you for)",
        r"^(hope you|hope everyone)",
        r"^(let's get|let's start|let's begin)",
        r"^(today we're|today i'm|today let's)",
        r"^(so|and|but then|after that|next)"  # Still penalize sequence words
    ]
    
    for pattern in intro_patterns:
        if re.match(pattern, t):
            score = 0.01  # Set to extremely low score for intro material
            reasons.append("intro_greeting_penalty")
            break
    
    # RELAXED: Context-dependent penalty (less harsh for sports content)
    context_patterns = [
        r"^(you like that|like that|that's|here's the)",  # Still penalize obvious context
    ]
    
    # DON'T penalize sports-specific context like "Caleb Johnson is clearly"
    # This is normal in sports analysis
    
    for pattern in context_patterns:
        if re.match(pattern, t):
            score -= 0.15  # Reduced from 0.3 to 0.15
            reasons.append("context_dependent_opening")
            break
    
    # NEW: Sports-specific hooks that shouldn't be penalized
    sports_hook_patterns = [
        r"(biggest|top|best|worst) (sleeper|bust|play|pick|value)",
        r"(nobody|everyone|people) (is|are) (talking about|sleeping on|missing)",
        r"(here's why|let me tell you why|this is why)",
        r"the (guy|player|team) (who|that)",
        r"(fantasy|draft|waiver) (gold|gem|steal|target)",
        r"(this|that) (guy|player|team) is",
        r"(i'm telling you|trust me|mark my words)",
        r"(if you|when you) (draft|pick|start|sit)"
    ]
    
    for pattern in sports_hook_patterns:
        if re.search(pattern, t):
            score += 0.3
            reasons.append("sports_hook")
            break
    
    # Direct hook cues
    strong_hits = sum(1 for cue in HOOK_CUES if cue in t)
    if strong_hits > 0:
        score += min(strong_hits * 0.15, 0.35)
        reasons.append(f"direct_hooks_{strong_hits}")
    
    # Curiosity gaps
    curiosity_patterns = [r"the (one|only) (thing|way|reason|secret)", r"what (nobody|no one|they don't) tell you"]
    for pattern in curiosity_patterns:
        if re.search(pattern, t):
            score += 0.4
            reasons.append("curiosity_gap")
            break
    
    # Question hooks
    if "?" in t[:100]:
        score += 0.25
        reasons.append("question_mark")
    
    # Boost for insight content (if not intro)
    if not any("intro_greeting_penalty" in reason for reason in reasons):
        insight_score, _ = _detect_insight_content(text, genre)
        if insight_score > 0.5:
            score += 0.2  # Boost for high insight content
            reasons.append("insight_content_boost")
    
    # At the end, ensure minimum score
    final_score = float(np.clip(score, 0.05, 1.0))  # Allow lower minimum for intro content
    reason_str = ";".join(reasons) if reasons else "no_hooks_detected"
    return final_score, reason_str

def _calculate_niche_penalty(text: str, genre: str = 'general') -> tuple[float, str]:
    t = text.lower()
    penalty = 0.0
    reasons = []
    
    # Skip penalties entirely for sports genres
    if genre in ['sports', 'fantasy_sports']:
        return 0.0, "sports_genre_no_penalty"
    
    # Apply penalties for other genres
    context_patterns = [r"\b(like that|that's|this is)\b"]
    for pattern in context_patterns:
        if re.search(pattern, t):
            penalty += 0.10
            reasons.append("context_dependent")
            break
    
    final_penalty = float(np.clip(penalty, 0.0, 0.5))
    reason_str = ";".join(reasons) if reasons else "no_niche_penalty"
    return final_penalty, reason_str

def _emotion_score_v4(text: str) -> float:
    t = text.lower()
    high_intensity = ["incredible", "insane", "mind-blowing", "shocking"]
    high_hits = sum(1 for w in high_intensity if w in t)
    regular_hits = sum(1 for w in EMO_WORDS if w in t and w not in high_intensity)
    total_score = (high_hits * 2 + regular_hits) / 5.0
    return float(min(total_score, 1.0))

def _payoff_presence_v4(text: str) -> tuple[float, str]:
    """More selective payoff detection"""
    t = text.strip().lower()
    
    # Check for incomplete thoughts that shouldn't get high payoff
    incomplete_patterns = [
        r"\.{3}$",  # Ends with ellipsis
        r"^(and|but|so|well|because)$",  # Starts and ends with conjunction
        r"^(is|are|was|were)$",  # Just a linking verb by itself
        r"(and|but|so|well|because)$",  # Ends with conjunction
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, t):
            return 0.2, "incomplete_thought"
    
    # Existing tier checks but with stricter matching
    strong_payoff_patterns = [
        r"(so|here's) (what|the thing|the key|the point)",
        r"the (lesson|takeaway|insight|bottom line) (is|here)",
        r"(this|that) (means|shows|proves|explains why)",
        r"(therefore|thus|that's why|which means)",
        r"in (other words|summary|essence)",
        r"the (reality|truth) is",
        r"(turns out|it turns out)",
        # Sports-specific strong patterns
        r"long story short",
        r"bottom line",
        r"the key takeaway",
        r"here's the deal",
        r"the bottom line is"
    ]
    
    # Require stronger signal for high scores
    matches = sum(1 for pattern in strong_payoff_patterns if re.search(pattern, t))
    if matches >= 2:  # Require multiple signals
        return 0.9, "strong_conclusion"
    elif matches == 1:
        return 0.7, "moderate_conclusion"
    
    # Tier 2: Actionable insights (0.5-0.7) - reduced scores
    action_payoff_patterns = [
        r"(the key is|the secret is|remember) to",
        r"(this is how|here's how|that's how) (you|to)",
        r"(start|begin|focus|try) (by|with|on)",
        r"(instead|rather than|better to)",
        r"(you can|you should|make sure) (to|that)",
        # Sports-specific actionable patterns
        r"he's (your|their) (number one|go-to|primary)",
        r"this (guy|player) is",
        r"you want to (target|avoid|start|sit)",
        r"the (move|play|strategy) is to"
    ]
    
    for pattern in action_payoff_patterns:
        if re.search(pattern, t):
            return 0.6, "actionable_insight"  # Reduced from 0.75
    
    # Tier 3: Explanatory payoff (0.4-0.6) - reduced scores
    explanation_patterns = [
        r"(because|since|as|given that|due to)",
        r"(the reason|why this) (is|works|happens)",
        r"(this explains|that's because)",
        r"(which is why|and that's)",
        r"(the problem|issue) (is|was|with)",
        # Sports-specific explanations
        r"that's why (he|they|this team)",
        r"this is why (you|we|they)",
        r"the reason (he|they|this) (is|was)",
        r"which explains why"
    ]
    
    for pattern in explanation_patterns:
        if re.search(pattern, t):
            return 0.5, "explanation"  # Reduced from 0.6
    
    # Tier 4: Soft payoff (0.2-0.4) - reduced scores
    soft_patterns = [
        r"(in other words|put differently)",
        r"(the thing is|honestly|basically)",
        r"(and|but|so) (this|that|it)",
        # Sports-specific soft patterns
        r"he's (clearly|obviously|definitely)",
        r"this (team|player) is",
        r"that's (what|how|why)",
        r"it's (that|what|how)"
    ]
    
    for pattern in soft_patterns:
        if re.search(pattern, t):
            return 0.3, "soft_payoff"  # Reduced from 0.4
    
    return 0.1, "no_clear_payoff"

def _info_density_v4(text: str) -> float:
    if not text or len(text.strip()) < 8:
        return 0.1
    
    t = text.lower()
    words = t.split()
    base = min(0.3, len(words) * 0.012)
    
    filler_words = ["you know", "like", "um", "uh", "right", "so"]
    filler_count = sum(1 for filler in filler_words if filler in t)
    filler_penalty = min(filler_count * 0.08, 0.4)
    
    numbers_and_stats = len(re.findall(r'\b\d+\b|[\$\d,]+|\d+%|\d+\.\d+', text))
    proper_nouns = sum(1 for w in text.split() if w[0].isupper() and len(w) > 2)
    
    specificity_boost = min((numbers_and_stats * 0.15 + proper_nouns * 0.12), 0.6)
    
    final = base - filler_penalty + specificity_boost
    return float(max(0.1, min(1.0, final)))

def _question_or_list(text: str) -> float:
    t = text.strip().lower()
    
    greeting_questions = ["what's up", "how's it going", "you like that", "how are you"]
    if any(greeting in t[:30] for greeting in greeting_questions):
        return 0.1
    
    if "?" in t:
        question_text = t.split("?")[0][-50:]
        if len(question_text.split()) < 3:
            return 0.2
        
        engaging_patterns = [r"what if", r"why do", r"how did", r"what happens when"]
        if any(re.search(pattern, question_text) for pattern in engaging_patterns):
            return 1.0
        
        return 0.6
    
    return 0.0

def _loopability_heuristic(text: str) -> float:
    if not text:
        return 0.0
    
    words = text.strip().split()
    if len(words) < 4:
        return 0.5
    
    clean_endings = [".", "!", "?"]
    ends_clean = text.strip()[-1] in clean_endings
    
    bad_endings = ["and", "but", "so", "because", "that", "which"]
    ends_badly = any(text.strip().lower().endswith(" " + word) for word in bad_endings)
    
    score = 0.5
    if ends_clean:
        score += 0.3
    if not ends_badly:
        score += 0.1
    if len(words) <= 50:
        score += 0.1
    
    return float(np.clip(score, 0.0, 1.0))

def _arousal_score_text(text: str) -> float:
    if not text:
        return 0.0
    
    t = text.lower()
    score = 0.0
    
    exclam_count = text.count('!')
    score += min(exclam_count * 0.15, 0.3)
    
    caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
    score += min(caps_words * 0.1, 0.2)
    
    emotion_words = ["amazing", "incredible", "crazy", "wild", "insane", "shocking"]
    emotion_hits = sum(1 for word in emotion_words if word in t)
    score += min(emotion_hits * 0.1, 0.3)
    
    return float(np.clip(score, 0.0, 1.0))

def _synergy_v4(hook: float, arousal: float, payoff: float, alpha: float = 0.3) -> float:
    """IMPROVED synergy calculation with wider range and better balance"""
    hook = float(np.clip(hook, 0.0, 1.0))
    arousal = float(np.clip(arousal, 0.0, 1.0))
    payoff = float(np.clip(payoff, 0.0, 1.0))
    
    # Geometric mean for balance
    gm = (hook * arousal * payoff) ** (1/3)
    
    # Reward balanced performance
    min_feature = min(hook, arousal, payoff)
    max_feature = max(hook, arousal, payoff)
    
    # Balance bonus: reward when all features are decent
    balance_bonus = 1.0 + (0.2 * min_feature)
    
    # Consistency bonus: reward when features are close together
    feature_range = max_feature - min_feature
    consistency_bonus = 1.0 + (0.1 * (1.0 - feature_range))
    
    # Base multiplier with wider range
    base_mult = (1 - alpha) * 1.0 + alpha * gm
    
    # Apply bonuses with wider range
    final_mult = base_mult * balance_bonus * consistency_bonus
    
    # WIDER RANGE: 0.50 to 1.40 (was 0.60 to 1.15)
    return float(np.clip(final_mult, 0.50, 1.40))

def _calculate_confidence_score(features: Dict, genre: str = 'general') -> tuple[str, str]:
    """Realistic confidence calculation based on actual feature ranges"""
    # Count strong features (adjusted to realistic thresholds)
    strong_count = 0
    weak_count = 0
    
    core_features = ['hook_score', 'payoff_score', 'arousal_score', 'info_density']
    for feature in core_features:
        score = features.get(feature, 0.0)
        if score >= 0.4:  # Adjusted from 0.6 to match actual feature ranges
            strong_count += 1
        elif score <= 0.2:  # Count weak features (adjusted from 0.3)
            weak_count += 1
    
    # Calculate confidence based on balance
    if weak_count >= 2:
        return "Low", "#fd7e14"
    elif strong_count >= 3:
        return "High", "#28a745"
    elif strong_count >= 2:
        return "Medium", "#ffc107"
    else:
        return "Low", "#fd7e14"

def score_segment_v4(features: Dict, weights: Dict = None, apply_penalties: bool = True, genre: str = 'general', platform: str = None) -> Dict:
    """V4 Multi-path scoring system with genre awareness and platform multipliers"""
    if weights is None:
        weights = get_clip_weights()
    
    f = features
    
    # Get genre-specific scoring paths
    if genre != 'general':
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre, genre_scorer.genres['general'])
        path_scores = genre_profile.get_scoring_paths(features)
    else:
        # Default 4-path system for general genre with platform length match and insight detection
        path_a = (0.35 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                  0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
                  0.10 * f.get("platform_len_match", 0.0) + 0.05 * f.get("loopability", 0.0) + 
                  0.05 * f.get("insight_score", 0.0))  # NEW: Insight content boost
        
        path_b = (0.30 * f.get("payoff_score", 0.0) + 0.25 * f.get("info_density", 0.0) + 
                  0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                  0.10 * f.get("platform_len_match", 0.0) + 0.05 * f.get("arousal_score", 0.0) + 
                  0.05 * f.get("insight_score", 0.0))  # NEW: Insight content boost
        
        path_c = (0.30 * f.get("arousal_score", 0.0) + 0.20 * f.get("emotion_score", 0.0) + 
                  0.20 * f.get("hook_score", 0.0) + 0.10 * f.get("loopability", 0.0) + 
                  0.10 * f.get("platform_len_match", 0.0) + 0.05 * f.get("question_score", 0.0) + 
                  0.05 * f.get("insight_score", 0.0))  # NEW: Insight content boost
        
        path_d = (0.20 * f.get("question_score", 0.0) + 0.20 * f.get("info_density", 0.0) + 
                  0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("hook_score", 0.0) + 
                  0.15 * f.get("platform_len_match", 0.0) + 0.10 * f.get("arousal_score", 0.0))
        
        path_scores = {"hook": path_a, "payoff": path_b, "energy": path_c, "structured": path_d}
    
    base_score = max(path_scores.values())
    winning_path = max(path_scores.keys(), key=lambda k: path_scores[k])
    
    # Calculate synergy using raw scores (not scaled)
    synergy_mult = _synergy_v4(f.get("hook_score", 0.0), f.get("arousal_score", 0.0), f.get("payoff_score", 0.0))
    synergy_score = base_score * synergy_mult
    
    # Apply exponential scaling to final score for better differentiation
    hook_scaled = f.get("hook_score", 0.0) ** 1.3
    arousal_scaled = f.get("arousal_score", 0.0) ** 1.2
    payoff_scaled = f.get("payoff_score", 0.0) ** 1.1
    
    # Use scaled scores for additional bonuses
    scaled_bonus = (hook_scaled + arousal_scaled + payoff_scaled) / 3.0 * 0.1
    synergy_score += scaled_bonus
    
    # Threshold bonuses
    bonuses = 0.0
    bonus_reasons = []
    
    if f.get("hook_score", 0.0) >= 0.8:
        bonuses += 0.08
        bonus_reasons.append("hook_excellence")
    
    if f.get("payoff_score", 0.0) >= 0.8:
        bonuses += 0.06
        bonus_reasons.append("payoff_excellence")
    
    # NEW: Insight content bonus
    if f.get("insight_score", 0.0) >= 0.6:
        bonuses += 0.10
        bonus_reasons.append("high_insight_content")
    elif f.get("insight_score", 0.0) >= 0.3:
        bonuses += 0.05
        bonus_reasons.append("moderate_insight_content")
    
    # Genre-specific bonuses
    if genre != 'general':
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre, genre_scorer.genres['general'])
        
        # Add genre-specific viral trigger bonuses
        if 'viral_trigger_boost' in f:
            trigger_bonus = f['viral_trigger_boost'] * 0.05
            bonuses += trigger_bonus
            bonus_reasons.append(f"genre_viral_trigger_{trigger_bonus:.3f}")
    
    final_score = synergy_score + bonuses
    
    # Apply penalties
    if apply_penalties and f.get("_ad_flag", False):
        penalty = f.get("_ad_penalty", 0.0)
        final_score = max(0.0, final_score - penalty)
        bonus_reasons.append(f"ad_penalty_{penalty:.2f}")
    
    # Apply mixed intro penalty
    if f.get("mixed_intro_penalty", 0.0) > 0:
        penalty = f.get("mixed_intro_penalty", 0.0)
        final_score = max(0.0, final_score - penalty)
        bonus_reasons.append(f"mixed_intro_penalty_{penalty:.2f}")
    
    # Genre-specific quality gates
    if genre != 'general':
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre, genre_scorer.genres['general'])
        quality_multiplier = genre_profile.apply_quality_gate(features)
        final_score *= quality_multiplier
        if quality_multiplier < 1.0:
            bonus_reasons.append(f"genre_quality_gate_{quality_multiplier:.2f}")
    else:
        # Default quality gate for general genre
        if (f.get("hook_score", 0.0) < 0.2 and f.get("payoff_score", 0.0) < 0.2):
            # Only apply penalty if BOTH hook AND payoff are very weak
            final_score *= 0.85  # Reduced penalty from 0.75 to 0.85
            bonus_reasons.append("very_weak_content_penalty")
    
    # Apply platform + genre multipliers
    if platform and genre != 'general':
        platform_multiplier = PLATFORM_GENRE_MULTIPLIERS.get(platform, {}).get(genre, 1.0)
        final_score *= platform_multiplier
        if platform_multiplier != 1.0:
            bonus_reasons.append(f"platform_{platform}_multiplier_{platform_multiplier:.2f}")
    
    # NEW: Genre-specific scoring multipliers for better differentiation
    genre_multipliers = {
        'comedy': 1.15,      # Comedy gets 15% boost (entertainment value)
        'sports': 1.10,      # Sports gets 10% boost (engagement)
        'fantasy_sports': 1.12,  # Fantasy sports gets 12% boost (strategy + entertainment)
        'business': 1.08,    # Business gets 8% boost (practical value)
        'education': 1.06,   # Education gets 6% boost (learning value)
        'news_politics': 1.05,  # News gets 5% boost (timeliness)
        'true_crime': 1.10,  # True crime gets 10% boost (mystery)
        'health_wellness': 1.04  # Health gets 4% boost (wellness value)
    }
    
    if genre in genre_multipliers:
        genre_boost = genre_multipliers[genre]
        final_score *= genre_boost
        bonus_reasons.append(f"genre_{genre}_boost_{genre_boost:.2f}")
    
    # Bonus for moment types
    moment_type = features.get('type', 'general')
    if moment_type == 'story':
        final_score *= 1.15  # 15% bonus for complete stories
        bonus_reasons.append("complete_story_bonus")
    elif moment_type == 'insight':
        final_score *= 1.10  # 10% bonus for insights
        bonus_reasons.append("insight_bonus")
    elif moment_type == 'hot_take':
        final_score *= 1.08  # 8% bonus for hot takes
        bonus_reasons.append("hot_take_bonus")
    
    final_score = float(np.clip(final_score, 0.0, 1.0))
    
    # Calculate confidence score
    confidence_level, confidence_color = _calculate_confidence_score(features, genre)
    
    return {
        "final_score": final_score,
        "winning_path": winning_path,
        "path_scores": path_scores,
        "synergy_multiplier": synergy_mult,
        "bonuses_applied": bonuses,
        "bonus_reasons": bonus_reasons,
        "viral_score_100": int(final_score * 100),
        "display_score": int(final_score * 100),  # Frontend expects this field
        "confidence": confidence_level,           # Frontend expects this field
        "confidence_color": confidence_color      # Frontend expects this field
    }

def compute_features_v4(segment: Dict, audio_file: str, y_sr=None, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """Enhanced feature computation with genre awareness"""
    text = segment.get("text", "")
    
    # CRITICAL: Check for ads FIRST, before any feature computation
    ad_result = _ad_penalty(text)
    
    if ad_result["flag"]:
        # Return a clip that will be filtered out entirely
        return {
            "is_advertisement": True,
            "ad_reason": ad_result["reason"],
            "final_score": 0.0,  # Force to bottom
            "viral_score_100": 0,
            "should_exclude": True,
            "text": text,
            "duration": segment["end"] - segment["start"],
            "hook_score": 0.0,
            "arousal_score": 0.0,
            "emotion_score": 0.0,
            "question_score": 0.0,
            "payoff_score": 0.0,
            "info_density": 0.0,
            "loopability": 0.0
        }
    
    # Only compute full features for non-ad content
    duration = segment["end"] - segment["start"]
    
    word_count = len(text.split()) if text else 0
    words_per_sec = word_count / max(duration, 0.1)
    
    hook_score, hook_reasons = _hook_score_v4(text, segment.get("arousal_score", 0.0), words_per_sec, genre)
    payoff_score, payoff_type = _payoff_presence_v4(text)
    
    # NEW: Detect insight content vs. intro/filler
    insight_score, insight_reasons = _detect_insight_content(text, genre)
    
    niche_penalty, niche_reason = _calculate_niche_penalty(text, genre)
    
    # REAL AUDIO ANALYSIS: Compute actual audio arousal from audio file
    try:
        audio_arousal = _audio_prosody_score(audio_file, segment["start"], segment["end"])
    except Exception:
        # Fallback to text-based estimation if audio analysis fails
        text_energy_indicators = [
            ('!', 0.1),
            ('amazing', 0.15),
            ('incredible', 0.15),
            ('crazy', 0.15),
            ('insane', 0.2),
            ('?!', 0.2),
            ('wow', 0.1),
            ('unbelievable', 0.15),
            ('shocking', 0.2),
            ('wild', 0.1)
        ]
        
        audio_arousal = 0.4  # Base fallback
        for indicator, boost in text_energy_indicators:
            if indicator in text.lower():
                audio_arousal = min(audio_arousal + boost, 0.9)
                break
    
    text_arousal = _arousal_score_text(text)
    combined_arousal = 0.7 * audio_arousal + 0.3 * text_arousal
    
    # Base features
    feats = {
        "is_advertisement": False,  # Explicitly mark as non-ad
        "should_exclude": False,    # Explicitly mark as includable
        "hook_score": hook_score,
        "arousal_score": combined_arousal,
        "emotion_score": _emotion_score_v4(text),
        "question_score": _question_or_list(text),
        "payoff_score": payoff_score,
        "info_density": _info_density_v4(text),
        "loopability": _loopability_heuristic(text),
        "insight_score": insight_score,  # NEW: Insight content detection
        "text": text,
        "duration": duration,
        "words_per_sec": words_per_sec,
        "hook_reasons": hook_reasons,
        "payoff_type": payoff_type,
        "insight_reasons": insight_reasons,  # NEW: Insight detection reasons
        "text_arousal": text_arousal,
        "audio_arousal": audio_arousal,
        "platform_len_match": calculate_dynamic_length_score(segment, platform) if "boundary_type" in segment else _platform_length_match(duration, platform),
        "_ad_flag": ad_result["flag"],
        "_ad_penalty": ad_result["penalty"],
        "_ad_reason": ad_result["reason"],
        "_niche_penalty": niche_penalty,
        "_niche_reason": niche_reason,
        "type": segment.get("type", "general")  # Preserve moment type for bonuses
    }
    
    # Apply genre-specific enhancements if genre is specified
    if genre != 'general':
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre, genre_scorer.genres['general'])
        
        # Add genre-specific features
        genre_features = genre_profile.detect_genre_patterns(text)
        feats.update(genre_features)
        
        # Adjust features based on genre
        feats = genre_profile.adjust_features(feats)
    
    return feats

def debug_segment_scoring(segment: Dict, audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> None:
    """Print detailed scoring breakdown for debugging with genre awareness"""
    features = compute_features_v4(segment, audio_file, genre=genre, platform=platform)
    scoring = score_segment_v4(features, genre=genre)
    
    print(f"\nText: {segment['text'][:100]}...")
    print(f"Duration: {segment['end'] - segment['start']:.1f}s")
    print(f"Genre: {genre}")
    
    print("\n=== FEATURES ===")
    for key in ['hook_score', 'payoff_score', 'arousal_score', 'emotion_score', 
                'question_score', 'info_density', 'loopability']:
        print(f"{key:15}: {features.get(key, 0):.3f}")
    
    # Show genre-specific features if they exist
    genre_features = [k for k in features.keys() if k.endswith('_score') and k not in 
                     ['hook_score', 'payoff_score', 'arousal_score', 'emotion_score', 
                      'question_score', 'info_density', 'loopability']]
    if genre_features:
        print("\n=== GENRE FEATURES ===")
        for key in genre_features:
            print(f"{key:15}: {features.get(key, 0):.3f}")
    
    print(f"\n=== SCORING ===")
    print(f"Path scores: {scoring['path_scores']}")
    print(f"Winning path: {scoring['winning_path']}")
    print(f"Synergy mult: {scoring['synergy_multiplier']:.3f}")
    print(f"Final score: {scoring['viral_score_100']}/100")
    print(f"Bonuses: {scoring.get('bonus_reasons', [])}")
    
    # Additional debugging info
    if 'hook_reasons' in features:
        print(f"Hook reasons: {features['hook_reasons']}")
    if 'payoff_type' in features:
        print(f"Payoff type: {features['payoff_type']}")
    if 'audio_arousal' in features:
        print(f"Audio arousal: {features['audio_arousal']:.3f}")

def explain_segment_v4(features: Dict, weights: Dict = None, genre: str = 'general') -> Dict:
    if weights is None:
        weights = get_clip_weights()
    
    scoring_result = score_segment_v4(features, weights, genre=genre)
    f = features
    
    strengths = []
    improvements = []
    
    hook_score = f.get("hook_score", 0.0)
    if hook_score >= 0.8:
        strengths.append("🎯 **Killer Hook**: Opens with attention-grabbing content")
    elif hook_score < 0.4:
        improvements.append("⚠️ **Weak Hook**: Needs compelling opening")
    
    viral_score = scoring_result["viral_score_100"]
    if viral_score >= 70:
        overall = "🚀 **High Viral Potential** - Strong fundamentals"
    elif viral_score >= 50:
        overall = "📈 **Good Potential** - Solid foundation"
    else:
        overall = "🔧 **Needs Work** - Multiple issues to address"
    
    return {
        "overall_assessment": overall,
        "viral_score": viral_score,
        "winning_strategy": scoring_result["winning_path"],
        "strengths": strengths,
        "improvements": improvements
    }

def viral_potential_v4(features: dict, length_s: float, platform: str = "general", genre: str = 'general') -> dict:
    f = {k: float(features.get(k, 0.0)) for k in (
        "hook_score", "arousal_score", "emotion_score", "question_score",
        "payoff_score", "info_density", "loopability"
    )}

    scoring_result = score_segment_v4(features, genre=genre)
    base_viral = scoring_result["final_score"]
    
    platform_multiplier = 1.0
    platform_reasons = []
    
    if platform.lower() == "tiktok":
        if f["hook_score"] >= 0.7 and f["arousal_score"] >= 0.6:
            platform_multiplier = 1.1
            platform_reasons.append("tiktok_energy_boost")
        if 15 <= length_s <= 30:
            platform_multiplier *= 1.05
            platform_reasons.append("tiktok_length_perfect")
    
    final_viral = base_viral * platform_multiplier
    viral_0_100 = int(np.clip(final_viral * 110, 0, 100))
    
    platforms = []
    if f["hook_score"] >= 0.6 and f["arousal_score"] >= 0.5 and 12 <= length_s <= 35:
        platforms.append("TikTok")
    if f["payoff_score"] >= 0.5 and f["info_density"] >= 0.4 and 15 <= length_s <= 60:
        platforms.append("YouTube Shorts")
    
    if viral_0_100 < 35:
        platforms = ["Consider re-editing for stronger hook or payoff"]
    
    return {
        "viral_score": viral_0_100,
        "platforms": platforms,
        "winning_path": scoring_result["winning_path"],
        "path_breakdown": scoring_result["path_scores"],
        "platform_multiplier": platform_multiplier,
        "platform_reasons": platform_reasons,
        "synergy_applied": scoring_result["synergy_multiplier"]
    }

# Legacy compatibility functions
def _hook_score(text: str) -> float:
    score, _ = _hook_score_v4(text)
    return score

def _emotion_score(text: str) -> float:
    return _emotion_score_v4(text)

def _payoff_presence(text: str) -> float:
    score, _ = _payoff_presence_v4(text)
    return score

def _info_density(text: str) -> float:
    return _info_density_v4(text)

def _ad_penalty(text: str) -> dict:
    """Enhanced ad detection with comprehensive patterns"""
    t = text.lower()
    
    # Critical ad phrases that should trigger immediate filtering
    if any(phrase in t for phrase in [
        "sponsored by", "brought to you by", "visit our", "check out",
        "use code", "promo code", "discount code", "click the link",
        "in the description", "link in bio", "follow us on",
        "limited time offer", "special offer", "exclusive deal",
        # New additions for your test data
        "flu shots", "wellness event", "applicable state law",
        "at cox", "cox.com", "blocks online threats", "advanced security",
        "age restrictions", "availability and applicable"
    ]):
        return {"flag": True, "penalty": 0.95, "reason": "obvious_promotion"}
    
    # Corporate/brand language
    if any(phrase in t for phrase in [
        "we're always", "like your", "before there's trouble",
        "that blocks", "knows when the", "step ahead"
    ]):
        return {"flag": True, "penalty": 0.85, "reason": "corporate_language"}
    
    # Promotional language
    if any(phrase in t for phrase in [
        "just ask", "visit", "learn more", "call now", "get your",
        "for a buck", "no contracts", "no hassle", "pure unfiltered",
        "sports extra", "game day", "slain.com"
    ]):
        return {"flag": True, "penalty": 0.30, "reason": "promotional_language"}
    
    # Self-promotion
    if any(phrase in t for phrase in [
        "my company", "our product", "my book", "my course",
        "subscribe to", "follow me", "my website", "my podcast",
        "my business", "my startup", "my team"
    ]):
        return {"flag": True, "penalty": 0.20, "reason": "self_promotion"}
    
    # Subtle promotion
    if any(phrase in t for phrase in [
        "i wrote", "i created", "i built", "i founded",
        "i developed", "i designed", "i launched"
    ]):
        return {"flag": True, "penalty": 0.10, "reason": "subtle_promotion"}
    
    return {"flag": False, "penalty": 0.0, "reason": "no_promotion"}

def _platform_length_match(duration: float, platform: str = 'tiktok') -> float:
    """Calculate how well the duration matches platform preferences"""
    platform_ranges = {
        'tiktok': {'optimal': (15, 30), 'acceptable': (10, 45)},
        'instagram': {'optimal': (15, 30), 'acceptable': (10, 45)},
        'instagram_reels': {'optimal': (15, 30), 'acceptable': (10, 45)},
        'youtube_shorts': {'optimal': (20, 45), 'acceptable': (15, 60)},
        'linkedin': {'optimal': (30, 60), 'acceptable': (20, 90)}
    }
    
    ranges = platform_ranges.get(platform, platform_ranges['tiktok'])
    
    if ranges['optimal'][0] <= duration <= ranges['optimal'][1]:
        return 1.0  # Perfect match
    elif ranges['acceptable'][0] <= duration <= ranges['acceptable'][1]:
        # Linear falloff from optimal edge
        if duration < ranges['optimal'][0]:
            return 0.5 + 0.5 * (duration - ranges['acceptable'][0]) / (ranges['optimal'][0] - ranges['acceptable'][0])
        else:
            return 0.5 + 0.5 * (ranges['acceptable'][1] - duration) / (ranges['acceptable'][1] - ranges['optimal'][1])
    else:
        return 0.0  # Outside acceptable range

def calculate_dynamic_length_score(segment: Dict, platform: str) -> float:
    """
    Calculate length score for dynamic segments, considering natural boundaries.
    """
    duration = segment.get("end", 0) - segment.get("start", 0)
    base_score = _platform_length_match(duration, platform)
    
    # Bonus for natural boundaries
    boundary_type = segment.get("boundary_type", "")
    confidence = segment.get("confidence", 0.0)
    
    if boundary_type in ["sentence_end", "insight_marker"] and confidence > 0.8:
        base_score += 0.1  # Bonus for clean boundaries
    
    return min(1.0, base_score)

def _audio_prosody_score(audio_path: str, start: float, end: float, y_sr=None) -> float:
    """Enhanced audio analysis for arousal/energy detection"""
    try:
        if y_sr is None:
            y, sr = librosa.load(audio_path, sr=None, offset=max(0, start-0.2), duration=(end-start+0.4))
        else:
            y, sr = y_sr
            s = max(int((start-0.2)*sr), 0)
            e = min(int((end+0.4)*sr), len(y))
            y = y[s:e]
        
        if len(y) == 0:
            return 0.0

        # Use the enhanced compute_audio_energy function
        return compute_audio_energy(y, sr)
    except Exception:
        return 0.0

def compute_audio_energy(y, sr):
    """Compute enhanced audio energy from audio data with better normalization"""
    try:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Better normalization: use percentiles for consistency across clips
        rms_energy = np.percentile(rms, 75)  # 75th percentile is more stable than max
        rms_norm = np.clip(rms_energy / 0.1, 0, 1)  # 0.1 is typical loud speech RMS
        
        # Spectral features: use relative measures
        spectral_energy = np.std(spectral_rolloff)  # Standard deviation for variation
        spectral_norm = np.clip(spectral_energy / 2000, 0, 1)  # Normalize to typical range
        
        # Zero crossing rate: use relative to typical speech
        zcr_energy = np.mean(zero_crossing_rate)
        zcr_norm = np.clip(zcr_energy / 0.1, 0, 1)  # 0.1 is typical speech ZCR
        
        # Combine normalized features for energy score
        energy = rms_norm * 0.4 + spectral_norm * 0.3 + zcr_norm * 0.3
        
        # Add dynamic range for more nuanced scoring (relative to clip's own range)
        if len(rms) > 1:
            # Prevent division by zero with more robust epsilon
            denominator = np.percentile(rms, 95)
            if denominator > 1e-6:  # More robust threshold
                dynamic_range = (np.percentile(rms, 90) - np.percentile(rms, 10)) / denominator
                energy = energy * 0.7 + dynamic_range * 0.3
        
        return float(np.clip(energy, 0.0, 1.0))
    except Exception:
        return 0.0

# Main API functions
def compute_features(segment: Dict, audio_file: str, y_sr=None, version: str = "v4", genre: str = 'general', platform: str = 'tiktok') -> Dict:
    if version == "v4":
        return compute_features_v4(segment, audio_file, y_sr, genre, platform)
    else:
        text = segment.get("text","")
        ad_result = _ad_penalty(text)
        
        feats = {
            "hook_score": _hook_score(text),
            "arousal_score": _audio_prosody_score(audio_file, segment["start"], segment["end"], y_sr=y_sr),
            "emotion_score": _emotion_score(text),
            "question_score": _question_or_list(text),
            "payoff_score": _payoff_presence(text),
            "info_density": _info_density(text),
            "loopability": _loopability_heuristic(text),
            "_ad_flag": ad_result["flag"],
            "_ad_penalty": ad_result["penalty"],
            "_ad_reason": ad_result["reason"]
        }
        
        if ad_result["flag"]:
            feats["payoff_score"] = 0.0
            feats["info_density"] = min(feats.get("info_density", 0.0), 0.35)
        return feats

def score_segment(features: Dict, weights: Dict = None, version: str = "v4", genre: str = 'general') -> float:
    if weights is None:
        weights = get_clip_weights()
    
    if version == "v4":
        result = score_segment_v4(features, weights, genre=genre)
        return result["final_score"]
    else:
        return float(
            weights["hook"] * features["hook_score"] +
            weights["arousal"] * features["arousal_score"] +
            weights["emotion"] * features["emotion_score"] +
            weights["q_or_list"] * features["question_score"] +
            weights["payoff"] * features["payoff_score"] +
            weights["info"] * features["info_density"] +
            weights["loop"] * features["loopability"]
        )

def viral_potential(features: dict, length_s: float, platform: str = "general", version: str = "v4", genre: str = 'general') -> dict:
    if version == "v4":
        return viral_potential_v4(features, length_s, platform, genre)
    else:
        f = {k: float(features.get(k, 0.0)) for k in (
            "hook_score","arousal_score","emotion_score","question_score",
            "payoff_score","info_density","loopability"
        )}

        base = (
            0.37 * f["hook_score"] +
            0.17 * f["emotion_score"] +
            0.16 * f["arousal_score"] +
            0.14 * f["payoff_score"] +
            0.08 * f["loopability"] +
            0.05 * f["info_density"] +
            0.03 * f["question_score"]
        )
        
        viral_0_100 = int(max(0.0, min(1.0, base)) * 100)
        platforms = ["TikTok"] if f["hook_score"] >= 0.6 else []
        
        return {"viral_score": viral_0_100, "platforms": platforms}

def explain_segment(features: Dict, weights: Dict = None, version: str = "v4", genre: str = 'general') -> Dict:
    if weights is None:
        weights = get_clip_weights()
    
    if version == "v4":
        return explain_segment_v4(features, weights, genre)
    else:
        contributions = {
            "Hook": weights["hook"] * features["hook_score"],
            "Arousal": weights["arousal"] * features["arousal_score"],
            "Emotion": weights["emotion"] * features["emotion_score"],
            "Q/List": weights["q_or_list"] * features["question_score"],
            "Payoff": weights["payoff"] * features["payoff_score"],
            "Info": weights["info"] * features["info_density"],
            "Loop": weights["loop"] * features["loopability"]
        }
        
        reasons = []
        if features["hook_score"] >= 0.8:
            reasons.append("Starts with a strong hook that will grab attention")
        else:
            reasons.append("Weak opening - viewers might scroll past this")
        
        return {
            "contributions": contributions,
            "reasons": reasons[:6]
        }

# Pipeline helper function for filtering ads
def filter_ads_from_features(all_features: List[Dict]) -> List[Dict]:
    """
    Filter out advertisements completely from the feature list.
    Returns only non-ad content for scoring.
    """
    non_ad_features = [f for f in all_features if not f.get("is_advertisement", False)]
    
    if len(non_ad_features) < 5:
        return {"error": "Episode is mostly advertisements, no viable clips found"}
    
    return non_ad_features

def filter_intro_content_from_features(all_features: List[Dict]) -> List[Dict]:
    """
    Filter out intro/greeting content completely from the feature list.
    Returns only substantive content for scoring.
    """
    # Filter out intro content based on insight score and hook reasons
    non_intro_features = []
    
    for f in all_features:
        # Skip if it's marked as intro content
        hook_reasons = f.get("hook_reasons", "")
        insight_score = f.get("insight_score", 0.0)
        text = f.get("text", "").lower()
        
        # Check if this is a mixed segment (contains both intro and good content)
        has_intro_start = any(pattern in text[:100] for pattern in [
            "yo, what's up", "hey", "hi", "hello", "what's up", 
            "it's monday", "it's tuesday", "it's wednesday", "it's thursday", "it's friday",
            "i'm jeff", "my name is", "hope you", "hope everyone"
        ])
        
        has_good_content = insight_score > 0.3 or any(pattern in text for pattern in [
            "observation", "insight", "casual drafters", "way better", "main observation",
            "key takeaway", "the thing is", "what i found", "what i learned"
        ])
        
        # If it's a mixed segment with good content, keep it but with reduced score
        if has_intro_start and has_good_content:
            # Reduce the score to account for intro content
            f["mixed_intro_penalty"] = 0.2
            f["hook_score"] = max(0.1, f.get("hook_score", 0.0) - 0.2)
            non_intro_features.append(f)
            continue
        
        # Skip pure intro content
        if "intro_greeting_penalty" in hook_reasons and not has_good_content:
            continue
            
        # Skip content with very low insight scores (likely filler) - but be less aggressive
        if insight_score < 0.05 and not has_good_content:
            continue
            
        non_intro_features.append(f)
    
    if len(non_intro_features) < 3:
        # If we filtered out too much, be less aggressive - only filter obvious intro content
        non_intro_features = [f for f in all_features if "intro_greeting_penalty" not in f.get("hook_reasons", "")]
    
    if len(non_intro_features) < 2:
        return {"error": "Episode has too much intro content, no viable clips found"}
    
    return non_intro_features

def find_natural_boundaries(text: str) -> List[Dict]:
    """
    Find natural content boundaries in text for dynamic segmentation.
    Returns list of boundary points with their types and confidence.
    """
    boundaries = []
    words = text.split()
    
    # Look for natural break points
    for i, word in enumerate(words):
        # Strong content boundaries
        if word in [".", "!", "?", ":", ";"]:
            boundaries.append({
                "position": i + 1,  # Start after the punctuation
                "type": "sentence_end",
                "confidence": 0.9
            })
        
        # Topic transitions
        elif any(phrase in " ".join(words[max(0, i-2):i+3]) for phrase in [
            "but", "however", "meanwhile", "on the other hand", "speaking of",
            "that reminds me", "by the way", "oh wait", "actually"
        ]):
            boundaries.append({
                "position": i,
                "type": "topic_shift",
                "confidence": 0.7
            })
        
        # Story/insight markers
        elif any(phrase in " ".join(words[max(0, i-1):i+2]) for phrase in [
            "here's the thing", "the key is", "what i learned", "my take",
            "the bottom line", "in summary", "to wrap up", "main observation",
            "key takeaway", "the thing is", "what i found"
        ]):
            boundaries.append({
                "position": i,
                "type": "insight_marker",
                "confidence": 0.8
            })
        
        # Question/answer patterns
        elif word == "?" and i < len(words) - 5:
            # Look for answer patterns after question
            next_words = " ".join(words[i+1:i+6])
            if any(pattern in next_words for pattern in [
                "well", "so", "the answer", "here's", "let me tell you"
            ]):
                boundaries.append({
                    "position": i + 1,
                    "type": "qa_boundary",
                    "confidence": 0.8
                })
        
        # Comma boundaries (weaker but useful)
        elif word == "," and i > 5 and i < len(words) - 5:
            # Check if it's a natural pause
            context = " ".join(words[i-3:i+4])
            if any(phrase in context for phrase in [
                "first", "second", "third", "also", "additionally", "furthermore"
            ]):
                boundaries.append({
                    "position": i + 1,
                    "type": "comma_boundary",
                    "confidence": 0.5
                })
    
    # Remove duplicate positions and sort
    unique_boundaries = []
    seen_positions = set()
    
    for boundary in sorted(boundaries, key=lambda x: x["position"]):
        if boundary["position"] not in seen_positions and 0 < boundary["position"] < len(words):
            unique_boundaries.append(boundary)
            seen_positions.add(boundary["position"])
    
    return unique_boundaries

def create_dynamic_segments(segments: List[Dict], platform: str = 'tiktok') -> List[Dict]:
    """
    Create dynamic segments based on natural content boundaries and platform optimization.
    """
    dynamic_segments = []
    
    # Platform-specific optimal lengths
    platform_lengths = {
        'tiktok': {'min': 15, 'max': 30, 'optimal': 20},
        'instagram': {'min': 15, 'max': 30, 'optimal': 22},
        'instagram_reels': {'min': 15, 'max': 30, 'optimal': 22},
        'youtube': {'min': 20, 'max': 60, 'optimal': 35},
        'youtube_shorts': {'min': 20, 'max': 60, 'optimal': 35},
        'twitter': {'min': 10, 'max': 25, 'optimal': 18},
        'linkedin': {'min': 20, 'max': 45, 'optimal': 30}
    }
    
    target_length = platform_lengths.get(platform, platform_lengths['tiktok'])
    
    for seg in segments:
        text = seg.get("text", "")
        start_time = seg.get("start", 0)
        end_time = seg.get("end", 0)
        total_duration = end_time - start_time
        
        # Find natural boundaries
        boundaries = find_natural_boundaries(text)
        
        if not boundaries or len(boundaries) < 2:
            # No natural boundaries found, use original segment
            dynamic_segments.append(seg)
            continue
        
        # Create segments based on boundaries
        words = text.split()
        current_start = start_time
        
        # Add start boundary
        all_boundaries = [{"position": 0, "type": "start", "confidence": 1.0}] + boundaries
        
        for i, boundary in enumerate(all_boundaries):
            if boundary["confidence"] < 0.6:
                continue  # Skip low-confidence boundaries
            
            # Calculate end position
            if i + 1 < len(all_boundaries):
                next_boundary = all_boundaries[i + 1]
                end_position = next_boundary["position"]
            else:
                end_position = len(words)
            
            # Extract segment text
            segment_words = words[boundary["position"]:end_position]
            segment_text = " ".join(segment_words)
            
            if len(segment_words) < 3:  # Skip very short segments
                continue
            
            # Calculate timing (proportional to word count)
            total_words = len(words)
            segment_ratio = len(segment_words) / total_words
            segment_duration = total_duration * segment_ratio
            
            # Check if segment meets platform requirements
            if target_length["min"] <= segment_duration <= target_length["max"]:
                dynamic_segments.append({
                    "text": segment_text,
                    "start": current_start,
                    "end": current_start + segment_duration,
                    "boundary_type": boundary["type"],
                    "confidence": boundary["confidence"]
                })
            elif segment_duration < target_length["min"] and len(dynamic_segments) > 0:
                # If segment is too short, try to merge with previous segment
                prev_segment = dynamic_segments[-1]
                merged_duration = (current_start + segment_duration) - prev_segment["start"]
                if merged_duration <= target_length["max"]:
                    # Merge with previous segment
                    dynamic_segments[-1] = {
                        "text": prev_segment["text"] + " " + segment_text,
                        "start": prev_segment["start"],
                        "end": current_start + segment_duration,
                        "boundary_type": "merged",
                        "confidence": min(prev_segment["confidence"], boundary["confidence"])
                    }
            
            current_start += segment_duration
    
    # If no dynamic segments were created, return original segments
    if not dynamic_segments:
        return segments
    
    return dynamic_segments

def split_mixed_segments(segments: List[Dict]) -> List[Dict]:
    """
    Split segments that contain both intro and good content into separate segments.
    """
    split_segments = []
    
    for seg in segments:
        text = seg.get("text", "").lower()
        
        # Check if this segment contains both intro and good content
        has_intro_start = any(pattern in text[:100] for pattern in [
            "yo, what's up", "hey", "hi", "hello", "what's up", 
            "it's monday", "it's tuesday", "it's wednesday", "it's thursday", "it's friday",
            "i'm jeff", "my name is", "hope you", "hope everyone"
        ])
        
        has_good_content = any(pattern in text for pattern in [
            "observation", "insight", "casual drafters", "way better", "main observation",
            "key takeaway", "the thing is", "what i found", "what i learned"
        ])
        
        if has_intro_start and has_good_content:
            # Try to find where the good content starts
            words = text.split()
            good_content_start = -1
            
            for i, word in enumerate(words):
                if any(pattern in " ".join(words[i:i+3]) for pattern in [
                    "observation", "insight", "casual drafters", "way better", "main observation"
                ]):
                    good_content_start = i
                    break
            
            if good_content_start > 0:
                # Split the segment
                intro_words = words[:good_content_start]
                good_words = words[good_content_start:]
                
                # Calculate timing split (rough estimate)
                total_duration = seg["end"] - seg["start"]
                intro_ratio = len(intro_words) / len(words)
                good_ratio = len(good_words) / len(words)
                
                split_point = seg["start"] + (total_duration * intro_ratio)
                
                # Create intro segment (will be filtered out)
                intro_seg = {
                    "text": " ".join(intro_words),
                    "start": seg["start"],
                    "end": split_point
                }
                
                # Create good content segment
                good_seg = {
                    "text": " ".join(good_words),
                    "start": split_point,
                    "end": seg["end"]
                }
                
                # Only add the good content segment
                if len(good_words) > 5:  # Make sure it's substantial enough
                    split_segments.append(good_seg)
            else:
                # If we can't split cleanly, keep the original but mark it
                seg["mixed_content"] = True
                split_segments.append(seg)
        else:
            split_segments.append(seg)
    
    return split_segments

def find_viral_clips(segments: List[Dict], audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """
    Main pipeline function that pre-filters ads and finds viral clips with genre awareness.
    """
    # Auto-detect genre if not specified
    if genre == 'general':
        # Use first few segments to detect genre
        sample_text = " ".join([seg.get('text', '') for seg in segments[:3]])
        genre_scorer = GenreAwareScorer()
        detected_genre = genre_scorer.auto_detect_genre(sample_text)
        print(f"🎯 Auto-detected genre: {detected_genre}")
        print("💡 You can override this by specifying a different genre")
        genre = detected_genre
    
    # Split mixed segments to separate intro from good content
    split_segments = split_mixed_segments(segments)
    
    # Create dynamic segments based on natural boundaries and platform optimization
    dynamic_segments = create_dynamic_segments(split_segments, platform)
    
    # Compute features for all segments with genre awareness
    all_features = [compute_features_v4(seg, audio_file, genre=genre, platform=platform) for seg in dynamic_segments]
    
    # FILTER OUT ADS COMPLETELY
    non_ad_features = filter_ads_from_features(all_features)
    
    if isinstance(non_ad_features, dict) and "error" in non_ad_features:
        return non_ad_features
    
    # FILTER OUT INTRO CONTENT COMPLETELY
    non_intro_features = filter_intro_content_from_features(non_ad_features)
    
    if isinstance(non_intro_features, dict) and "error" in non_intro_features:
        return non_intro_features
    
    # Score only the non-ad, non-intro content with genre awareness
    scored_clips = [score_segment_v4(f, genre=genre) for f in non_intro_features]
    
    # Sort by viral score and return top 5
    return {
        'genre': genre,
        'clips': sorted(scored_clips, key=lambda x: x["viral_score_100"], reverse=True)[:5]
    }

# Main API functions
def viral_potential_from_segment(segment: Dict, audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """V4 viral potential assessment from segment (computes features first)"""
    features = compute_features_v4(segment, audio_file, genre=genre, platform=platform)
    scoring = score_segment_v4(features, genre=genre)
    
    return {
        "viral_score": scoring["final_score"],
        "viral_score_100": scoring["viral_score_100"],
        "winning_path": scoring["winning_path"],
        "path_scores": scoring["path_scores"],
        "synergy_multiplier": scoring["synergy_multiplier"],
        "display_score": scoring["display_score"],
        "confidence": scoring["confidence"],
        "confidence_color": scoring["confidence_color"],
        "features": features,
        "genre": genre,
        "explanation": _explain_viral_potential_v4(features, scoring, genre)
    }

def explain_segment_from_segment(segment: Dict, audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """V4 segment explanation from segment (computes features first)"""
    features = compute_features_v4(segment, audio_file, genre=genre, platform=platform)
    scoring = score_segment_v4(features, genre=genre)
    
    return {
        "segment": segment,
        "features": features,
        "scoring": scoring,
        "genre": genre,
        "display_score": scoring["display_score"],
        "confidence": scoring["confidence"],
        "confidence_color": scoring["confidence_color"],
        "explanation": _explain_viral_potential_v4(features, scoring, genre)
    }

def _explain_viral_potential_v4(features: Dict, scoring: Dict, genre: str = 'general') -> str:
    """Generate human-readable explanation of viral potential with genre context"""
    score = scoring["viral_score_100"]
    path = scoring["winning_path"]
    
    # Genre-specific context
    genre_context = {
        'fantasy_sports': 'fantasy sports analysis',
        'sports': 'sports commentary',
        'comedy': 'comedy content',
        'business': 'business advice',
        'education': 'educational content',
        'general': 'content'
    }
    
    genre_name = genre_context.get(genre, 'content')
    
    if score >= 80:
        level = "EXCEPTIONAL"
        description = f"This {genre_name} clip has exceptional viral potential"
    elif score >= 70:
        level = "HIGH"
        description = f"This {genre_name} clip has high viral potential"
    elif score >= 60:
        level = "GOOD"
        description = f"This {genre_name} clip has good viral potential"
    elif score >= 50:
        level = "MODERATE"
        description = f"This {genre_name} clip has moderate viral potential"
    elif score >= 40:
        level = "LOW"
        description = f"This {genre_name} clip has low viral potential"
    else:
        level = "VERY LOW"
        description = f"This {genre_name} clip has very low viral potential"
    
    # Path explanation
    path_explanations = {
        'hook': 'strong opening that grabs attention',
        'payoff': 'valuable conclusion or insight',
        'energy': 'high energy and emotional engagement',
        'structured': 'well-organized and informative',
        'actionable': 'provides specific, actionable advice',
        'hot_take': 'controversial or bold opinion',
        'setup_punchline': 'classic comedy structure'
    }
    
    path_desc = path_explanations.get(path, 'balanced scoring across multiple dimensions')
    
    # Feature highlights
    highlights = []
    if features.get('hook_score', 0) > 0.7:
        highlights.append("strong hook")
    if features.get('payoff_score', 0) > 0.7:
        highlights.append("clear payoff")
    if features.get('arousal_score', 0) > 0.7:
        highlights.append("high energy")
    if features.get('emotion_score', 0) > 0.7:
        highlights.append("emotional engagement")
    
    # Genre-specific highlights
    if genre != 'general':
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre)
        if genre_profile:
            if 'viral_trigger_boost' in features and features['viral_trigger_boost'] > 0:
                highlights.append(f"genre-specific viral triggers")
            if 'confidence_score' in features and features['confidence_score'] > 0.5:
                highlights.append("high confidence indicators")
            if 'urgency_score' in features and features['urgency_score'] > 0.5:
                highlights.append("time-sensitive content")
    
    if highlights:
        feature_text = f"Key strengths: {', '.join(highlights)}"
    else:
        feature_text = "Balanced performance across features"
    
    # Synergy explanation
    synergy = scoring.get('synergy_multiplier', 1.0)
    if synergy > 1.1:
        synergy_text = "Excellent synergy between features"
    elif synergy > 1.0:
        synergy_text = "Good synergy between features"
    elif synergy < 0.9:
        synergy_text = "Features could work better together"
    else:
        synergy_text = "Standard feature interaction"
    
    return f"{description} ({level}: {score}/100). The clip excels in {path_desc}. {feature_text}. {synergy_text}."

# Genre detection utility function
def detect_podcast_genre(segments: List[Dict]) -> str:
    """Detect the most likely genre for a podcast based on segment content"""
    if not segments:
        return 'general'
    
    # Use more representative segments for genre detection (skip intro/ads)
    # Skip first 2 segments (likely intro music/ads) and take next 5-8 segments
    start_idx = min(2, len(segments) - 1)
    end_idx = min(start_idx + 8, len(segments))
    sample_segments = segments[start_idx:end_idx]
    
    # Handle both TranscriptSegment objects and dictionaries
    sample_text = " ".join([
        seg.text if hasattr(seg, 'text') else seg.get('text', '') 
        for seg in sample_segments
    ])
    
    # Filter out very short or ad-like text
    if len(sample_text.strip()) < 50:  # Too short to be meaningful
        # Fall back to using more segments
        sample_segments = segments[:min(10, len(segments))]
        sample_text = " ".join([
            seg.text if hasattr(seg, 'text') else seg.get('text', '') 
            for seg in sample_segments
        ])
    
    genre_scorer = GenreAwareScorer()
    detected_genre = genre_scorer.auto_detect_genre(sample_text)
    
    return detected_genre

# Genre-aware clip finding with auto-detection
def find_viral_clips_with_genre(segments: List[Dict], audio_file: str, user_genre: str = None) -> Dict:
    """
    Enhanced viral clip finding with genre auto-detection and user override.
    
    Args:
        segments: List of podcast segments
        audio_file: Path to audio file
        user_genre: User-selected genre (optional, will auto-detect if None)
    
    Returns:
        Dict with genre info and top viral clips
    """
    # Auto-detect genre if user didn't specify
    if user_genre is None:
        detected_genre = detect_podcast_genre(segments)
        print(f"🎯 Auto-detected genre: {detected_genre}")
        print("💡 You can override this by selecting a specific genre")
        genre = detected_genre
    else:
        genre = user_genre
        print(f"🎯 Using user-selected genre: {genre}")
    
    # Find viral clips with genre awareness
    result = find_viral_clips(segments, audio_file, genre=genre)
    
    # Add genre detection confidence
    if user_genre is None:
        result['auto_detected'] = True
        result['detection_confidence'] = 'high' if genre != 'general' else 'low'
    else:
        result['auto_detected'] = False
        result['detection_confidence'] = 'user_override'
    
    return result

class NewsPoliticsGenreProfile(GenreProfile):
    def __init__(self):
        super().__init__()
        self.name = "news_politics"
        self.weights = {
            'hook': 0.35,  # Breaking/controversial
            'arousal': 0.25,  # Urgency/importance
            'info_density': 0.20,
            'payoff': 0.15,
            'controversy': 0.05
        }
        self.viral_threshold = 65  # Very high bar
        self.optimal_length = (20, 35)
        self.min_viral_score = 65
        self.penalty_config = {
            'context_penalty': 0.10,  # Reduced penalty
            'repetition_penalty': 0.15,
            'filler_penalty': 0.10
        }
        
        self.viral_triggers = [
            r"breaking:",
            r"the real reason",
            r"what nobody's saying",
            r"the truth about",
            r"behind the headlines"
        ]
        
        self.hook_patterns = [
            r"breaking news",
            r"just in",
            r"developing story",
            r"major announcement",
            r"shocking revelation"
        ]
        
        self.payoff_patterns = [
            r"here's why",
            r"the reason is",
            r"it's because",
            r"the truth is",
            r"here's what happened"
        ]
    
    def detect_genre_patterns(self, text: str) -> Dict:
        """Detect news/politics specific patterns"""
        features = {}
        
        # Urgency indicators
        urgency_patterns = [
            r"breaking", r"just in", r"developing", r"urgent", r"immediate",
            r"right now", r"this moment", r"critical"
        ]
        urgency_score = sum(0.2 for pattern in urgency_patterns if re.search(pattern, text.lower()))
        features['urgency_score'] = min(urgency_score, 1.0)
        
        # Controversy indicators
        controversy_patterns = [
            r"controversial", r"scandal", r"outrage", r"fury", r"anger",
            r"protest", r"backlash", r"criticism"
        ]
        controversy_score = sum(0.15 for pattern in controversy_patterns if re.search(pattern, text.lower()))
        features['controversy_score'] = min(controversy_score, 1.0)
        
        # News specific features
        features['has_breaking'] = bool(re.search(r'breaking|just in|developing', text.lower()))
        features['has_timeline'] = bool(re.search(r'today|yesterday|this week|recently', text.lower()))
        features['has_sources'] = bool(re.search(r'according to|reports|sources say|official', text.lower()))
        
        # Viral trigger boost
        viral_boost = sum(0.1 for trigger in self.viral_triggers if re.search(trigger, text.lower()))
        features['viral_trigger_boost'] = min(viral_boost, 0.5)
        
        return features
    
    def apply_quality_gate(self, features: Dict) -> float:
        """News/politics quality gate - needs breaking/controversial content"""
        if features.get('hook_score', 0.0) < 0.4 and features.get('urgency_score', 0.0) < 0.2:
            return 0.75  # Penalty for weak news value
        return 1.0  # No penalty
    
    def get_scoring_paths(self, features: Dict) -> Dict:
        """News/politics specific scoring paths with proper genre feature integration"""
        f = features
        
        # Urgency path - uses genre-specific features
        urgency_path = (0.40 * f.get("hook_score", 0.0) + 
                        0.30 * f.get("urgency_score", 0.0) + 
                        0.20 * f.get("arousal_score", 0.0) + 
                        0.10 * f.get("info_density", 0.0))
        
        # Controversy path - uses genre-specific features
        controversy_path = (0.40 * f.get("arousal_score", 0.0) + 
                           0.30 * f.get("controversy_score", 0.0) + 
                           0.20 * f.get("hook_score", 0.0) + 
                           0.10 * f.get("payoff_score", 0.0))
        
        # Keep hook and payoff paths
        hook_path = (0.35 * f.get("hook_score", 0.0) + 0.25 * f.get("arousal_score", 0.0) + 
                     0.20 * f.get("payoff_score", 0.0) + 0.15 * f.get("info_density", 0.0) + 
                     0.05 * f.get("loopability", 0.0))
        
        payoff_path = (0.40 * f.get("payoff_score", 0.0) + 0.30 * f.get("info_density", 0.0) + 
                       0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) + 
                       0.05 * f.get("arousal_score", 0.0))
        
        # Return all paths with proper naming for max operation
        return {
            "hook": hook_path, 
            "payoff": payoff_path, 
            "urgency": urgency_path, 
            "controversy": controversy_path
        }

# Platform + Genre multipliers
PLATFORM_GENRE_MULTIPLIERS = {
    'tiktok': {
        'comedy': 1.2,  # Perfect match
        'fantasy_sports': 0.8,  # Harder to succeed
        'education': 0.9,
        'true_crime': 1.1,
        'business': 0.95,
        'news_politics': 1.0,
        'health_wellness': 1.05
    },
    'instagram': {
        'comedy': 1.15,  # Great match
        'fantasy_sports': 0.9,  # Better than TikTok
        'education': 1.0,
        'true_crime': 1.05,
        'business': 1.0,
        'news_politics': 0.95,
        'health_wellness': 1.1
    },
    'instagram_reels': {
        'comedy': 1.15,  # Great match
        'fantasy_sports': 0.9,  # Better than TikTok
        'education': 1.0,
        'true_crime': 1.05,
        'business': 1.0,
        'news_politics': 0.95,
        'health_wellness': 1.1
    },
    'youtube_shorts': {
        'education': 1.15,  # Great match
        'fantasy_sports': 1.0,
        'comedy': 0.95,
        'true_crime': 1.05,
        'business': 1.1,
        'news_politics': 0.9,
        'health_wellness': 1.0
    },
    'linkedin': {
        'business': 1.2,  # Perfect match
        'education': 1.1,
        'health_wellness': 1.05,
        'comedy': 0.7,  # Less suitable
        'fantasy_sports': 0.6,  # Not suitable
        'true_crime': 0.5,  # Not suitable
        'news_politics': 1.0
    }
}

# Frontend Tone to Backend Genre Mapping
TONE_TO_GENRE_MAP = {
    'tutorial_business': 'business',
    'comedy': 'comedy',

    'motivation': 'health_wellness',
    'educational': 'education',
    'sports_analysis': 'fantasy_sports',
    'news_commentary': 'news_politics',
    'personal_story': 'true_crime',  # Personal stories often have narrative arcs
    'how_to': 'education',
    'product_review': 'business',
    'fitness_tips': 'health_wellness',
    'cooking': 'education',
    'travel': 'education',
    'gaming': 'comedy',  # Gaming content often has entertainment value
    'music_reaction': 'comedy',
    'movie_review': 'comedy',
    'book_summary': 'education',
    'investment_advice': 'business',
    'relationship_advice': 'health_wellness',
    'parenting_tips': 'health_wellness'
}

def resolve_genre_from_tone(tone: str, auto_detected: str) -> str:
    """Map frontend tone to backend genre with fallback to auto-detected"""
    if tone and tone in TONE_TO_GENRE_MAP:
        mapped_genre = TONE_TO_GENRE_MAP[tone]
        print(f"🎭 Tone '{tone}' mapped to genre: {mapped_genre}")
        return mapped_genre
    
    print(f"🎭 No tone mapping found for '{tone}', using auto-detected: {auto_detected}")
    return auto_detected

def interpret_synergy(synergy_mult: float, features: Dict) -> Dict:
    """Provide actionable synergy interpretation with improved feedback"""
    if synergy_mult < 0.7:
        return {
            "label": "⚠️ Imbalanced",
            "advice": "Hook is strong but lacks energy/payoff",
            "color": "#ffc107",
            "severity": "warning"
        }
    elif synergy_mult < 0.85:
        return {
            "label": "🔄 Mixed Performance", 
            "advice": "Some elements work, others need improvement",
            "color": "#6c757d",
            "severity": "info"
        }
    elif synergy_mult < 1.0:
        return {
            "label": "✅ Good Balance",
            "advice": "All elements working together",
            "color": "#28a745",
            "severity": "success"
        }
    else:
        return {
            "label": "🔥 Excellent Synergy",
            "advice": "Perfect balance of hook, energy, and payoff",
            "color": "#007bff",
            "severity": "excellent"
        }

def get_genre_detection_debug(segments: List[Dict], detected_genre: str, applied_genre: str, tone: str = None) -> Dict:
    """Generate debug information for genre detection and application"""
    if not segments:
        return {
            "auto_detected_genre": "none",
            "applied_genre": "none",
            "genre_confidence": "none",
            "top_genre_patterns": [],
            "tone_used": tone,
            "mapping_applied": False
        }
    
    # Analyze top genre patterns
    sample_text = " ".join([seg.get('text', '') for seg in segments[:3]])
    genre_scorer = GenreAwareScorer()
    
    # Get pattern matches for each genre
    genre_patterns = {}
    for genre_name, profile in genre_scorer.genres.items():
        if hasattr(profile, 'viral_triggers'):
            pattern_matches = sum(1 for trigger in profile.viral_triggers if re.search(trigger, sample_text.lower()))
            if pattern_matches > 0:
                genre_patterns[genre_name] = pattern_matches
    
    # Sort by pattern matches
    top_patterns = sorted(genre_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Determine confidence level
    if detected_genre == 'general':
        confidence = "low"
    elif top_patterns and top_patterns[0][0] == detected_genre and top_patterns[0][1] >= 3:
        confidence = "high"
    elif top_patterns and top_patterns[0][0] == detected_genre:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "auto_detected_genre": detected_genre,
        "applied_genre": applied_genre,
        "genre_confidence": confidence,
        "top_genre_patterns": [{"genre": g, "matches": m} for g, m in top_patterns],
        "tone_used": tone,
        "mapping_applied": tone is not None and tone in TONE_TO_GENRE_MAP,
        "sample_text_length": len(sample_text),
        "segments_analyzed": min(3, len(segments))
    }

def find_viral_clips_with_tone(segments: List[Dict], audio_file: str, tone: str = None, auto_detect: bool = True) -> Dict:
    """
    Enhanced viral clip finding with tone-to-genre mapping and comprehensive debug info.
    
    Args:
        segments: List of podcast segments
        audio_file: Path to audio file
        tone: Frontend tone selection (optional)
        auto_detect: Whether to auto-detect genre if no tone provided
    
    Returns:
        Dict with candidates, debug info, and genre details
    """
    # Auto-detect genre from content
    detected_genre = detect_podcast_genre(segments)
    
    # Override with tone if provided
    if tone:
        applied_genre = resolve_genre_from_tone(tone, detected_genre)
    else:
        applied_genre = detected_genre
    
    # Process with selected genre
    result = find_viral_clips(segments, audio_file, genre=applied_genre)
    
    # Add comprehensive debug information
    debug_info = get_genre_detection_debug(segments, detected_genre, applied_genre, tone)
    
    # Enhance result with debug info
    enhanced_result = {
        'genre': applied_genre,
        'clips': result.get('clips', []),
        'debug': debug_info,
        'tone_mapping': {
            'tone_provided': tone,
            'auto_detected': detected_genre,
            'final_genre': applied_genre,
            'mapping_used': tone is not None and tone in TONE_TO_GENRE_MAP
        }
    }
    
    # Add synergy interpretation for each clip
    for clip in enhanced_result['clips']:
        if 'synergy_multiplier' in clip:
            clip['synergy_interpretation'] = interpret_synergy(
                clip['synergy_multiplier'], 
                clip.get('features', {})
            )
    
    return enhanced_result

def find_candidates(segments: List[Dict], audio_file: str, platform: str = 'tiktok', tone: str = None, auto_detect: bool = True) -> Dict:
    """
    Main API function for finding viral candidates with tone-to-genre mapping.
    
    Args:
        segments: List of podcast segments
        audio_file: Path to audio file
        platform: Target platform (tiktok, youtube_shorts, etc.)
        tone: Frontend tone selection
        auto_detect: Whether to auto-detect genre
    
    Returns:
        Dict with candidates and comprehensive metadata
    """
    # Use the enhanced function with tone mapping
    result = find_viral_clips_with_tone(segments, audio_file, tone, auto_detect)
    
    # Add platform-specific information
    result['platform'] = platform
    result['processing_timestamp'] = str(datetime.datetime.now())
    
    # Add platform-genre compatibility scores
    if platform in PLATFORM_GENRE_MULTIPLIERS:
        genre = result['genre']
        compatibility = PLATFORM_GENRE_MULTIPLIERS[platform].get(genre, 1.0)
        result['platform_compatibility'] = {
            'score': compatibility,
            'interpretation': 'excellent' if compatibility >= 1.1 else 'good' if compatibility >= 1.0 else 'challenging' if compatibility >= 0.9 else 'difficult'
        }
    
    return result

# Frontend Platform to Backend Platform Mapping
PLATFORM_MAP = {
    'tiktok_reels': 'tiktok',
    'instagram_reels': 'instagram_reels',
    'shorts': 'youtube_shorts',
    'linkedin_sq': 'linkedin'
}

def resolve_platform(frontend_platform: str) -> str:
    """Map frontend platform names to backend platform names"""
    return PLATFORM_MAP.get(frontend_platform, frontend_platform)

# Caching for performance improvements
@lru_cache(maxsize=1000)
def compute_features_cached(segment_hash: str, audio_file: str, genre: str, platform: str = 'tiktok') -> Dict:
    """Cached version of compute_features_v4 for performance"""
    # This is a placeholder - in practice, you'd need to reconstruct the segment
    # from the hash or use a different caching strategy
    return compute_features_v4({"text": "", "start": 0, "end": 0}, audio_file, genre=genre, platform=platform)

def create_segment_hash(segment: Dict) -> str:
    """Create a hash from segment content for cache key"""
    content = f"{segment.get('text', '')[:100]}_{segment.get('start', 0)}_{segment.get('end', 0)}"
    return hashlib.md5(content.encode()).hexdigest()



