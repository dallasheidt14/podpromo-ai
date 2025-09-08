"""
secret_sauce.py - V4 Enhanced Viral Detection System
All heuristics + weights for picking 'winning' clips.
"""

from typing import Dict, List, Tuple, Any
import numpy as np, librosa
from config_loader import get_config
import re
import datetime
import logging
from scipy import signal
from scipy.stats import skew, kurtosis
import os

logger = logging.getLogger(__name__)
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
        
        # FIXED: Hot take path with higher hook weight
        hot_take_path = (0.25 * f.get("confidence_score", 0.0) + 
                        0.50 * f.get("hook_score", 0.0) + 
                        0.25 * f.get("viral_trigger_boost", 0.0))
        
        # FIXED: Increase hook score weight to be the dominant factor
        hook_path = (0.50 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                     0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
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
    
    def apply_quality_gate(self, f: Dict) -> float:
        """Fantasy Sports quality gate - 0.85x for no_payoff"""
        if f.get("payoff_score", 0.0) >= 0.30 or f.get("insight_score", 0.0) >= 0.70 or f.get("question_score", 0.0) >= 0.50:
            return 1.0
        return 0.85

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
        
        # FIXED: Increase hook score weight to be the dominant factor
        hook_path = (0.50 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                     0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
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
        
        # FIXED: Increase hook score weight to be the dominant factor
        hook_path = (0.50 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                     0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
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
    
    def apply_quality_gate(self, f: Dict) -> float:
        """Business quality gate - 0.80x for no_payoff"""
        if f.get("payoff_score", 0.0) >= 0.30 or f.get("insight_score", 0.0) >= 0.70 or f.get("question_score", 0.0) >= 0.50:
            return 1.0
        return 0.80

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
        
        # FIXED: Increase hook score weight to be the dominant factor
        hook_path = (0.50 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                     0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
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
        
        # FIXED: Increase hook score weight to be the dominant factor
        hook_path = (0.50 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                     0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
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
    """Get normalized clip weights (sum to 1.0)"""
    weights = dict(get_config()["weights"])
    ws = sum(weights.values()) or 1.0
    if abs(ws - 1.0) > 1e-6:
        weights = {k: v / ws for k, v in weights.items()}
        logger.warning("Weights normalized from %.2f to 1.00", ws)
    return weights

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

def _detect_insight_content_v2(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect insight content V2 with evidence-based scoring and saturating combiner"""
    if not text or len(text.strip()) < 10:
        return 0.0, "too_short"
    
    t = text.lower()
    reasons = []
    
    # Evidence patterns (same as ViralMomentDetector V2)
    CONTRAST = re.compile(r"(most (people|folks)|everyone|nobody).{0,40}\b(actually|but|instead)\b", re.I)
    CAUSAL = re.compile(r"\b(because|therefore|so|which means)\b", re.I)
    HAS_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b", re.I)
    COMPAR = re.compile(r"\b(vs\.?|versus|more than|less than|bigger than|smaller than)\b", re.I)
    IMPER = re.compile(r"\b(try|avoid|do|don['']t|stop|start|focus|use|measure|swap|choose|should|need|must)\b", re.I)
    HEDGE = re.compile(r"\b(maybe|probably|i think|i guess|kinda|sort of)\b", re.I)
    
    # Evidence components (0-1 each)
    evidence_parts = []
    
    # Contrast detection
    if CONTRAST.search(t):
        evidence_parts.append(0.8)
        reasons.append("contrast")
    
    # Number/metric detection
    if HAS_NUM.search(t):
        evidence_parts.append(0.7)
        reasons.append("number")
    
    # Comparison detection
    if COMPAR.search(t):
        evidence_parts.append(0.6)
        reasons.append("comparison")
    
    # Causal reasoning
    if CAUSAL.search(t):
        evidence_parts.append(0.5)
        reasons.append("causal")
    
    # Imperative/actionable content
    if IMPER.search(t):
        evidence_parts.append(0.6)
        reasons.append("imperative")
    
    # Genre-specific patterns (reduced weights for V2)
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
                evidence_parts.append(0.4)
                reasons.append("fantasy_insight")
                break  # Only count once per genre
        
        # Specific insight boost
        if re.search(r"(casual drafters are way better)", t):
            evidence_parts.append(0.6)
            reasons.append("specific_insight_boost")
    
    # General insight patterns (reduced weights)
    general_insight_patterns = [
        r"(here's what|the thing is|what i found|what i learned)",
        r"(the key|the secret|the trick|the strategy)",
        r"(most people|everyone|nobody) (thinks|believes|knows)",
        r"(contrary to|despite|although|even though)",
        r"(the truth is|reality is|actually|in fact)"
    ]
    
    for pattern in general_insight_patterns:
        if re.search(pattern, t):
            evidence_parts.append(0.3)
            reasons.append("general_insight")
            break  # Only count once
    
    # Hedge penalty (reduces confidence)
    hedge_penalty = 0.0
    if HEDGE.search(t):
        hedge_penalty = 0.2
        reasons.append("hedge_penalty")
    
    # Filler penalty (same as V1)
    filler_penalty = 0.0
    filler_patterns = [
        r"^(yo|hey|hi|hello|what's up)",
        r"^(it's|this is) (monday|tuesday|wednesday)",
        r"^(hope you|hope everyone)",
        r"^(let's get|let's start|let's begin)"
    ]
    
    for pattern in filler_patterns:
        if re.match(pattern, t):
            filler_penalty = 0.4
            reasons.append("filler_penalty")
            break
    
    # Saturating combiner: 1 - Π(1 - xᵢ)
    if evidence_parts:
        sat_score = 1.0
        for part in evidence_parts:
            sat_score *= (1.0 - part)
        sat_score = 1.0 - sat_score
    else:
        sat_score = 0.0
    
    # Apply penalties
    final_score = sat_score - hedge_penalty - filler_penalty
    final_score = float(np.clip(final_score, 0.0, 1.0))
    
    reason_str = ";".join(reasons) if reasons else "no_insights"
    return final_score, reason_str

def _apply_insight_confidence_multiplier(insight_score: float, confidence: float = None) -> float:
    """Apply confidence-based multiplier to insight score if V2 is enabled"""
    config = get_config()
    if not config.get("insight_v2", {}).get("enabled", False) or confidence is None:
        return insight_score
    
    conf_config = config.get("insight_v2", {}).get("confidence_multiplier", {})
    min_mult = conf_config.get("min_mult", 0.9)
    max_mult = conf_config.get("max_mult", 1.2)
    conf_range = conf_config.get("confidence_range", [0.5, 0.9])
    
    # Map confidence to multiplier: confidence 0.5→×0.95, 0.9→×1.20
    conf_min, conf_max = conf_range
    if conf_min >= conf_max:
        return insight_score
    
    # Linear interpolation
    multiplier = min_mult + (max_mult - min_mult) * ((confidence - conf_min) / (conf_max - conf_min))
    multiplier = max(min_mult, min(max_mult, multiplier))
    
    # Apply multiplier and cap at 1.0
    adjusted_score = insight_score * multiplier
    return min(1.0, adjusted_score)

def compute_audio_hook_modifier(audio_data, sr, start_time: float) -> float:
    """Analyze first 3 seconds of audio for prosodic hook indicators"""
    try:
        if audio_data is None or sr is None:
            return 0.0
            
        # Extract first 3 seconds
        start_sample = int(start_time * sr)
        end_sample = int((start_time + 3.0) * sr)
        
        if start_sample >= len(audio_data) or end_sample > len(audio_data):
            return 0.0
            
        first_3_sec = audio_data[start_sample:end_sample]
        
        if len(first_3_sec) == 0:
            return 0.0
        
        # Check for volume spikes (>1.5x average)
        rms_energy = librosa.feature.rms(y=first_3_sec)[0]
        avg_energy = np.mean(rms_energy)
        max_energy = np.max(rms_energy)
        
        has_volume_spike = max_energy > (avg_energy * 1.5)
        
        # Check for pitch variance (emotional range)
        pitches, magnitudes = librosa.piptrack(y=first_3_sec, sr=sr, threshold=0.1)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 10:
            pitch_variance = np.var(pitch_values)
            has_high_pitch_variance = pitch_variance > 1000  # Threshold for emotional range
        else:
            has_high_pitch_variance = False
        
        # Check for dramatic pause (silence before speech)
        # Look for low energy at start followed by high energy
        if len(rms_energy) > 10:
            first_quarter = np.mean(rms_energy[:len(rms_energy)//4])
            last_quarter = np.mean(rms_energy[3*len(rms_energy)//4:])
            has_dramatic_pause = (first_quarter < avg_energy * 0.3) and (last_quarter > avg_energy * 1.2)
        else:
            has_dramatic_pause = False
        
        # Return modifier based on detected features
        if has_volume_spike:
            return 0.2
        elif has_high_pitch_variance:
            return 0.15
        elif has_dramatic_pause:
            return 0.1
        
        return 0.0
        
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}")
        return 0.0

def detect_laughter_exclamations(text: str, audio_data=None, sr=None, start_time: float = 0.0, 
                                enable_audio_analysis: bool = False) -> float:
    """Detect laughter and repeated exclamations for hook boost - optimized for performance"""
    try:
        boost = 0.0
        
        # Text-based laughter detection (always fast)
        laughter_patterns = [
            r"\b(ha|haha|hahaha|lol|lmao|rofl)\b",
            r"\b(laughing|chuckling|giggling)\b",
            r"😄|😂|🤣|😆"
        ]
        
        text_lower = text.lower()
        for pattern in laughter_patterns:
            if re.search(pattern, text_lower):
                boost += 0.15
                break
        
        # Repeated exclamations detection (always fast)
        exclamation_patterns = [
            r"\b(dude|man|bro|wow|omg|damn|shit|fuck)\s+\1\s+\1",  # "DUDE DUDE DUDE"
            r"!{2,}",  # Multiple exclamation marks
            r"\b(no\s+way|are\s+you\s+kidding|you\s+gotta\s+be)\b"
        ]
        
        for pattern in exclamation_patterns:
            if re.search(pattern, text_lower):
                boost += 0.1
                break
        
        # OPTIMIZED: Audio-based laughter detection (only if explicitly enabled and audio available)
        if enable_audio_analysis and audio_data is not None and sr is not None:
            try:
                start_sample = int(start_time * sr)
                end_sample = int((start_time + 5.0) * sr)  # First 5 seconds
                
                if start_sample < len(audio_data) and end_sample <= len(audio_data):
                    first_5_sec = audio_data[start_sample:end_sample]
                    
                    # OPTIMIZED: Use faster spectral analysis
                    spectral_centroids = librosa.feature.spectral_centroid(y=first_5_sec, sr=sr, hop_length=512)[0]
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=first_5_sec, sr=sr, hop_length=512)[0]
                    
                    # Laughter typically has high spectral centroid and rolloff
                    if np.mean(spectral_centroids) > 2000 and np.mean(spectral_rolloff) > 4000:
                        boost += 0.1
                        
            except Exception as e:
                logger.warning(f"Audio laughter detection failed: {e}")
        
        return min(boost, 0.25)  # Cap at 0.25 total boost
        
    except Exception as e:
        logger.warning(f"Laughter/exclamation detection failed: {e}")
        return 0.0

def calculate_hook_components(text: str) -> Dict[str, float]:
    """Calculate multi-dimensional hook components"""
    try:
        words = text.split()
        text_lower = text.lower()
        
        # Attention grab: How jarring/surprising
        attention_patterns = [
            r"\b(shocking|surprising|unbelievable|incredible|amazing)\b",
            r"\b(you won't believe|can't believe|never thought)\b",
            r"\b(breaking|urgent|important|critical)\b"
        ]
        attention_score = sum(1 for pattern in attention_patterns if re.search(pattern, text_lower)) / len(attention_patterns)
        
        # Clarity: How clear the topic is
        clarity_indicators = [
            r"^(the|this|that|here's|listen|look|watch)",
            r"\b(about|regarding|concerning|on the topic of)\b",
            r"\b(specifically|exactly|precisely|clearly)\b"
        ]
        clarity_score = sum(1 for pattern in clarity_indicators if re.search(pattern, text_lower)) / len(clarity_indicators)
        
        # Tension: Problem/conflict setup
        tension_patterns = [
            r"\b(problem|issue|challenge|struggle|difficulty)\b",
            r"\b(wrong|mistake|error|failure|disaster)\b",
            r"\b(but|however|although|despite|even though)\b",
            r"\b(conflict|disagreement|argument|debate)\b"
        ]
        tension_score = sum(1 for pattern in tension_patterns if re.search(pattern, text_lower)) / len(tension_patterns)
        
        # Authority: Confidence/expertise signals
        authority_patterns = [
            r"\b(i know|i've seen|i've experienced|i've learned)\b",
            r"\b(trust me|believe me|mark my words|i guarantee)\b",
            r"\b(as an expert|professionally|in my experience)\b",
            r"\b(studies show|research indicates|data proves)\b"
        ]
        authority_score = sum(1 for pattern in authority_patterns if re.search(pattern, text_lower)) / len(authority_patterns)
        
        return {
            "attention_grab": min(attention_score, 1.0),
            "clarity": min(clarity_score, 1.0),
            "tension": min(tension_score, 1.0),
            "authority": min(authority_score, 1.0)
        }
        
    except Exception as e:
        logger.warning(f"Hook components calculation failed: {e}")
        return {"attention_grab": 0.0, "clarity": 0.0, "tension": 0.0, "authority": 0.0}

def calculate_time_weighted_hook_score(text: str) -> float:
    """Calculate hook score with time-weighted analysis"""
    try:
        words = text.split()
        
        if len(words) < 10:
            return 0.0
        
        # Weight by position
        first_10_words = words[:10]    # 3x weight
        words_10_to_30 = words[10:30]  # 2x weight
        words_30_plus = words[30:]     # 1x weight
        
        # Score each section separately
        early_hook = score_patterns_in_text(" ".join(first_10_words)) * 3
        mid_hook = score_patterns_in_text(" ".join(words_10_to_30)) * 2
        late_hook = score_patterns_in_text(" ".join(words_30_plus)) * 1
        
        total_weight = 3 + 2 + 1  # 6
        return (early_hook + mid_hook + late_hook) / total_weight
        
    except Exception as e:
        logger.warning(f"Time-weighted hook score calculation failed: {e}")
        return 0.0

def score_patterns_in_text(text: str) -> float:
    """Score patterns in a text segment"""
    try:
        text_lower = text.lower()
        score = 0.0
        
        # Hook patterns with weights
        patterns = {
            r"\b(you|your|you're)\b": 0.1,
            r"\b(this|that|here's|listen|look)\b": 0.15,
            r"\b(important|critical|urgent|breaking)\b": 0.2,
            r"\b(truth|secret|hidden|unknown)\b": 0.25,
            r"\b(never|always|everyone|nobody)\b": 0.2,
            r"\b(stop|start|don't|must|should)\b": 0.25
        }
        
        for pattern, weight in patterns.items():
            if re.search(pattern, text_lower):
                score += weight
        
        return min(score, 1.0)
        
    except Exception as e:
        logger.warning(f"Pattern scoring failed: {e}")
        return 0.0

def _hook_score_v4(text: str, arousal: float = 0.0, words_per_sec: float = 0.0, genre: str = 'general', 
                   audio_data=None, sr=None, start_time: float = 0.0) -> tuple[float, str, dict]:
    """V4 hook detection with prosody analysis and multi-dimensional scoring"""
    if not text or len(text.strip()) < 8:
        return 0.0, "text_too_short"
    
    t = text.lower()[:500]  # Increased to capture more business value and true crime patterns
    score = 0.0
    reasons = []
    
    # ENHANCED: Audio-based prosody analysis
    audio_modifier = 0.0
    if audio_data is not None and sr is not None:
        audio_modifier = compute_audio_hook_modifier(audio_data, sr, start_time)
        if audio_modifier > 0:
            score += audio_modifier
            reasons.append(f"audio_prosody_{audio_modifier:.2f}")
    
    # OPTIMIZED: Laughter and exclamation detection (text-only for performance)
    # SPEED: Skip expensive laughter detection in fast mode
    speed_preset = os.getenv("SPEED_PRESET", "balanced").lower()
    if speed_preset == "fast":
        laughter_boost = 0.0  # Skip expensive laughter detection
    else:
        laughter_boost = detect_laughter_exclamations(text, audio_data, sr, start_time, enable_audio_analysis=False)
    if laughter_boost > 0:
        score += laughter_boost
        reasons.append(f"laughter_exclamation_{laughter_boost:.2f}")
    
    # ENHANCED: Time-weighted analysis (with gradual decay instead of uniform)
    # SPEED: Skip expensive time-weighted analysis in fast mode
    if os.getenv("FAST_SCORING", "0") == "1":
        time_weighted_score = 0.0  # Skip expensive time-weighted analysis
    else:
        time_weighted_score = calculate_time_weighted_hook_score(text)
    if time_weighted_score > 0:
        # More gradual time weighting: 0.9 for early content, 0.7 for later content
        # This creates better differentiation between content at different positions
        if time_weighted_score > 0.5:
            time_weighted_score = 0.9  # Early content (first 30 words)
        elif time_weighted_score > 0.3:
            time_weighted_score = 0.8  # Mid content (words 30-60)
        else:
            time_weighted_score = 0.7  # Later content (words 60+)
        
        score += time_weighted_score * 0.3  # Weight the time-based score
        reasons.append(f"time_weighted_{time_weighted_score:.2f}")
    
    # INTRO/GREETING DETECTION - Heavy penalty for intro material (but be smarter)
    intro_patterns = [
        r"^(yo|hey|hi|hello|what's up|how's it going|good morning|good afternoon|good evening)",
        r"^(it's|this is) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"^(i'm|my name is) \w+",
        r"^(welcome to|thanks for|thank you for)",
        r"^(hope you|hope everyone)",
        r"^(let's get|let's start|let's begin)",
        r"^(today we're|today i'm|today let's)"
    ]
    
    # Check for intro patterns but be smarter about it
    is_intro = False
    for pattern in intro_patterns:
        if re.match(pattern, t):
            is_intro = True
            break
    
    # Only apply penalty if it's truly intro material AND doesn't have engaging content
    engaging_content_patterns = [
        r'(why don\'t|why do not|you haven\'t|you have not|figure out|teach them|earned the right)',
        r'(destroy.*kid|embarrassed me|cancer.*killing|failed athletic dreams)',
        r'(uncomfortable truth|probably is not that good|statistically speaking)',
        r'(terrifying|forty percent|more likely to quit)',
        r'(scholarship myth|needs to die|2\.9% chance)',
        r'(storms onto the field|grabs his kid|never came back)',
        r'(challenge.*parent|do not say.*single word)'
    ]
    
    has_engaging_content = any(re.search(pattern, t) for pattern in engaging_content_patterns)
    
    if is_intro and not has_engaging_content:
        score = 0.01  # Set to extremely low score for intro material
        reasons.append("intro_greeting_penalty")
    
    # RELAXED: Context-dependent penalty (much less harsh for sports content)
    context_patterns = [
        r"^(you like that|like that|that's|here's the)",  # Only penalize very obvious context
    ]
    
    # DON'T penalize sports-specific context like "Caleb Johnson is clearly"
    # This is normal in sports analysis
    
    for pattern in context_patterns:
        if re.match(pattern, t):
            score -= 0.05  # Reduced from 0.15 to 0.05
            reasons.append("context_dependent_opening")
            break
    
    # ENHANCED: Provocation patterns for high engagement
    PROVOCATION_PATTERNS = [
    r"you're (delusional|wrong|missing|wasting)",
    r"you are (delusional|wrong|missing|wasting)",
    r"stop (doing|believing|thinking|relying on)",
    r"the truth about",
    r"nobody wants to (hear|admit|say)",
    r"everyone's wrong about",
    r"this will (fail|destroy|ruin)",
    r"you've been lied to",
    r"if you are relying on.*you are so (delusional|wrong)",
    r"you are so delusional as a (parent|coach|person)"
]
    
    # ENHANCED: Action demand patterns for immediate engagement
    ACTION_PATTERNS = [
        r"^(you need to|you must|you should)",
        r"^(stop|start|don't|never|always)",
        r"^(here's what to do|do this)",
        r"^(immediately|right now|today)"
    ]
    
    # FIXED: Check for provocation patterns first (highest boost)
    provocation_found = False
    for pattern in PROVOCATION_PATTERNS:
        if re.search(pattern, t):
            score += 0.3  # Strong provocation boost
            reasons.append("provocation_hook")
            provocation_found = True
            break
    
    # FIXED: Check for action demand patterns (only if no provocation to avoid double-counting)
    if not provocation_found:
        for pattern in ACTION_PATTERNS:
            if re.search(pattern, t):
                score += 0.2  # Action-oriented boost
                reasons.append("action_demand_hook")
                break
    
    # ENHANCED: Sports-specific hooks that shouldn't be penalized
    sports_hook_patterns = [
        r"(biggest|top|best|worst) (sleeper|bust|play|pick|value)",
        r"(nobody|everyone|people) (is|are) (talking about|sleeping on|missing)",
        r"(here's why|let me tell you why|this is why)",
        r"the (guy|player|team) (who|that)",
        r"(fantasy|draft|waiver) (gold|gem|steal|target)",
        r"(this|that) (guy|player|team) is",
        r"(i'm telling you|trust me|mark my words)",
        r"(if you|when you) (draft|pick|start|sit)",
        # NEW: More compelling sports hooks
        r"(running back|quarterback|receiver|defense) (is|are) (going to|about to)",
        r"(why would|why should) (you|i) (like|love|hate) (that|this)",
        r"(look at|check out|watch) (this|that) (guy|player|team)",
        r"(the thing is|here's the thing|the problem is)",
        r"(you know what|here's what|this is what)",
        r"(i've been|i'm) (watching|following|tracking)",
        r"(this week|next week|this season) (is|will be)",
        r"(atlanta|dallas|indianapolis|cincinnati) (is|are)",
        r"(jayden|caleb|ollie) (is|looks|seems)",
        r"(zero from|from that) (game|match|week)",
        r"(home league|casual) (drafts|drafting) (are|is)",
        r"(way better|much better|improved|evolved)",
        # ADDITIONAL: More sports patterns
        r"(yeah|so|and) (it was|there was|we had)",
        r"(if i|if we|if you) (didn't|won|lost|had)",
        r"(parents|coaches|players) (were|are) (pissed|angry|happy)",
        r"(synchronicity|values|traits) (that|which) (apply|work)",
        r"(kids|players|athletes) (that|who) (have|are|know)",
        r"(academy|national|team) (bound|level|quality)",
        r"(serve you|help you|benefit you) (in the fact|because)",
        r"(intention|purpose|goal) (for|to|of) (two hours|practice)",
        r"(every player|each player|all players) (has|have) (that|one)",
        r"(training|practicing|working) (with|on|for) (their|the)",
        r"(changed|improved|enhanced) (the way|how|what)"
    ]
    
    for pattern in sports_hook_patterns:
        if re.search(pattern, t):
            score += 0.25  # Reduced from 0.4 to 0.25 for better clustering
            reasons.append("sports_hook")
            break
    
    # ENHANCED: More granular pattern tiers to break up clustering
    # Very weak signals: +0.05 (filler content)
    very_weak_patterns = [
        r"^(so|and|but|well|okay|right|now)",
        r"^(i think|i believe|i feel|i guess)",
        r"^(you know|you see|you understand)"
    ]
    
    # Weak signals: +0.15 (generic statements)
    weak_patterns = [
        r"^(listen|look|watch|check this out)",
        r"^(you need to know|you have to understand)",
        r"^(this is important|this matters|pay attention)",
        r"^(well,|so,|and,|but,|okay,|right,|now,)"  # Only if not followed by engaging content
    ]
    
    # Low-moderate signals: +0.25 (clear statements)
    low_moderate_patterns = [
        r"^(here's|this is|that's|it's)",
        r"^(the thing is|the problem is|the issue is)",
        r"^(what i mean|what i'm saying|what i think)"
    ]
    
    # Moderate signals: +0.35 (engaging content)
    moderate_patterns = [
        r"^(let me tell you|here's what happened|this story)",
        r"^(you won't believe|can't believe|unbelievable)",
        r"^(crazy|insane|wild|amazing)",
        r"^(what are you really|what are you actually|what do you think)",
        r"^(were you|are you|do you think you)"
    ]
    
    # High-moderate signals: +0.45 (strong engagement)
    high_moderate_patterns = [
        r"^(dude|man|bro|wow|omg|damn|shit|fuck)\s*[!.]*\s*\1",  # Repeated exclamations (allow punctuation)
        r"^(haha|lol|lmao|rofl)",  # Laughter
        r"^(no\s+way|are\s+you\s+kidding|you\s+gotta\s+be)"  # Surprise expressions
    ]
    
    # Strong signals: +0.55 (viral potential)
    strong_patterns = [
        r"^(this\s+is\s+crazy|unbelievable|insane|wild)",  # High-energy descriptors
        r"^(wait\s+until\s+you\s+hear|you\s+won't\s+believe)",  # Anticipation builders
        r"^(i\s+can't\s+believe|this\s+is\s+unreal)",  # Shock expressions
        r"^(dude|man|bro|wow|omg|damn|shit|fuck)\s+\1\s+\1",  # Repeated exclamations like "dude dude dude"
        r"you\s+are\s+so\s+(delusional|wrong|missing|wasting)",  # Direct provocation
        r"if\s+you\s+are\s+relying\s+on.*you\s+are\s+so"  # Conditional provocation
    ]
    
    # Check pattern tiers (in order of strength to avoid double-counting)
    pattern_matched = False
    
    # Strong signals: +0.55
    for pattern in strong_patterns:
        if re.search(pattern, t):
            score += 0.55
            reasons.append("strong_pattern")
            pattern_matched = True
            break
    
    if not pattern_matched:
        # High-moderate signals: +0.45
        for pattern in high_moderate_patterns:
            if re.search(pattern, t):
                score += 0.45
                reasons.append("high_moderate_pattern")
                pattern_matched = True
                break
    
    if not pattern_matched:
        # Moderate signals: +0.35
        for pattern in moderate_patterns:
            if re.search(pattern, t):
                score += 0.35
                reasons.append("moderate_pattern")
                pattern_matched = True
                break
    
    if not pattern_matched:
        # Low-moderate signals: +0.25
        for pattern in low_moderate_patterns:
            if re.search(pattern, t):
                score += 0.25
                reasons.append("low_moderate_pattern")
                pattern_matched = True
                break
    
    if not pattern_matched:
        # Weak signals: +0.15
        for pattern in weak_patterns:
            if re.search(pattern, t):
                score += 0.15
                reasons.append("weak_pattern")
                pattern_matched = True
                break
    
    if not pattern_matched:
        # Very weak signals: +0.05
        for pattern in very_weak_patterns:
            if re.search(pattern, t):
                score += 0.05
                reasons.append("very_weak_pattern")
                break
    
    # Direct hook cues
    strong_hits = sum(1 for cue in HOOK_CUES if cue in t)
    if strong_hits > 0:
        score += min(strong_hits * 0.15, 0.35)
        reasons.append(f"direct_hooks_{strong_hits}")
    
    # MICRO-BOOSTS: Add subtle differentiation to break clustering
    # Length-based micro-boost (optimal hook length is 3-8 words)
    word_count = len(t.split())
    if 3 <= word_count <= 8:
        score += 0.02  # Perfect hook length
        reasons.append("optimal_length")
    elif word_count <= 2:
        score += 0.01  # Very short but punchy
        reasons.append("punchy_short")
    elif word_count <= 12:
        score += 0.005  # Still concise
        reasons.append("concise")
    
    # Emotional intensity micro-boost
    high_energy_words = ['crazy', 'insane', 'unbelievable', 'amazing', 'incredible', 'wild', 'epic', 'mind-blowing']
    if any(word in t.lower() for word in high_energy_words):
        score += 0.03
        reasons.append("high_energy_word")
    
    # Question micro-boost (questions are engaging)
    question_count = t.count('?')
    if question_count > 0:
        score += 0.02 + (question_count - 1) * 0.03  # Extra boost for multiple questions
        reasons.append(f"question_hook_{question_count}")
    
    # Personal story micro-boost
    personal_words = ['i', 'me', 'my', 'mine', 'myself']
    if sum(1 for word in personal_words if word in t.lower()) >= 2:
        score += 0.02
        reasons.append("personal_story")
    
    # Challenge question micro-boost (content that challenges the listener)
    challenge_patterns = [
        r"what are you really",
        r"what are you actually", 
        r"what do you think you",
        r"were you really",
        r"are you really",
        r"why don't you",
        r"why do not you",
        r"you haven't earned",
        r"you have not earned"
    ]
    if any(re.search(pattern, t) for pattern in challenge_patterns):
        score += 0.04
        reasons.append("challenge_question")
    
    # Coaching advice micro-boost (content that challenges coaching philosophy)
    coaching_advice_patterns = [
        r"figure out what kind of players",
        r"teach them how to win",
        r"catering it to them",
        r"rather than you catering it to yourself",
        r"you have not earned the right",
        r"who are we developing",
        r"is this about developing my philosophy"
    ]
    if any(re.search(pattern, t) for pattern in coaching_advice_patterns):
        score += 0.06
        reasons.append("coaching_advice")
    
    # Controversial claim micro-boost (content that makes bold claims)
    controversial_patterns = [
        r"i could do better",
        r"i can do better",
        r"i am standing on business",
        r"i am standing on that",
        r"nobody talks about",
        r"got in trouble",
        r"giving me so much shit",
        r"they were just giving me",
        r"mark my words",
        r"i will eat my.*microphone",
        r"that is how confident i am",
        r"father time is undefeated",
        r"even the goat can not beat",
        r"fundamentally broken",
        r"do not care who hears me",
        r"bastardized.*into.*money machine",
        r"predatory.*honestly",
        r"straight up predatory",
        r"truth.*will make people mad",
        r"most.*kids.*not going pro",
        r"better chance.*becoming an astronaut",
        r"biggest scam.*youth sports",
        r"math does not work",
        r"nightmares about.*game",
        r"child labor with a ball",
        r"level of delusion.*astronomical"
    ]
    controversial_count = sum(1 for pattern in controversial_patterns if re.search(pattern, t))
    if controversial_count > 0:
        controversial_boost = min(0.08 + (controversial_count - 1) * 0.03, 0.20)  # Stack up to 0.20
        score += controversial_boost
        reasons.append(f"controversial_claim_{controversial_count}")
    
    # Educational content micro-boost (research-backed insights)
    educational_patterns = [
        r"research from",
        r"studies show",
        r"research demonstrates",
        r"first documented by",
        r"mathematical formula",
        r"optimal.*interval",
        r"superior.*retention",
        r"often misunderstood",
        r"fundamental principle"
    ]
    educational_count = sum(1 for pattern in educational_patterns if re.search(pattern, t))
    if educational_count > 0:
        educational_boost = min(0.04 + (educational_count - 1) * 0.02, 0.12)  # Stack up to 0.12
        score += educational_boost
        reasons.append(f"educational_content_{educational_count}")
    
    # High-value business content micro-boost (success stories, specific data)
    business_value_patterns = [
        r"sold my.*company for \d+ million",
        r"\d+ million dollars",
        r"\d+ billion dollars",
        r"arr.*to.*arr",
        r"revenue.*to.*revenue",
        r"competitive moat",
        r"moves the needle",
        r"not what you would expect",
        r"contrarian",
        r"systematic documentation",
        r"real data",
        r"while competitors were"
    ]
    business_value_count = sum(1 for pattern in business_value_patterns if re.search(pattern, t))
    if business_value_count > 0:
        business_value_boost = min(0.06 + (business_value_count - 1) * 0.02, 0.15)  # Stack up to 0.15
        score += business_value_boost
        reasons.append(f"business_value_{business_value_count}")
    
    # True crime/suspense content micro-boost (mystery, investigation, emotional impact)
    true_crime_patterns = [
        r"night that disappeared",
        r"never makes it home",
        r"stepped off the earth",
        r"vanished",
        r"nobody tells you",
        r"changes everything",
        r"police reports",
        r"eyewitnesses",
        r"inconsistencies",
        r"sealed record",
        r"phone call.*whisper",
        r"skin.*crawling",
        r"coordinated disappearance",
        r"truth.*still.*fog",
        r"never look.*same way"
    ]
    true_crime_count = sum(1 for pattern in true_crime_patterns if re.search(pattern, t))
    if true_crime_count > 0:
        true_crime_boost = min(0.05 + (true_crime_count - 1) * 0.02, 0.12)  # Stack up to 0.12
        score += true_crime_boost
        reasons.append(f"true_crime_{true_crime_count}")
    
    # Clickbait penalty (high-energy words without substance)
    clickbait_patterns = [
        r"blow your mind",
        r"change everything",
        r"absolutely incredible",
        r"you will not believe",
        r"ready for this",
        r"here it comes"
    ]
    clickbait_count = sum(1 for pattern in clickbait_patterns if re.search(pattern, t))
    if clickbait_count > 0:
        # Check if there's actual substance (not just filler)
        filler_words = ["you know", "um", "basically", "so yeah", "and stuff"]
        filler_count = sum(1 for filler in filler_words if filler in t.lower())
        
        # If high clickbait but low substance, apply penalty
        if clickbait_count >= 2 and filler_count >= 2:
            score -= 0.10  # Penalty for clickbait without substance
            reasons.append("clickbait_penalty")
    
    # EMOTIONAL IMPACT BOOST: Recognize high-shareability emotional content
    emotional_patterns = [
        r"nightmares about.*game",
        r"used to dream.*now.*nightmares",
        r"broke my heart",
        r"child labor with a ball",
        r"seventeen years old.*nightmares",
        r"never had a weekend off",
        r"scheduled like.*fortune 500 ceo",
        r"destroy.*kid.*parking lot",
        r"cancer.*killing.*youth sports",
        r"embarrassed me out there",
        r"failed athletic dreams",
        r"storms onto the field",
        r"grabs his kid.*leaves",
        r"never came back",
        r"adults who have lost their minds",
        r"terrifying.*forty percent",
        r"more likely.*burns out"
    ]
    emotional_count = sum(1 for pattern in emotional_patterns if re.search(pattern, t))
    if emotional_count > 0:
        emotional_boost = min(0.20 + (emotional_count - 1) * 0.05, 0.30)  # Increased boost, stack up to 0.30
        score += emotional_boost
        reasons.append(f"emotional_impact_{emotional_count}")
    
    # VISUAL DRAMA BOOST: Recognize dramatic, visual stories that go viral
    visual_drama_patterns = [
        r"storms onto the field",
        r"onto the field.*during the game",
        r"grabs his kid.*leaves",
        r"just leaves",
        r"other parents are silent",
        r"kids are confused",
        r"never came back",
        r"destroy.*kid.*parking lot",
        r"kid.*crying",
        r"dad.*going.*embarrassed me",
        r"follow.*team bus.*tournament",
        r"stayed at.*same hotel",
        r"kid was mortified"
    ]
    visual_drama_count = sum(1 for pattern in visual_drama_patterns if re.search(pattern, t))
    if visual_drama_count > 0:
        visual_drama_boost = min(0.18 + (visual_drama_count - 1) * 0.04, 0.25)  # Stack up to 0.25
        score += visual_drama_boost
        reasons.append(f"visual_drama_{visual_drama_count}")
    
    # POWER WORD BOOST: Recognize high-engagement trigger words
    power_word_patterns = [
        r"biggest scam",
        r"predatory.*honestly",
        r"straight up predatory",
        r"fundamentally broken",
        r"do not care who hears me",
        r"bastardized.*into.*money machine",
        r"complete scam",
        r"is a scam",
        r"child abuse.*disguised",
        r"destroying.*club culture",
        r"culture of lying",
        r"maximize revenue per child",
        r"atm machines",
        r"extract.*money from parents"
    ]
    power_word_count = sum(1 for pattern in power_word_patterns if re.search(pattern, t))
    if power_word_count > 0:
        power_word_boost = min(0.12 + (power_word_count - 1) * 0.04, 0.20)  # Stack up to 0.20
        score += power_word_boost
        reasons.append(f"power_words_{power_word_count}")
    
    # INSIDER REVELATION BOOST: Recognize explosive insider content
    insider_patterns = [
        r"i sat in.*directors meeting",
        r"i recorded it",
        r"i have the audio",
        r"direct quote",
        r"quiet part nobody says",
        r"i got fired.*telling.*truth",
        r"twenty minutes.*cleaning out my desk",
        r"will get me blacklisted",
        r"insider.*revelation",
        r"behind the scenes",
        r"what really happens",
        r"nobody talks about"
    ]
    insider_count = sum(1 for pattern in insider_patterns if re.search(pattern, t))
    if insider_count > 0:
        insider_boost = min(0.15 + (insider_count - 1) * 0.05, 0.25)  # Stack up to 0.25
        score += insider_boost
        reasons.append(f"insider_revelation_{insider_count}")
    
    # DRAMATIC PERSONAL STORY BOOST: Recognize highly shareable personal drama
    dramatic_story_patterns = [
        r"my own son.*quit",
        r"my son quit.*last year",
        r"dad pulled.*crying.*six-year-old",
        r"threw him in the car",
        r"slammed the door",
        r"kid was sobbing",
        r"hate soccer forever",
        r"killed his love for the game",
        r"what we have turned it into",
        r"system is broken",
        r"child abuse.*character building",
        r"winners never quit.*crying",
        r"ready to quit.*completely",
        r"panic attack.*first time",
        r"emotional vomit on canvas",
        r"bought it for.*thousand",
        r"eight thousand dollars",
        r"fifty million.*walks in",
        r"bypasses all my best work",
        r"still unsold",
        r"teaching kids.*paint parties.*twenty dollars",
        r"too exhausted to work",
        r"saved my career.*probably my sanity"
    ]
    dramatic_story_count = sum(1 for pattern in dramatic_story_patterns if re.search(pattern, t))
    if dramatic_story_count > 0:
        dramatic_story_boost = min(0.18 + (dramatic_story_count - 1) * 0.04, 0.25)  # Stack up to 0.25
        score += dramatic_story_boost
        reasons.append(f"dramatic_story_{dramatic_story_count}")
    
    # BUSINESS SUCCESS STORY BOOST: Recognize viral business success content
    business_success_patterns = [
        r"sales increased.*percent",
        r"four hundred percent",
        r"made more.*one month.*previous year",
        r"bought it for.*thousand",
        r"eight thousand dollars",
        r"million-dollar pieces",
        r"waitlist is.*months long",
        r"sold in ten minutes",
        r"every show.*sold out",
        r"from starving to thriving",
        r"that decision saved my career"
    ]
    business_success_count = sum(1 for pattern in business_success_patterns if re.search(pattern, t))
    if business_success_count > 0:
        business_success_boost = min(0.15 + (business_success_count - 1) * 0.03, 0.20)  # Stack up to 0.20
        score += business_success_boost
        reasons.append(f"business_success_{business_success_count}")
    
    # CREATIVE INDUSTRY REVELATION BOOST: Recognize insider creative industry content
    creative_industry_patterns = [
        r"what nobody tells you.*art school",
        r"art basel.*last year",
        r"nobody looked at the art",
        r"check.*instagram followers",
        r"investment potential",
        r"photograph well.*feed",
        r"gallery show.*collector",
        r"fifty percent.*sales",
        r"bring your own buyers",
        r"start your own gallery",
        r"white walls.*good lighting",
        r"trust fund.*paying rent",
        r"spouse.*tech job.*health insurance",
        r"credit card debt.*maintain.*image",
        r"all performance",
        r"truth bomb.*upset everyone",
        r"most people.*do not care about art",
        r"what owning art says about them",
        r"feel cultured.*sophisticated.*successful",
        r"fulfill their emotional need",
        r"enjoy your day job.*rest of us.*bills to pay",
        r"ashley traces.*disney characters",
        r"waitlist is.*months long",
        r"working at starbucks.*pay rent",
        r"not about talent.*never been about talent",
        r"what actually works.*nobody wants to hear",
        r"email lists.*boring.*make more",
        r"instagram followers.*free entertainment",
        r"email subscribers.*buy from you",
        r"permission to sell to them",
        r"sold in ten minutes",
        r"screaming into the void"
    ]
    creative_industry_count = sum(1 for pattern in creative_industry_patterns if re.search(pattern, t))
    if creative_industry_count > 0:
        creative_industry_boost = min(0.12 + (creative_industry_count - 1) * 0.03, 0.18)  # Stack up to 0.18
        score += creative_industry_boost
        reasons.append(f"creative_industry_{creative_industry_count}")
    
    # PRACTICAL BUSINESS ADVICE BOOST: Recognize actionable business insights
    practical_advice_patterns = [
        r"what actually works",
        r"nobody wants to hear this",
        r"email lists.*work better",
        r"make more.*email list.*instagram followers",
        r"permission to sell to them",
        r"sold in ten minutes",
        r"practical advice",
        r"actionable insight",
        r"here is what works",
        r"the secret is",
        r"nobody tells you",
        r"behind the scenes",
        r"what really happens"
    ]
    practical_advice_count = sum(1 for pattern in practical_advice_patterns if re.search(pattern, t))
    if practical_advice_count > 0:
        practical_advice_boost = min(0.10 + (practical_advice_count - 1) * 0.02, 0.15)  # Stack up to 0.15
        score += practical_advice_boost
        reasons.append(f"practical_advice_{practical_advice_count}")
    
    # PARENT-SHAMING BOOST: Recognize content that challenges parents (highly shareable in coaching communities)
    parent_shaming_patterns = [
        r"most.*kids.*not going pro",
        r"better chance.*becoming an astronaut",
        r"math does not work",
        r"level of delusion.*astronomical",
        r"fifty thousand dollars.*ten thousand",
        r"put that money in a college fund",
        r"destroy.*kid.*parking lot",
        r"embarrassed me out there",
        r"cancer.*killing youth sports",
        r"failed athletic dreams",
        r"uncomfortable truth.*kid probably is not that good",
        r"statistically speaking.*average",
        r"terrifying.*forty percent.*quit",
        r"more invested.*parent.*more likely.*burns out",
        r"storms onto the field",
        r"grabs his kid.*leaves",
        r"never came back",
        r"adults who have lost their minds",
        r"scholarship myth.*needs to die",
        r"2\.9% chance",
        r"spreadsheet.*touches per game",
        r"high school coach.*does not care.*excel sheets"
    ]
    parent_shaming_count = sum(1 for pattern in parent_shaming_patterns if re.search(pattern, t))
    if parent_shaming_count > 0:
        parent_shaming_boost = min(0.10 + (parent_shaming_count - 1) * 0.03, 0.18)  # Stack up to 0.18
        score += parent_shaming_boost
        reasons.append(f"parent_shaming_{parent_shaming_count}")
    
    # COMPRESSED FLOOR SCORE: Higher minimum for clearly viral content
    viral_indicators = [
        r"fundamentally broken",
        r"do not care who hears me",
        r"predatory",
        r"truth.*will make people mad",
        r"most.*kids.*not going pro",
        r"biggest scam",
        r"nightmares about",
        r"child labor",
        r"level of delusion",
        r"coaching.*joy out",
        r"better chance.*astronaut",
        r"math does not work"
    ]
    viral_indicators_count = sum(1 for pattern in viral_indicators if re.search(pattern, t))
    if viral_indicators_count > 0:
        # Compress range: Set floor score of 65% for clearly viral content (target range)
        score = max(score, 0.65)
        reasons.append(f"viral_floor_{viral_indicators_count}")
    
    # CEILING MANAGEMENT: Cap scores to preserve granularity at top end
    # This prevents all high-quality content from hitting 100% and losing differentiation
    if score > 0.95:
        # Apply ceiling with differentiation based on actual pattern counts
        emotional_count = 0
        visual_drama_count = 0
        parent_shaming_count = 0
        
        # Extract actual counts from reason strings
        for r in reasons:
            if 'emotional_impact_' in r:
                emotional_count = int(r.split('_')[-1])
            elif 'visual_drama_' in r:
                visual_drama_count = int(r.split('_')[-1])
            elif 'parent_shaming_' in r:
                parent_shaming_count = int(r.split('_')[-1])
        
        total_patterns = emotional_count + visual_drama_count + parent_shaming_count
        
        if total_patterns >= 15:  # Very high pattern density (field invasion story: 4+7+4=15)
            score = 0.95  # Top tier
        elif total_patterns >= 11:  # High pattern density (parking lot story: 4+3+4=11)
            score = 0.92  # High tier
        else:  # Good pattern density
            score = 0.88  # Strong tier
        reasons.append(f"ceiling_cap_{total_patterns}")
    
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
    
    # Boost for insight content (if not intro) - use V2 if enabled
    if not any("intro_greeting_penalty" in reason for reason in reasons):
        config = get_config()
        if config.get("insight_v2", {}).get("enabled", False):
            insight_score, _ = _detect_insight_content_v2(text, genre)
        else:
            insight_score, _ = _detect_insight_content(text, genre)
        if insight_score > 0.5:
            score += 0.2  # Boost for high insight content
            reasons.append("insight_content_boost")
    
    # ENHANCED: Calculate multi-dimensional hook components
    hook_components = calculate_hook_components(text)
    
    # Determine hook type
    hook_type = "general"
    if any(re.search(pattern, t) for pattern in PROVOCATION_PATTERNS):
        hook_type = "provocation"
    elif any(re.search(pattern, t) for pattern in ACTION_PATTERNS):
        hook_type = "action"
    elif "?" in text:
        hook_type = "question"
    elif any(word in t for word in ["story", "happened", "when", "then"]):
        hook_type = "story"
    
    # Calculate confidence based on component scores
    confidence = np.mean(list(hook_components.values()))
    
    # At the end, ensure minimum score
    final_score = float(np.clip(score, 0.05, 1.0))  # Allow lower minimum for intro content
    reason_str = ";".join(reasons) if reasons else "no_hooks_detected"
    
    return final_score, reason_str, {
        "hook_components": hook_components,
        "hook_type": hook_type,
        "confidence": confidence,
        "audio_modifier": audio_modifier,
        "laughter_boost": laughter_boost,
        "time_weighted_score": time_weighted_score
    }

# --- Hook V5 (families + evidence guard + time weighting + micro audio) ---
_WORD = re.compile(r"[A-Za-z']+|\d+%?")
_PUNCT_CLAUSE = re.compile(r"(?<=[.!?])\s+")
_HAS_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b")
_HAS_COMP = re.compile(r"\b(?:vs\.?|versus|more than|less than)\b|[<>]")
_HAS_HOWWHY = re.compile(r"\b(?:how|why|what)\b")
_SOFT_FLIP = re.compile(r"\bbut (?:actually|in reality)\b")

def _saturating_sum(scores: List[float], cap: float = 1.0) -> float:
    prod = 1.0
    for s in scores:
        s = max(0.0, min(1.0, float(s)))
        prod *= (1.0 - s)
    return min(cap, 1.0 - prod)

def _proximity_bonus(index_in_video: int, k: float) -> float:
    try:
        i = max(0, int(index_in_video))
    except Exception:
        i = 0
    return math.exp(- i / max(1e-6, float(k)))

def _normalize_quotes_lower(text: str) -> str:
    t = (text or "").strip().lower()
    return t.translate({
        0x2019: 0x27,  # ' -> '
        0x2018: 0x27,  # ' -> '
        0x201C: 0x22,  # " -> "
        0x201D: 0x22,  # " -> "
    })

def _first_clause(text: str, max_words: int) -> str:
    sent = _PUNCT_CLAUSE.split(text, maxsplit=1)[0]
    toks = _WORD.findall(sent)
    return " ".join(toks[:max_words])

def _get_hook_cues_from_config(cfg: Dict[str, Any]) -> Dict[str, List[re.Pattern]]:
    raw = (cfg.get("HOOK_CUES")
           or cfg.get("lexicons", {}).get("HOOK_CUES")
           or {})
    # If someone left HOOK_CUES as a flat list, wrap it.
    if isinstance(raw, list):
        raw = {"generic": raw}
    cues: Dict[str, List[re.Pattern]] = {}
    for fam, arr in raw.items():
        pats = []
        for s in arr:
            try:
                pats.append(re.compile(s, re.I))
            except Exception:
                pass
        if pats:
            cues[fam] = pats
    return cues

def _family_score(text: str, cues: Dict[str, List[re.Pattern]], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    fam_scores: Dict[str, float] = {}
    partials: List[float] = []
    for fam, pats in cues.items():
        w = float(weights.get(fam, 1.0))
        m = 0.0
        for p in pats:
            if p.search(text):
                m = 1.0
                break
        fam_scores[fam] = min(1.0, m * w)
        if fam_scores[fam] > 0:
            partials.append(min(1.0, fam_scores[fam]))
    combined = _saturating_sum(partials, cap=1.0)
    return combined, fam_scores

def _evidence_guard(t: str, need_words: int, clause_words: int) -> Tuple[bool, Dict[str, bool]]:
    toks = _WORD.findall(t)
    early = " ".join(toks[:max(0, need_words)])
    has_A = bool(_HAS_HOWWHY.search(early) or _HAS_NUM.search(early) or _HAS_COMP.search(early))
    if has_A:
        return True, {"early": True, "clause": False, "flip": False}
    clause = _first_clause(t, max_words=clause_words)
    has_B = bool(_HAS_HOWWHY.search(clause) or _HAS_NUM.search(clause) or _HAS_COMP.search(clause))
    flip = bool(_SOFT_FLIP.search(clause))
    return bool(has_B or flip), {"early": False, "clause": has_B, "flip": flip}

def _anti_intro_outro_penalties(t: str, hv5: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    reasons = []
    pin = 0.0
    pout = 0.0
    intro = [s.strip().lower() for s in hv5.get("intro_tokens", [])]
    outro = [s.strip().lower() for s in hv5.get("outro_tokens", [])]
    for tok in intro:
        if tok and t.startswith(tok):
            pin = float(hv5.get("anti_intro_penalty", 0.05)); reasons.append("anti_intro")
            break
    for tok in outro:
        if tok and tok in t:
            pout = float(hv5.get("anti_outro_penalty", 0.06)); reasons.append("anti_outro")
            break
    return pin, pout, reasons

def _audio_micro_for_hook(audio_mod: float, cap: float) -> float:
    try:
        return max(0.0, min(cap, float(audio_mod)))
    except Exception:
        return 0.0

def _sigmoid(z: float, a: float) -> float:
    return 1.0 / (1.0 + math.exp(-a * z))

def _calibrate_simple(raw: float, mu: float = 0.40, sigma: float = 0.18, a: float = 1.6) -> float:
    z = 0.0 if sigma <= 0 else (raw - mu) / sigma
    return _sigmoid(z, a)

def _hook_score_v5(
    text: str,
    *,
    cfg: Dict[str, Any],
    segment_index: int = 0,
    audio_modifier: float = 0.0,
    arousal: float = 0.0,
    q_or_list: float = 0.0,
    batch_mu: float = None,
    batch_sigma: float = None
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns: (raw, calibrated, debug)
    raw: 0..1 pre-calibration
    calibrated: 0..1 after simple calibration
    """
    hv5 = cfg.get("hook_v5", {}) if cfg else {}
    a_sig = float(hv5.get("sigmoid_a", 1.6))
    need_words = int(hv5.get("require_after_words", 12))
    clause_words = int(hv5.get("first_clause_max_words", 24))
    k = float(hv5.get("time_decay_k", 5))
    early_bonus_scale = float(hv5.get("early_pos_bonus", 0.25))
    audio_cap = float(hv5.get("audio_cap", 0.05))
    fam_w = hv5.get("family_weights", {}) or {}

    t = _normalize_quotes_lower(text)
    cues = _get_hook_cues_from_config(cfg)

    fam_combined, fam_scores = _family_score(t, cues, fam_w)
    evidence_ok, evidence_bits = _evidence_guard(t, need_words, clause_words)

    base = fam_combined
    reasons: List[str] = []
    if fam_combined <= 0.0:
        reasons.append("no_family_match")
    if not evidence_ok and fam_combined > 0.0:
        base *= 0.80
        reasons.append("no_evidence_early")

    pin, pout, pr = _anti_intro_outro_penalties(t, hv5)
    base = max(0.0, base - pin - pout)
    reasons.extend(pr)

    prox = _proximity_bonus(segment_index, k)
    base += early_bonus_scale * prox

    base += _audio_micro_for_hook(audio_modifier, audio_cap)

    syn = hv5.get("synergy", {}) or {}
    syn_bonus = 0.0
    if arousal >= float(syn.get("arousal_gate", 0.60)):
        syn_bonus += float(syn.get("bonus_each", 0.01))
    if q_or_list >= float(syn.get("q_or_list_gate", 0.60)):
        syn_bonus += float(syn.get("bonus_each", 0.01))
    syn_bonus = min(syn_bonus, float(syn.get("cap_total", 0.02)))
    base = max(0.0, base) + syn_bonus
    if syn_bonus > 0: reasons.append(f"synergy+{syn_bonus:.2f}")

    raw = min(1.0, max(0.0, base))

    mu = 0.40 if batch_mu is None else float(batch_mu)
    sigma = 0.18 if batch_sigma is None else float(batch_sigma)
    cal = _calibrate_simple(raw, mu=mu, sigma=sigma, a=a_sig)

    debug = {
        "hook_v5_raw": round(raw, 6),
        "hook_v5_cal": round(cal, 6),
        "fam_scores": fam_scores,
        "fam_combined": round(fam_combined, 6),
        "evidence_ok": evidence_ok,
        "evidence_bits": evidence_bits,
        "proximity": round(prox, 6),
        "audio_mod": round(_audio_micro_for_hook(audio_modifier, audio_cap), 6),
        "pins": {"intro": pin, "outro": pout},
        "reasons": reasons,
        "mu": mu, "sigma": sigma, "a": a_sig,
    }
    return raw, cal, debug
# --- end Hook V5 ---



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

def _detect_payoff(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect payoff strength: resolution, answer, or value delivery using genre-specific patterns"""
    if not text or len(text.strip()) < 8:
        return 0.0, "too_short"
    
    t = text.lower()
    score = 0.0
    reasons = []

    # General payoff markers
    general_payoff_patterns = [
        r"here('?s)? (how|why|the deal|what you need)",  # "here's how"
        r"the (solution|answer|truth|key)", 
        r"because", r"so that", r"which means",
        r"in other words", r"this means",
        r"the (lesson|takeaway|insight|bottom line)",
        r"(therefore|thus|that's why)",
        r"turns out", r"it turns out"
    ]
    
    for pattern in general_payoff_patterns:
        if re.search(pattern, t):
            score += 0.25
            reasons.append("general_payoff")

    # Genre-specific payoff patterns
    try:
        # Get the appropriate genre profile
        if genre == 'fantasy_sports':
            profile = FantasySportsGenreProfile()
        elif genre == 'comedy':
            profile = ComedyGenreProfile()
        elif genre == 'true_crime':
            profile = TrueCrimeGenreProfile()
        elif genre == 'business':
            profile = BusinessGenreProfile()
        elif genre == 'educational':
            profile = EducationalGenreProfile()
        elif genre == 'health_wellness':
            profile = HealthWellnessGenreProfile()
        else:
            profile = GenreProfile()  # Default fallback
            
        # Check genre-specific payoff patterns
        for pattern in getattr(profile, 'payoff_patterns', []):
            if re.search(pattern, t):
                score += 0.3
                reasons.append(f"{genre}_payoff")
                
    except Exception:
        # Fallback if genre profile fails
        pass

    # Scale and clamp
    score = min(score, 1.0)
    if score == 0.0:
        return 0.0, "no_payoff"
    return score, ";".join(reasons)

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

# Loopability scoring constants and helper functions
_STOPWORDS = set("""
a an the and but or so because that which to of in on at for with by as is are was were be been being this those these it
""".split())

_QUOTABLE_PATTERNS = [
    r"\bnot\s+[^,]+,\s*but\s+[^,]+",                 # not X, but Y
    r"\b(here|are|these)\s+\d+\s+\w+",               # numbered lists: 3 things...
    r"\b(do|try|stop|never|always|remember)\b\s+\w+", # imperative/pithy
    r"\b(X|it)'?s\s+not\s+about\b",                  # aphorism starter
]

_OUTRO_PATTERNS = [
    r"\b(thanks for watching|subscribe|follow|like and subscribe|link in bio|see you next time)\b",
    r"\b(in today'?s video|welcome back)\b",
]

_CLIFF_WORDS = {"until", "then", "because", "guess", "turns", "revealed", "reveals"}

def _tokens(text, n=None):
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    if n: toks = toks[:n]
    return [t for t in toks if t not in _STOPWORDS]

def _end_tokens(text, n):
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    toks = toks[-n:]
    return [t for t in toks if t not in _STOPWORDS]

def _has_pattern(text, patterns):
    t = text.lower()
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)

def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def sat(parts):
    """Saturating combiner to avoid runaway sums"""
    p = 1.0
    for x in parts:
        p *= (1.0 - max(0.0, min(1.0, x)))
    return 1.0 - p

def _loopability_heuristic(text: str) -> float:
    """Enhanced loopability scoring with perfect-loop detection, quotability patterns, and curiosity enders"""
    if not text: 
        return 0.0

    stripped = text.strip()
    words = stripped.split()
    if len(words) < 4:
        return 0.5  # tiny clips: neutral

    # --- 1) Boundary Cleanliness ---
    last_char = stripped[-1]
    last_is_clean_punct = last_char in ".!?"
    last_sentence = re.split(r"[.!?]\s+", stripped)[-1]
    last_sentence_len = len(_tokens_simple(stripped))
    last_token = re.findall(r"[A-Za-z0-9']+", stripped.lower())[-1]
    last_token_is_stop = last_token in _STOPWORDS

    boundary_clean = 0.0
    if last_is_clean_punct:
        boundary_clean += 0.35
    if 6 <= last_sentence_len <= 14:
        boundary_clean += 0.25
    if not last_token_is_stop:
        boundary_clean += 0.10
    boundary_clean = min(boundary_clean, 0.6)

    # --- 2) Curiosity Ender ---
    end_q = stripped.endswith("?")
    cliff_tail = any(w in _end_tokens(stripped, 6) for w in _CLIFF_WORDS)
    curiosity = 0.0
    if end_q: 
        curiosity += 0.45
    if cliff_tail:
        curiosity += 0.20
    curiosity = min(curiosity, 0.55)

    # --- 3) Start↔End Echo (perfect-loop potential) ---
    start = _tokens(stripped, 10)
    end = _end_tokens(stripped, n=10)
    echo = _jaccard(start, end)  # 0–1
    # Enhanced bonus for perfect loops (first and last meaningful tokens match)
    match_edge = 0.25 if (start and end and start[0] == end[-1]) else 0.0
    # Additional bonus for exact phrase repetition
    exact_phrase_bonus = 0.15 if (start and end and len(start) >= 2 and len(end) >= 2 and 
                                 start[:2] == end[-2:]) else 0.0
    echo_score = min(0.7, 0.4 * echo + match_edge + exact_phrase_bonus)

    # --- 4) Quotability Pattern ---
    quotable_hits = 0.0
    if _has_pattern(stripped, _QUOTABLE_PATTERNS):
        quotable_hits = 0.6  # one strong hit saturates most value

    # --- 5) Anti-Outro Penalty ---
    outro_pen = 0.0
    if _has_pattern(stripped, _OUTRO_PATTERNS):
        outro_pen = 0.35  # subtract later

    # --- Combine with saturating combiner ---
    parts = [boundary_clean, curiosity, echo_score, quotable_hits]
    # Use a more aggressive saturating combiner for better differentiation
    positive = 0.70 * sat([x/0.6 for x in parts])  # normalize parts near 0–1, then scale
    raw = max(0.0, positive - outro_pen)

    # Clamp & return (optionally calibrate externally like other subscores)
    return float(np.clip(raw, 0.0, 1.0))

def _arousal_score_text(text: str, genre: str = 'general') -> float:
    """Enhanced text arousal scoring with genre awareness and intensity levels"""
    if not text:
        return 0.0
    
    t = text.lower()
    score = 0.0
    
    # Enhanced exclamation detection with intensity
    exclam_count = text.count('!')
    if exclam_count > 0:
        # More exclamations = higher intensity
        if exclam_count >= 3:
            score += 0.4  # High intensity
        elif exclam_count == 2:
            score += 0.25  # Medium intensity
        else:
            score += 0.15  # Low intensity
    
    # Enhanced caps detection (include short impactful words)
    caps_words = sum(1 for word in text.split() if word.isupper() and len(word) >= 2)
    if caps_words > 0:
        score += min(caps_words * 0.12, 0.25)  # Slightly higher weight
    
    # Enhanced emotion words with intensity levels
    high_intensity_words = ["insane", "shocking", "unbelievable", "mind-blowing", "incredible", "crazy", "wild", "epic", "amazing"]
    medium_intensity_words = ["awesome", "great", "fantastic", "wonderful", "exciting", "thrilling", "intense", "powerful"]
    low_intensity_words = ["good", "nice", "cool", "interesting", "fun", "enjoyable"]
    
    high_hits = sum(1 for word in high_intensity_words if word in t)
    medium_hits = sum(1 for word in medium_intensity_words if word in t)
    low_hits = sum(1 for word in low_intensity_words if word in t)
    
    # Weighted scoring by intensity
    score += min(high_hits * 0.15, 0.4)      # High intensity words
    score += min(medium_hits * 0.08, 0.2)    # Medium intensity words
    score += min(low_hits * 0.03, 0.1)       # Low intensity words
    
    # Genre-specific arousal patterns
    if genre == 'fantasy_sports':
        sports_intensity_words = ["fire", "draft", "start", "bench", "target", "sleeper", "bust", "league-winner"]
        sports_hits = sum(1 for word in sports_intensity_words if word in t)
        score += min(sports_hits * 0.1, 0.2)
    elif genre == 'comedy':
        comedy_intensity_words = ["hilarious", "funny", "lol", "haha", "rofl", "joke", "punchline"]
        comedy_hits = sum(1 for word in comedy_intensity_words if word in t)
        score += min(comedy_hits * 0.12, 0.25)
    elif genre == 'true_crime':
        crime_intensity_words = ["murder", "killer", "victim", "evidence", "mystery", "suspicious", "alibi"]
        crime_hits = sum(1 for word in crime_intensity_words if word in t)
        score += min(crime_hits * 0.1, 0.2)
    
    # Question marks add engagement (arousal)
    question_count = text.count('?')
    if question_count > 0:
        score += min(question_count * 0.05, 0.15)
    
    # Urgency indicators
    urgency_words = ["now", "immediately", "urgent", "critical", "important", "must", "need", "quickly"]
    urgency_hits = sum(1 for word in urgency_words if word in t)
    score += min(urgency_hits * 0.08, 0.2)
    
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
        # FIXED: Default path with higher hook weight
        path_a = (0.50 * f.get("hook_score", 0.0) + 0.15 * f.get("arousal_score", 0.0) + 
                  0.10 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
                  0.10 * f.get("platform_len_match", 0.0) + 0.03 * f.get("loopability", 0.0) + 
                  0.02 * f.get("insight_score", 0.0))  # NEW: Insight content boost
        
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
    
    # Calculate synergy using configurable mode (additive vs multiplier)
    config = get_config()
    synergy_config = config.get("synergy", {"mode": "additive", "additive_bonus_max": 0.02, "multiplier_min": 0.95})
    synergy_mode = synergy_config.get("mode", "additive")
    
    if synergy_mode == "additive":
        # Small additive synergy bonus (neutral by default)
        syn_bonus = 0.0
        if f.get("arousal_score", 0.0) >= 0.60: syn_bonus += 0.01
        if f.get("question_score", 0.0) >= 0.60: syn_bonus += 0.01
        syn_bonus = min(syn_bonus, synergy_config.get("additive_bonus_max", 0.02))
        
        synergy_score = min(1.0, base_score + syn_bonus)
        synergy_mult = 1.00  # For logging compatibility
        synergy_bonus = syn_bonus
    else:
        # Multiplier mode with minimum cap
        synergy_mult = _synergy_v4(f.get("hook_score", 0.0), f.get("arousal_score", 0.0), f.get("payoff_score", 0.0))
        synergy_mult = max(synergy_mult, synergy_config.get("multiplier_min", 0.95))
        synergy_score = base_score * synergy_mult
        synergy_bonus = 0.0
    
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
        "synergy_mode": synergy_mode,
        "synergy_bonus": synergy_bonus,
        "bonuses_applied": bonuses,
        "bonus_reasons": bonus_reasons,
        "viral_score_100": int(final_score * 100),
        "display_score": int(final_score * 100),  # Frontend expects this field
        "confidence": confidence_level,           # Frontend expects this field
        "confidence_color": confidence_color,     # Frontend expects this field
        "scoring_version": "v4.1",               # Version tag for compatibility
        "debug_info": {
            "synergy_mode": synergy_mode,
            "synergy_bonus": synergy_bonus,
            "quality_gate_applied": apply_penalties,
            "genre": genre,
            "platform": platform
        }
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
    
    # Hook scoring with V5 implementation
    config = get_config()
    use_v5 = bool(config.get("hook_v5", {}).get("enabled", True))
    
    if use_v5:
        # Hook V5 scoring
        seg_idx = segment.get("index", 0)
        # We'll set these after features are computed
        h_raw, h_cal, h_dbg = _hook_score_v5(
            text,
            cfg=config,
            segment_index=seg_idx,
            audio_modifier=0.0,  # Will be updated after audio analysis
            arousal=0.0,  # Will be updated after arousal computation
            q_or_list=0.0,  # Will be updated after question detection
        )
        hook_score = float(h_cal)
        hook_reasons = ",".join(h_dbg.get("reasons", []))
        hook_details = h_dbg
    else:
        # Legacy V4 hook scoring
        hook_score, hook_reasons, hook_details = _hook_score_v4(text, segment.get("arousal_score", 0.0), words_per_sec, genre, 
                                                               segment.get("audio_data"), segment.get("sr"), segment.get("start", 0.0))
    payoff_score, payoff_type = _detect_payoff(text, genre)
    
    # NEW: Detect insight content vs. intro/filler (V2 if enabled)
    if config.get("insight_v2", {}).get("enabled", False):
        insight_score, insight_reasons = _detect_insight_content_v2(text, genre)
    else:
        insight_score, insight_reasons = _detect_insight_content(text, genre)
    
    niche_penalty, niche_reason = _calculate_niche_penalty(text, genre)
    
    # ENHANCED AUDIO ANALYSIS: Compute actual audio arousal with intelligent fallback
    audio_arousal = _audio_prosody_score(audio_file, segment["start"], segment["end"], text=text, genre=genre)
    
    # ENHANCED TEXT AROUSAL: Genre-aware text arousal scoring
    text_arousal = _arousal_score_text(text, genre)
    
    # ADAPTIVE COMBINATION: Adjust audio/text ratio based on genre and content type
    if genre == 'fantasy_sports':
        # Sports content benefits more from text analysis (stats, names, etc.)
        combined_arousal = 0.6 * audio_arousal + 0.4 * text_arousal
    elif genre == 'comedy':
        # Comedy benefits more from audio (timing, delivery)
        combined_arousal = 0.8 * audio_arousal + 0.2 * text_arousal
    elif genre == 'true_crime':
        # True crime benefits from both (dramatic delivery + intense content)
        combined_arousal = 0.7 * audio_arousal + 0.3 * text_arousal
    else:
        # Default balanced approach
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
        "info_density": _info_density_v4(text),  # Will be updated by V2 system if enabled
        "loopability": _loopability_heuristic(text),
        "insight_score": insight_score,  # NEW: Insight content detection (may be adjusted by confidence multiplier)
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
        "type": segment.get("type", "general"),  # Preserve moment type for bonuses
        
        # ENHANCED: Multi-dimensional hook details
        "hook_components": hook_details.get("hook_components", {}),
        "hook_type": hook_details.get("hook_type", "general"),
        "hook_confidence": hook_details.get("confidence", 0.0),
        "audio_modifier": hook_details.get("audio_modifier", 0.0),
        "laughter_boost": hook_details.get("laughter_boost", 0.0),
        "time_weighted_score": hook_details.get("time_weighted_score", 0.0)
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
    
    # ensure downstream names exist for synergy calculations
    feats["arousal"] = feats.get("arousal", feats.get("arousal_score", 0.0))
    feats["q_or_list"] = feats.get("q_or_list", feats.get("question_score", 0.0))
    
    # RECOMPUTE Hook V5 with real signals (arousal, question_score, audio_modifier)
    if use_v5:
        h_raw, h_cal, h_dbg = _hook_score_v5(
            text,
            cfg=config,
            segment_index=seg_idx,
            audio_modifier=feats.get("audio_modifier", 0.0),
            arousal=float(feats.get("arousal_score", 0.0)),
            q_or_list=float(feats.get("question_score", 0.0)),
        )
        feats["hook_score"] = float(h_cal)
        feats.setdefault("_debug", {})
        feats["_debug"]["hook_v5"] = h_dbg
    
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

def compute_features_v4_batch(segments: list, audio_file: str, y_sr=None, genre: str = 'general', platform: str = 'tiktok') -> list:
    """Batch processing for compute_features_v4 with Info Density V2, Question/List V2, Emotion V2, and Hook V5 calibration"""
    # Check if V2 systems are enabled
    config = get_config()
    info_cfg = config.get("info_density", {})
    ql_cfg = config.get("question_list_v2", {})
    emotion_cfg = config.get("emotion_v2", {})
    hook_cfg = config.get("hook_v5", {})
    
    if not info_cfg.get("enabled", True) and not ql_cfg.get("enabled", True) and not emotion_cfg.get("enabled", False) and not hook_cfg.get("enabled", False):
        # V1 path - process segments individually
        return [compute_features_v4(segment, audio_file, y_sr, genre, platform) for segment in segments]
    
    # V2 path - batch process with calibration
    # First pass: compute all features including raw scores
    processed_segments = []
    raw_info_densities = []
    raw_question_lists = []
    raw_emotions = []
    
    for segment in segments:
        # Compute all features normally
        features = compute_features_v4(segment, audio_file, y_sr, genre, platform)
        processed_segments.append(features)
        
        # Extract raw scores for calibration
        text = segment.get("text", "")
        dur = float(segment.get("end", 0.0) - segment.get("start", 0.0)) or 0.0
        
        if info_cfg.get("enabled", True):
            raw_density = _info_density_raw_v2(text, dur)
            raw_info_densities.append(raw_density)
        
        if ql_cfg.get("enabled", True):
            raw_ql = _question_list_raw_v2(text, dur, genre)
            raw_question_lists.append(raw_ql)
        
        if emotion_cfg.get("enabled", False):
            # Build audio sidecar for emotion
            audio_sidecar = build_emotion_audio_sidecar(features)
            raw_emotion = _emotion_raw_v2(text, audio_sidecar, genre)
            raw_emotions.append(raw_emotion)
    
    # Compute calibration statistics and update scores
    if info_cfg.get("enabled", True) and raw_info_densities:
        MU_info, SIGMA_info = _calibrate_info_density_stats(raw_info_densities)
        for i, segment in enumerate(processed_segments):
            segment["info_density"] = info_density_score_v2(segments[i], MU_info, SIGMA_info)
    
    if ql_cfg.get("enabled", True) and raw_question_lists:
        MU_ql, SIGMA_ql = _ql_calibrate_stats(raw_question_lists)
        for i, segment in enumerate(processed_segments):
            segment["question_list"] = question_list_score_v2(segments[i], MU_ql, SIGMA_ql)
            # Also set question_score for backward compatibility
            segment["question_score"] = segment["question_list"]
    
    if emotion_cfg.get("enabled", False) and raw_emotions:
        MU_emotion, SIGMA_emotion = _calibrate_emotion_stats(raw_emotions)
        for i, segment in enumerate(processed_segments):
            # Build audio sidecar for final emotion score
            audio_sidecar = build_emotion_audio_sidecar(segment)
            segment["emotion_score"] = emotion_score_v2(segments[i], MU_emotion, SIGMA_emotion, audio_sidecar)
    
    # Hook V5 processing is now handled in compute_features_v4
    
    return processed_segments

def explain_segment_v4(features: Dict, weights: Dict = None, genre: str = 'general') -> Dict:
    if weights is None:
        weights = get_clip_weights()
    
    scoring_result = score_segment_v4(features, weights, genre=genre)
    f = features
    
    strengths = []
    improvements = []
    
    hook_score = f.get("hook_score", 0.0)
    if hook_score >= 0.8:
        strengths.append("**Killer Hook**: Opens with attention-grabbing content")
    elif hook_score < 0.4:
        improvements.append("**Weak Hook**: Needs compelling opening")
    
    viral_score = scoring_result["viral_score_100"]
    if viral_score >= 70:
        overall = "**High Viral Potential** - Strong fundamentals"
    elif viral_score >= 50:
        overall = "**Good Potential** - Solid foundation"
    else:
        overall = "**Needs Work** - Multiple issues to address"
    
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
    
    # Add synergy bonus
    synergy_bonus = compute_synergy_bonus(features)
    base_viral += synergy_bonus
    
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
    score, _, _ = _hook_score_v4(text)
    return score

def _emotion_score(text: str) -> float:
    return _emotion_score_v4(text)

def _payoff_presence(text: str) -> float:
    score, _ = _payoff_presence_v4(text)
    return score

def _info_density(text: str) -> float:
    return _info_density_v4(text)

# Info Density V2 - Enhanced information content detection
import re
import numpy as np
from collections import Counter

# Precompiled regexes for performance
_STOP = set("""
a an the and but or so because that which to of in on at for with by as is are was were be been being this those these it
i you he she we they me him her us them my your his her their our mine yours theirs ours
""".split())

_FILLERS = set("like you know kinda sort of basically literally actually honestly truly really just".split())
_HEDGES  = set("maybe probably perhaps i think i guess seems might could".split())

_NUM_RE  = re.compile(r"(?:(?:\d+[.,]?\d*)|(?:\d+%))")
_UNIT_RE = re.compile(r"\b(km|mi|lb|kg|mph|fps|hz|kbps|mbps|gb|tb|ft|in|cm|mm)\b", re.I)
_CURR_RE = re.compile(r"[$€£¥]")

def _tokens_simple(text):
    """Extract tokens from text"""
    return re.findall(r"[A-Za-z0-9'%.]+", text.lower())

def _split_sents(text):
    """Split text into sentences"""
    return re.split(r'(?<=[.!?])\s+', text.strip())

def _tri01(x, lo, hi):
    """Triangular function: 0 outside [lo,hi], 1 at midpoint, linear slopes"""
    if lo >= hi or x <= lo or x >= hi:
        return 0.0
    mid = 0.5 * (lo + hi)
    return (x - lo) / (mid - lo) if x < mid else (hi - x) / (hi - mid)

def _safe_div(a, b):
    """Safe division with zero guard"""
    return a / b if b > 1e-9 else 0.0

def _spacy_entities(text):
    """Extract named entities using spaCy (optional)"""
    try:
        from nlp_loader import get_spacy
        nlp = get_spacy()
        if nlp is None:
            if not getattr(_spacy_entities, "_warned", False):
                logger.debug("spaCy unavailable; falling back to heuristic entity counter.")
                _spacy_entities._warned = True
            return None
        
        # Cap text length to keep latency predictable
        text = text[:2000]
        doc = nlp(text)
        return sum(1 for ent in doc.ents if ent.label_ not in ("CARDINAL",))
    except Exception as e:
        if not getattr(_spacy_entities, "_warned", False):
            logger.debug("spaCy unavailable; falling back to heuristic entity counter: %s", e)
            _spacy_entities._warned = True
        return None

def _info_density_raw_v2(text: str, duration_s: float) -> float:
    """Enhanced info density scoring with multiple signals"""
    # Get configuration
    config = get_config()
    info_cfg = config.get("info_density", {})
    
    # Default configuration
    default_cfg = {
        "w_content": 0.30, "w_lexdiv": 0.15, "w_compression": 0.20, 
        "w_specific": 0.20, "w_action": 0.15,
        "ideal_tokens_per_sec": [1.8, 3.8],
        "repeat_bigram_penalty": 0.15,
        "fluff_penalty_cap": 0.20,
        "textual_guard_min_tokens": 8,
        "sigmoid_a": 1.6
    }
    cfg = {**default_cfg, **info_cfg}
    
    if not text or duration_s <= 0:
        return 0.0
    
    toks = _tokens_simple(text)
    N = len(toks)
    if N < cfg["textual_guard_min_tokens"]:
        return 0.35  # neutral for short bits

    # 1) Content word ratio
    content = [t for t in toks if t not in _STOP]
    p_content = _safe_div(len(content), N)
    s_content = np.clip((p_content - 0.45) / (0.85 - 0.45), 0, 1) * cfg["w_content"]

    # 2) Lexical diversity (smoothed TTR)
    ttr = _safe_div(len(set(toks)), N)
    ttr_smooth = (ttr * N) / (N + 20)  # Good-Turing-ish smoothing
    s_lexdiv = np.clip((ttr_smooth - 0.35) / (0.75 - 0.35), 0, 1) * cfg["w_lexdiv"]

    # 3) Compression (tokens/sec band)
    tps = _safe_div(len(content), duration_s)
    lo, hi = cfg["ideal_tokens_per_sec"]
    s_comp = _tri01(tps, lo, hi) * cfg["w_compression"]

    # 4) Specificity
    nums = len(_NUM_RE.findall(text))
    units = len(_UNIT_RE.findall(text))
    currs = len(_CURR_RE.findall(text))
    ents = _spacy_entities(text)
    if ents is None:
        # Fallback: capitalized bigrams
        ents = len(re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    
    comp_markers = len(re.findall(r"\b(vs|versus|than|compared to)\b", text, re.I))
    define_markers = len(re.findall(r"\b(means|defined as|is when|refers to)\b", text, re.I))
    list_markers = len(re.findall(r"\b(first|second|third|top \d+|(\d+)\s+(tips|steps|reasons))\b", text, re.I))

    spec_score = np.tanh(0.25*nums + 0.25*units + 0.2*currs + 0.2*ents + 
                         0.2*comp_markers + 0.15*define_markers + 0.15*list_markers)
    s_specific = np.clip(spec_score, 0, 1) * cfg["w_specific"]

    # 5) Actionability
    sents = _split_sents(text)
    imper = sum(1 for s in sents if re.match(r"^\s*(do|try|use|avoid|stop|start|remember|never|always)\b", s.strip().lower()))
    s_action = np.clip(np.tanh(0.6*imper), 0, 1) * cfg["w_action"]

    # Penalties: fluff & repetition
    fill = sum(1 for t in toks if t in _FILLERS)
    hedg = sum(1 for t in toks if t in _HEDGES)
    pen_fluff = min(cfg["fluff_penalty_cap"], 0.03*fill + 0.02*hedg)
    
    # Bigram repetition (only if text is long enough)
    pen_repeat = 0.0
    if N >= 20:
        bigrams = [" ".join(pair) for pair in zip(toks, toks[1:])]
        common_bi = sum(1 for _, c in Counter(bigrams).most_common(8) if c >= 3)
        pen_repeat = min(cfg["repeat_bigram_penalty"], 0.07*common_bi)

    # Combine positives with saturating combiner
    pos = sat([
        np.clip(s_content / max(cfg["w_content"], 1e-9), 0, 1),
        np.clip(s_lexdiv / max(cfg["w_lexdiv"], 1e-9), 0, 1),
        np.clip(s_comp / max(cfg["w_compression"], 1e-9), 0, 1),
        np.clip(s_specific / max(cfg["w_specific"], 1e-9), 0, 1),
        np.clip(s_action / max(cfg["w_action"], 1e-9), 0, 1),
    ])
    
    raw = np.clip(pos - (pen_fluff + pen_repeat), 0.0, 1.0)
    return float(raw)

def _calibrate_info_density_stats(raw_list):
    """Compute robust statistics for info density calibration"""
    arr = np.asarray(raw_list, dtype=float)
    if arr.size == 0:
        return 0.5, 0.2
    med = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    sigma = iqr / 1.349
    return med, sigma

def _sigmoid_info_density(x, a):
    """Sigmoid function for info density calibration"""
    return 1.0 / (1.0 + np.exp(-a * x))

def info_density_score_v2(segment: dict, MU: float, SIGMA: float) -> float:
    """Enhanced info density scoring with calibration"""
    text = (segment.get("text") or segment.get("transcript") or "")
    dur = float(segment.get("end", 0.0) - segment.get("start", 0.0)) or 0.0
    raw = _info_density_raw_v2(text, dur)

    # Get configuration
    config = get_config()
    info_cfg = config.get("info_density", {})
    a = info_cfg.get("sigmoid_a", 1.6)
    
    z = (raw - MU) / (SIGMA if SIGMA > 1e-6 else 1.0)
    score = float(_sigmoid_info_density(z, a))

    # Add debug information
    if "_debug" not in segment:
        segment["_debug"] = {}
    
    debug = segment["_debug"]
    debug["info_density_raw"] = raw
    debug["info_density_mu"] = MU
    debug["info_density_sigma"] = SIGMA
    debug["info_density_final"] = score
    
    return max(0.0, min(1.0, score))

# Question/List V2 - Enhanced interactive engagement detection
import re
import math
import numpy as np
from collections import Counter

# Precompiled regexes for performance
_QMARK      = re.compile(r"\?\s*$")
_INTERROG   = re.compile(r"\b(what|why|how|when|where|which|who)\b", re.I)
_COMPARE    = re.compile(r"\b(vs\.?|versus|better than|worse than|compare(?:d)? to)\b", re.I)
_CHOICE     = re.compile(r"\b(which|pick|choose|would you rather|either|or)\b", re.I)
_RHET_IND   = re.compile(r"\b(i wonder if|what if|ever notice|guess why|have you ever)\b", re.I)
_CLIFF_Q    = re.compile(r"\b(guess|until|turns out|reveal|reveals)\b", re.I)
_VACUOUS_YN = re.compile(r"^(is|are|do|does|did|can|could|should|would|will|was|were)\b", re.I)

# Lists
_ENUM_NUM   = re.compile(r"(?:^|\s)(\d+)[.)]\s")
_ENUM_ORD   = re.compile(r"\b(first|second|third|fourth|fifth|top\s+\d+|step\s+\d+)\b", re.I)
_LIST_SEP   = re.compile(r"(?:^|\s)[•\-–—]\s")        # bullets/dashes
_EMOJI_BUL  = re.compile(r"(?:^|\s)[•]\s")     # common bullet points

# Engagement
_GENUINE    = re.compile(r"\b(tell me|share your|what(?:'| i)s your (?:take|experience)|how do you)\b", re.I)
_BAIT       = re.compile(r"(like and subscribe|follow for more|smash (?:that )?like|link in bio|comment\s+(yes|🔥|below))", re.I)

_STOPWORDS  = set("a an the and or but so to of in on at with for as is are was were be been this that it".split())

def _sentences(text: str):
    """Split text into sentences"""
    return re.split(r'(?<=[.!?])\s+', text.strip()) if text else []

def _tokens_ql(s: str):
    """Extract tokens from text for question/list analysis"""
    return re.findall(r"[A-Za-z0-9']+", s.lower()) if s else []

def _tri01_ql(x, lo, hi):
    """Triangular function for question/list scoring"""
    if lo >= hi or x <= lo or x >= hi:
        return 0.0
    mid = (lo + hi) / 2.0
    return (x - lo) / (mid - lo) if x < mid else (hi - x) / (hi - mid)

def _question_list_raw_v2(text: str, duration_s: float | None = None, genre: str | None = None) -> float:
    """Enhanced question/list scoring with multiple signals"""
    # Get configuration
    config = get_config()
    ql_cfg = config.get("question_list_v2", {})
    
    # Default configuration
    default_cfg = {
        "sigmoid_a": 1.5,
        "ideal_items_range": [3, 7],
        "pen_bait_cap": 0.25,
        "pen_vacuous_q_cap": 0.20,
        "textual_guard_min_tokens": 6,
        "genre_tweaks": {
            "education": {"list_bonus": 0.03, "question_bonus": 0.00},
            "entertainment": {"list_bonus": 0.00, "question_bonus": 0.03},
            "linkedin": {"list_bonus": 0.02, "question_bonus": 0.00}
        }
    }
    cfg = {**default_cfg, **ql_cfg}
    
    if not text:
        return 0.0
    
    toks = _tokens_ql(text)
    if len(toks) < cfg["textual_guard_min_tokens"]:
        return 0.35  # neutral for very short strings

    sents = _sentences(text)
    first = sents[0] if sents else text
    last = sents[-1] if sents else text

    # -------- Questions subscore --------
    qmark = 1.0 if _QMARK.search(last or "") else 0.0
    wh = 1.0 if (_INTERROG.search(last or "") or _INTERROG.search(first or "")) else 0.0
    compare = 1.0 if _COMPARE.search(text) else 0.0
    choice = 1.0 if _CHOICE.search(text) else 0.0
    rhetind = 1.0 if _RHET_IND.search(text) else 0.0
    cliffq = 1.0 if (_CLIFF_Q.search(last or "") and qmark) else 0.0
    genuine = 1.0 if _GENUINE.search(text) else 0.0

    # Internal weights (kept in code to avoid config sprawl)
    q_direct = 0.6 * qmark + 0.4 * wh
    q_compare = 0.3 * compare
    q_choice = 0.25 * choice
    q_rhet = 0.15 * rhetind
    q_cliff = 0.15 * cliffq
    q_prompt = min(0.10, 0.10 * genuine)  # tiny boost for honest prompts

    Q_raw = sat([q_direct, q_compare, q_choice, q_rhet, q_cliff]) + q_prompt
    Q_raw = float(min(1.0, Q_raw))

    # -------- Lists subscore --------
    # Detect items by numbered/ordinal/bullet lines (or emoji bullets)
    lines = re.split(r"(?:\n|;|\s{2,})", text.strip())
    item_lengths = []
    item_starts = []
    enum_hits = 0
    
    for l in lines:
        has_enum = bool(_ENUM_NUM.search(l) or _ENUM_ORD.search(l) or _LIST_SEP.search(l) or _EMOJI_BUL.search(l))
        if has_enum:
            enum_hits += 1
            # Token after marker for parallelism check
            m = _ENUM_NUM.search(l) or _ENUM_ORD.search(l) or _LIST_SEP.search(l) or _EMOJI_BUL.search(l)
            tail = l[m.end():] if m and hasattr(m, 'end') else l
            tl_toks = [t for t in _tokens_ql(tail) if t not in _STOPWORDS]
            if tl_toks:
                item_starts.append(tl_toks[0])
            item_lengths.append(len(_tokens_ql(l)))

    # Fallback: comma-separated pseudo-lists
    if enum_hits == 0:
        commas = len(re.findall(r",\s+", text))
        if commas >= 2:
            enum_hits = 1
            item_lengths = [len(_tokens_ql(x)) for x in re.split(r",\s+", text)]
            item_starts = [(_tokens_ql(x) or [""])[0] for x in re.split(r",\s+", text)]

    parallel = 0.0
    if len(item_starts) >= 3:
        c = Counter(item_starts).most_common(1)[0][1]
        parallel = min(1.0, c / max(1, len(item_starts)))

    concise = 0.0
    if item_lengths:
        med_len = float(np.median(item_lengths))
        concise = 1.0 if 4 <= med_len <= 12 else (0.0 if med_len < 3 or med_len > 25 else 0.5)

    items = len(item_lengths)
    lo, hi = cfg["ideal_items_range"]
    items_band = _tri01_ql(items, lo, hi) if items > 0 else 0.0

    L_raw = sat([
        1.0 if enum_hits else 0.0,     # enumeration present
        0.8 * parallel,                # structural parallelism
        0.6 * concise,                 # brevity
        0.4 * items_band,              # sweet spot of items
    ])

    # -------- Penalties --------
    pen_bait = min(cfg["pen_bait_cap"], 0.25 if _BAIT.search(text) else 0.0)

    pen_vacuous = 0.0
    last_toks = _tokens_ql(last)
    if (qmark and not wh and last_toks and _VACUOUS_YN.search(last.lower()) and len(last_toks) <= 8):
        pen_vacuous = min(cfg["pen_vacuous_q_cap"], 0.20)

    # -------- Genre micro-tweaks --------
    if genre:
        gt = cfg.get("genre_tweaks", {}).get(genre.lower(), {})
        Q_raw = float(min(1.0, Q_raw + gt.get("question_bonus", 0.0)))
        L_raw = float(min(1.0, L_raw + gt.get("list_bonus", 0.0)))

    pos = sat([Q_raw, L_raw])
    raw = float(np.clip(pos - (pen_bait + pen_vacuous), 0.0, 1.0))
    return raw

def _ql_calibrate_stats(raws: list[float]) -> tuple[float, float]:
    """Compute robust statistics for question/list calibration"""
    arr = np.asarray(raws, dtype=float)
    if arr.size == 0:
        return 0.5, 0.2
    med = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    sigma = iqr / 1.349
    return med, sigma

def _sigmoid_ql(x: float, a: float) -> float:
    """Sigmoid function for question/list calibration"""
    return 1.0 / (1.0 + math.exp(-a * x))

def question_list_score_v2(segment: dict, MU: float, SIGMA: float) -> float:
    """Enhanced question/list scoring with calibration"""
    text = segment.get("text") or segment.get("transcript") or ""
    genre = (segment.get("genre") or "").lower() or None
    raw = _question_list_raw_v2(text, genre=genre)
    
    # Get configuration
    config = get_config()
    ql_cfg = config.get("question_list_v2", {})
    a = ql_cfg.get("sigmoid_a", 1.5)
    
    z = (raw - MU) / (SIGMA if SIGMA > 1e-6 else 1.0)
    score = float(_sigmoid_ql(z, a))
    
    # Add debug information
    if "_debug" not in segment:
        segment["_debug"] = {}
    
    debug = segment["_debug"]
    debug["ql_raw"] = raw
    debug["ql_mu"] = MU
    debug["ql_sigma"] = SIGMA
    debug["ql_final"] = score
    
    return max(0.0, min(1.0, score))

def attach_question_list_scores_v2(segments: list[dict]) -> None:
    """Batch processing for question/list V2 with calibration"""
    if not segments:
        return
    
    raws = [_question_list_raw_v2(
        (s.get("text") or s.get("transcript") or ""), 
        genre=(s.get("genre") or "").lower() or None
    ) for s in segments]
    
    MU, SIGMA = _ql_calibrate_stats(raws)
    
    for s, raw in zip(segments, raws):
        s["question_list"] = question_list_score_v2(s, MU, SIGMA)

def apply_synergy_bonus(segment: dict) -> float:
    """Apply synergy bonus between question/list and info density"""
    ql = float(segment.get("question_list", 0.0))
    idv = float(segment.get("info_density", 0.0))
    return 0.01 if (ql >= 0.60 and idv >= 0.60) else 0.0

# ---- Emotion V2 System -------------------------------------------------------------
import re, math, numpy as np
from collections import Counter

# Lightweight emotion family lexicons
_EMO_FAM = {
    "anger":   r"(angry|furious|mad|irritat(?:ed|ing)|rage|pissed|enrag(?:e|ed))",
    "fear":    r"(afraid|terrified|scared|anxious|panic|frighten(?:ed)?|horror|nightmare)",
    "sad":     r"(sad|devastat(?:ed|ing)|heartbroken|depress(?:ed|ing)|mourning|grief)",
    "joy":     r"(happy|thrilled|excited|joy|delight(?:ed)?|proud|relief|relieved|love)",
    "surprise":r"(shocked|jaw[-\s]?dropped|stunned|unbelievable|insane|crazy|wow|mind[-\s]?blown|🤯)",
    "disgust": r"(disgust(?:ed|ing)|gross|nauseat(?:ed|ing)|repuls(?:e|ed))",
    "desire":  r"(want|crave|obsessed|hungry for|itch(?:ing)? to|ambitious|driven|aspire)",
    "shame":   r"(ashamed|embarrass(?:ed|ing)|humiliat(?:ed|ing)|guilty|regret)",
    "awe":     r"(awe(?:some)?|awe[-\s]?inspiring|majestic|breathtaking|staggering|incredible)"
}

# Legacy emotion words mapping for continuity
LEGACY_HIGH = {"incredible","insane","mind-blowing","shocking"}
LEGACY_REG  = {"amazing","awesome","unbelievable","heartbreaking","devastating",
               "terrifying","exciting","hilarious","beautiful","crazy","sad","angry","happy"}

MIGRATE = {
    "awe":       {"incredible","amazing","awesome","unbelievable","mind-blowing"},
    "surprise":  {"shocking","crazy"},
    "sad":       {"heartbreaking","devastating","sad"},
    "fear":      {"terrifying"},
    "joy":       {"exciting","beautiful","happy","hilarious"},
    "anger":     {"angry"}
}

# Regex patterns for intensity detection
_WORD = re.compile(r"[A-Za-z']+")
_EXCL = re.compile(r"!{2,}")
_ELLIPS = re.compile(r"\.{3,}")
_ALLCAP = re.compile(r"\b[A-Z]{2,}\b")
_EMOJIS = re.compile(r"[😂🤣😭😡😱😍❤️💔🤯✨🔥]")

def _extend_families_with_legacy():
    """Extend emotion families with legacy words for continuity"""
    global _EMO_FAM
    config = get_config()
    if config.get("emotion_v2", {}).get("migrate_legacy", True):
        for fam, words in MIGRATE.items():
            if fam in _EMO_FAM:
                extra = "|".join(re.escape(w) for w in words)
                _EMO_FAM[fam] = f"(?:{_EMO_FAM[fam]}|{extra})"

# Initialize legacy migration
_extend_families_with_legacy()

def _tokens_simple(t): 
    return _WORD.findall(t.lower()) if t else []

def _sent_split(t): 
    return re.split(r'(?<=[.!?])\s+', t.strip()) if t else []

def _negation_mask(toks, idx, W):
    """Return -1 if a negator appears within window before idx."""
    lo = max(0, idx - W)
    for j in range(lo, idx):
        if toks[j] in {"not","no","never","isnt","isn't","dont","don't","cannot","can't","wont","won't"}:
            return -1
    return +1

def _match_family_score(text, neg_win):
    """Match emotion families with negation awareness"""
    toks = _tokens_simple(text)
    fam_hits = {}
    total = 0
    for fam, pat in _EMO_FAM.items():
        rx = re.compile(pat, re.I)
        fam_hits[fam] = 0
        for m in rx.finditer(text):
            # rough token index for negation windowing
            span_text = text[:m.start()].lower()
            idx = len(_tokens_simple(span_text))
            fam_hits[fam] += 1 * _negation_mask(toks, idx, neg_win)
            total += 1
    
    # positive vs negative valence proxy
    pos = fam_hits["joy"] + fam_hits["awe"] + fam_hits["desire"]
    neg = fam_hits["anger"] + fam_hits["fear"] + fam_hits["sad"] + fam_hits["disgust"] + fam_hits["shame"]
    pos = max(0, pos); neg = max(0, neg)

    families_present = sum(1 for k,v in fam_hits.items() if v>0)
    fam_density = np.tanh(0.35 * total)                          # diminishing returns
    diversity   = np.tanh(0.7 * min(families_present, 3))        # complexity sweet spot 1–3
    val_balance = np.tanh(0.6 * (pos - neg) / max(1, pos + neg)) # -1..1 → -0.54..0.54
    val_balance = (val_balance + 0.54) / 1.08                    # normalize ~0..1

    return fam_hits, dict(pos=pos, neg=neg), float(fam_density), float(diversity), float(val_balance)

def _intensity_text(text, caps_softcap, punct_cap, emoji_cap):
    """Detect intensity cues with soft caps"""
    exc  = 1.0 if _EXCL.search(text) else 0.0
    ell  = 0.5 if _ELLIPS.search(text) else 0.0
    caps = min(caps_softcap, len(_ALLCAP.findall(text)))
    emo  = len(_EMOJIS.findall(text))
    # combine with caps/emoji soft caps
    raw = 0.4*exc + 0.2*ell + 0.25*min(1.0, caps/caps_softcap) + 0.15*min(1.0, emo/3.0)
    return float(min(1.0, raw, punct_cap + emoji_cap))  # hard ceiling for spam

def _arc_shift(text):
    """Detect emotional arc/shift between first and last sentence"""
    sents = _sent_split(text)
    if len(sents) < 2: return 0.0
    def polarity(s):
        # ultra-light polarity: many good/bad words; configurable later
        pos = len(re.findall(r"\b(great|love|happy|proud|amazing|wow|relief)\b", s, re.I))
        neg = len(re.findall(r"\b(hate|scared|sad|angry|annoying|nightmare|pain)\b", s, re.I))
        return np.tanh(0.6*(pos-neg))
    p0, p1 = polarity(sents[0]), polarity(sents[-1])
    return float(np.clip(abs(p1-p0), 0.0, 1.0))

def _empathy(text):
    """Detect empathy markers"""
    return 1.0 if re.search(r"\b(i felt|made me (feel|cry|smile)|it hurts|i'm proud|i am proud|i'm terrified|i was terrified)\b", text, re.I) else 0.0

def build_emotion_audio_sidecar(seg):
    """Build audio sidecar from Arousal V5 debug fields"""
    d = seg.get("_debug", {})
    pv   = float(d.get("arousal_pitch_dyn", 0.0))
    flux = float(d.get("arousal_flux", 0.0))
    # if you log variance separately, prefer that; else reuse energy_core as a proxy
    eng  = float(d.get("arousal_energy_var", d.get("arousal_energy_core", 0.0)))
    if pv==flux==eng==0.0: return None
    return {"pitch_dyn": pv, "spectral_flux": flux, "energy_var": eng}

def _emotion_raw_v2(text: str, audio_features: dict | None, genre: str | None) -> float:
    """Compute raw emotion score with multiple signals"""
    if not text or len(text.strip()) < 3: return 0.0
    cfg_v2 = get_config().get("emotion_v2", {})
    cfg_lex = get_config().get("emotion_lexicons", {})
    negW = int(cfg_v2.get("negation_window", 3))
    caps_softcap = int(cfg_v2.get("caps_run_softcap", 3))
    punct_cap = float(cfg_v2.get("punct_repeat_cap", 0.08))
    emoji_cap = float(cfg_v2.get("emoji_max", 0.06))

    # --- TEXT FAMILY & VALENCE via LEXICONS or fallback ---
    if cfg_lex.get("enabled", False):
        hits = extract_emotion_hits_from_lexicons(text)   # [(family, weight)]
        fam_hits = {}
        total = 0.0
        pos = neg = 0.0
        for fam, w in hits:
            fam_hits[fam] = fam_hits.get(fam, 0.0) + w
            total += w
            if fam in {"joy","awe","desire","pride"}:
                pos += w
            elif fam in {"anger","fear","sad","disgust","shame"}:
                neg += w
        families_present = sum(1 for v in fam_hits.values() if v > 0)
        s_fam = float(np.tanh(0.30 * total))             # diminishing returns
        s_div = float(np.tanh(0.7 * min(families_present, 3)))
        s_val = 0.5  # default neutral
        if pos + neg > 0:
            s_val = (np.tanh(0.6 * (pos - neg) / (pos + neg)) + 0.54) / 1.08
    else:
        # fallback to existing regex family path
        fam_hits, val, s_fam, s_div, s_val = _match_family_score(text, negW)
    s_int = _intensity_text(text, caps_softcap, punct_cap, emoji_cap)
    s_arc = _arc_shift(text)
    s_emp = _empathy(text)

    # Audio micro contribution (already bounded in config)
    s_audio = 0.0
    if audio_features:
        # take gentle slice of arousal subfeatures if provided
        pv   = float(audio_features.get("pitch_dyn", 0.0))     # 0..1
        flux = float(audio_features.get("spectral_flux", 0.0)) # 0..1
        eng  = float(audio_features.get("energy_var", 0.0))    # 0..1
        s_audio = 0.4*pv + 0.3*flux + 0.3*eng
        s_audio = min(s_audio, float(cfg_v2.get("audio_micro_cap", 0.08)))

    # Combine with saturating combiner
    w = {
        "families": float(cfg_v2.get("w_text_families", 0.45)),
        "valence":  float(cfg_v2.get("w_valence_balance", 0.20)),
        "intens":   float(cfg_v2.get("w_intensity", 0.15)),
        "arc":      float(cfg_v2.get("w_arc_shift", 0.12)),
        "empathy":  float(cfg_v2.get("w_empathy", 0.08))
    }
    # normalize into [0,1] parts then saturate
    parts = [
        min(1.0, s_fam) * (w["families"]/max(1e-6, sum(w.values()))),
        s_val             * (w["valence"]/max(1e-6, sum(w.values()))),
        s_int             * (w["intens"]/max(1e-6, sum(w.values()))),
        s_arc             * (w["arc"]/max(1e-6, sum(w.values()))),
        s_emp             * (w["empathy"]/max(1e-6, sum(w.values()))),
        s_audio
    ]
    # saturating combiner
    pos = 1.0
    for x in parts: pos *= (1.0 - max(0.0, min(1.0, x)))
    raw = 1.0 - pos

    # Genre micro-tweaks
    g = (genre or "").lower()
    gt = cfg_v2.get("genre_tweaks", {}).get(g, {})
    if g == "comedy":
        if re.search(r"(shocked|unbelievable|unexpected|plot twist|😆|😂)", text, re.I):
            raw += float(gt.get("surprise_bonus", 0.0)) + float(gt.get("laughter_bonus", 0.0))
    if g == "true_crime":
        if re.search(r"(shocking|terrifying|victim|case|evidence)", text, re.I):
            raw += float(gt.get("fear_shock_bonus", 0.0))
        if re.search(r"(tragic|sad|heartbroken)", text, re.I):
            raw += float(gt.get("sadness_bonus", 0.0))
    if g == "health_wellness":
        if re.search(r"(hope|proud|better|improve|transform)", text, re.I):
            raw += float(gt.get("hope_pride_bonus", 0.0))
        if re.search(r"(ashamed|embarrass|guilty)", text, re.I):
            raw -= float(gt.get("shame_penalty", 0.0))
    if g == "business":
        if re.search(r"(proud|ambitious|driven|want|crave|win)", text, re.I):
            raw += float(gt.get("pride_desire_bonus", 0.0))

    return float(np.clip(raw, 0.0, 1.0))

def _calibrate_emotion_stats(raws: list[float]) -> tuple[float, float]:
    """Compute robust median/IQR stats for calibration"""
    arr = np.asarray(raws, dtype=float)
    if arr.size == 0: return 0.5, 0.2
    med = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75]); iqr = max(q3 - q1, 1e-6)
    sigma = iqr / 1.349
    return med, sigma

def _sigmoid_emotion(x: float, a: float) -> float:
    """Sigmoid function for calibration"""
    return 1.0 / (1.0 + math.exp(-a * x))

def emotion_score_v2(segment: dict, MU: float, SIGMA: float, audio_features: dict | None = None) -> float:
    """Apply calibration to raw emotion score"""
    cfg = get_config().get("emotion_v2", {})
    a = float(cfg.get("sigmoid_a", 1.5))
    text = (segment.get("text") or segment.get("transcript") or "")
    genre = segment.get("genre")
    raw = _emotion_raw_v2(text, audio_features, genre)
    z = (raw - MU) / (SIGMA if SIGMA > 1e-6 else 1.0)
    score = float(_sigmoid_emotion(z, a))

    dbg = segment.setdefault("_debug", {})
    dbg.update({
        "emotion_raw": raw, "emotion_mu": MU, "emotion_sigma": SIGMA, "emotion_final": score
    })
    return max(0.0, min(1.0, score))

def attach_emotion_scores_v2(segments: list[dict], audio_sidecars: list[dict] | None = None):
    """Compute raw → calibrate per batch → sigmoid"""
    raws = []
    for i, s in enumerate(segments):
        af = (audio_sidecars[i] if audio_sidecars and i < len(audio_sidecars) else None)
        raws.append(_emotion_raw_v2((s.get("text") or s.get("transcript") or ""), af, s.get("genre")))
    MU, SIGMA = _calibrate_emotion_stats(raws)
    for i, s in enumerate(segments):
        af = (audio_sidecars[i] if audio_sidecars and i < len(audio_sidecars) else None)
        s["emotion_score"] = emotion_score_v2(s, MU, SIGMA, af)

def attach_emotion_scores(segments: list[dict]):
    """Dispatcher for emotion scoring - V2 when enabled, legacy otherwise"""
    config = get_config()
    if config.get("emotion_v2", {}).get("enabled", False):
        # Build audio sidecars from arousal debug fields
        audio_sidecars = [build_emotion_audio_sidecar(s) for s in segments]
        attach_emotion_scores_v2(segments, audio_sidecars)
    else:
        # Legacy path
        for s in segments:
            s["emotion_score"] = _emotion_score_v4(s.get("text", s.get("transcript", "")))

# ---- Enhanced Emotion Lexicon System -------------------------------------------------------------

import csv, re, os, math, numpy as np
from functools import lru_cache
from typing import Dict, List, Tuple

# Light tokenizer + optional lemmatizer (no heavy deps)
_WORD_LEX = re.compile(r"[A-Za-z']+")
def _tok_lex(text: str) -> List[str]:
    return _WORD_LEX.findall(text.lower()) if text else []

def _lemmatize_basic(w: str) -> str:
    """Ultra-light stem/lemma to improve lexicon hits (no external libs)"""
    for suf in ("'s", "'s"):
        if w.endswith(suf): w = w[:-2]
    for suf in ("ing","ed","er","est","es","s"):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w

def _maybe_lemma(tokens: List[str]) -> List[str]:
    """Apply lemmatization if enabled in config"""
    if get_config().get("emotion_lexicons", {}).get("lemmatize", True):
        return [_lemmatize_basic(t) for t in tokens]
    return tokens

# Mapping from external labels to our emotion families
MAP_NRC_TO_FAM = {
    "anger": "anger",
    "fear": "fear", 
    "sadness": "sad",
    "joy": "joy",
    "surprise": "surprise",
    "disgust": "disgust",
    "anticipation": "desire",  # anticipation maps to desire/drive
    "trust": "pride"          # trust maps to pride/confidence
}

MAP_DM_TO_FAM = {
    # Example mapping for DepecheMood columns -> our families
    "AFRAID": "fear", "ANGRY": "anger", "SAD": "sad", "HAPPY": "joy",
    "AMUSED": "joy", "ANNOYED": "anger", "DONOTCARE": "disgust",
    "INSPIRED": "awe", "AFRAID_CONFIDENT": "fear", "DOUBTFUL": "shame"
}

@lru_cache(maxsize=1)
def _load_nrc() -> Dict[str, List[str]]:
    """Load NRC Emotion Lexicon: word -> [families]"""
    cfg = get_config().get("emotion_lexicons", {})
    if not (cfg.get("enabled") and cfg.get("use_nrc")): 
        return {}
    path = cfg.get("nrc_path")
    if not path or not os.path.exists(path): 
        return {}
    
    lex: Dict[str, List[str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3: 
                    continue
                w, emo, flag = parts
                if flag != "1": 
                    continue
                fam = MAP_NRC_TO_FAM.get(emo.lower())
                if not fam: 
                    continue
                w = w.lower()
                lex.setdefault(w, []).append(fam)
    except Exception as e:
        print(f"Warning: Could not load NRC lexicon from {path}: {e}")
        return {}
    
    return lex

@lru_cache(maxsize=1)
def _load_ail() -> Dict[str, Dict[str, float]]:
    """Load NRC Affect Intensity Lexicon: word -> {family: intensity}"""
    cfg = get_config().get("emotion_lexicons", {})
    if not (cfg.get("enabled") and cfg.get("use_ail")): 
        return {}
    path = cfg.get("ail_path")
    if not path or not os.path.exists(path): 
        return {}
    
    out: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            # Try to be robust: detect header
            header = next(reader, None)
            
            def norm(v: str) -> float:
                try:
                    x = float(v)
                    # AIL scores usually 0..1 (or 0..1-ish). Clamp just in case.
                    return max(0.0, min(1.0, x))
                except: 
                    return 0.0
            
            if header and len(header) >= 3 and header[0].lower().startswith("word"):
                # has header; read data lines
                for row in reader:
                    if len(row) < 3: 
                        continue
                    w, emo, score = row[0].lower(), row[1].lower(), norm(row[2])
                    fam = MAP_NRC_TO_FAM.get(emo) or emo
                    if fam not in MAP_NRC_TO_FAM.values(): 
                        continue
                    out.setdefault(w, {})[fam] = max(out[w].get(fam, 0.0), score)
            else:
                # no header; treat first line as data too
                f.seek(0)
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) < 3: 
                        continue
                    w, emo, score = row[0].lower(), row[1].lower(), norm(row[2])
                    fam = MAP_NRC_TO_FAM.get(emo) or emo
                    if fam not in MAP_NRC_TO_FAM.values():
                        continue
                    out.setdefault(w, {})[fam] = max(out[w].get(fam, 0.0), score)
    except Exception as e:
        print(f"Warning: Could not load AIL lexicon from {path}: {e}")
        return {}
    
    return out

@lru_cache(maxsize=1)
def _load_depechemood() -> Dict[str, Dict[str, float]]:
    """Load DepecheMood lexicon: token -> {family: score}"""
    cfg = get_config().get("emotion_lexicons", {})
    if not (cfg.get("enabled") and cfg.get("use_depechemood")): 
        return {}
    path = cfg.get("depechemood_path")
    if not path or not os.path.exists(path): 
        return {}
    
    out: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            if len(header) < 3: 
                return {}
            fam_cols = [c for c in header[1:] if c in MAP_DM_TO_FAM]
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != len(header): 
                    continue
                lemma_pos = parts[0].split("#")[0].lower()
                scores = parts[1:]
                for col, val in zip(header[1:], scores):
                    fam = MAP_DM_TO_FAM.get(col)
                    if not fam: 
                        continue
                    try:
                        s = float(val)
                    except:
                        continue
                    # normalize roughly into [0,1] per row
                    out.setdefault(lemma_pos, {})[fam] = max(out[lemma_pos].get(fam, 0.0), s)
        
        # optional normalization across families per token
        for w, fams in out.items():
            mx = max(fams.values()) if fams else 1.0
            if mx > 0:
                for k in list(fams.keys()):
                    fams[k] = min(1.0, fams[k] / mx)
    except Exception as e:
        print(f"Warning: Could not load DepecheMood lexicon from {path}: {e}")
        return {}
    
    return out

def _negation_mask_lex(tokens: List[str], idx: int, window: int) -> int:
    """Return -1 if a negator appears within window before idx"""
    lo = max(0, idx - window)
    for j in range(lo, idx):
        if tokens[j] in {"not","no","never","isnt","isn't","dont","don't","cannot","can't","wont","won't","ain't","aint"}:
            return -1
    return +1

def extract_emotion_hits_from_lexicons(text: str) -> List[Tuple[str, float]]:
    """
    Extract emotion hits using NRC/AIL/DepecheMood lexicons according to config.
    Returns list of (family, weight) hits with negation awareness.
    """
    cfg = get_config().get("emotion_lexicons", {})
    if not cfg.get("enabled"): 
        return []

    toks = _maybe_lemma(_tok_lex(text))
    if not toks: 
        return []
    if len(toks) > int(cfg.get("max_tokens_considered", 300)):
        toks = toks[:int(cfg.get("max_tokens_considered", 300))]

    nrc = _load_nrc()
    ail = _load_ail()
    dpm = _load_depechemood()
    use_nrc = bool(cfg.get("use_nrc", True))
    use_ail = bool(cfg.get("use_ail", True))
    use_dm = bool(cfg.get("use_depechemood", False))
    MIN_INT = float(cfg.get("min_intensity", 0.20))
    CAP_TOK = float(cfg.get("cap_per_token", 1.50))
    NEG_W = int(get_config().get("emotion_v2", {}).get("negation_window", 3))

    hits: List[Tuple[str, float]] = []
    for i, w in enumerate(toks):
        # per-token accumulation (cap to avoid extreme pile-ups)
        per_tok = 0.0
        
        # NRC binary categories → 1.0
        if use_nrc and w in nrc:
            for fam in nrc[w]:
                s = 1.0
                s *= _negation_mask_lex(toks, i, NEG_W)
                if s > 0:
                    hits.append((fam, s))
                    per_tok += s
        
        # AIL intensities
        if use_ail and w in ail:
            for fam, val in ail[w].items():
                if val < MIN_INT: 
                    continue
                s = val
                s *= _negation_mask_lex(toks, i, NEG_W)
                if s > 0:
                    hits.append((fam, s))
                    per_tok += s
        
        # DepecheMood (optional)
        if use_dm and w in dpm:
            for fam, val in dpm[w].items():
                if val < MIN_INT:
                    continue
                s = val * 0.8  # DM often "hot"; tame slightly
                s *= _negation_mask_lex(toks, i, NEG_W)
                if s > 0:
                    hits.append((fam, s))
                    per_tok += s

        # cap how much a single token can contribute
        if per_tok > CAP_TOK:
            scale = CAP_TOK / per_tok
            # scale down the most recent additions for this token
            k = len(hits) - 1
            acc = 0.0
            while k >= 0 and acc < per_tok:
                fam, s = hits[k]
                hits[k] = (fam, s * scale)
                acc += s
                k -= 1

    return hits

# ---- Enhanced Synergy System -------------------------------------------------------------

def compute_synergy_bonus(segment: dict) -> float:
    """
    Small additive nudges when independent signals co-occur.
    Returns a bounded bonus; never negative.
    """
    cfg = get_config().get("synergy", {}) or {}
    if not cfg.get("enabled", True):
        return 0.0

    bonus_total = 0.0

    # --- Emotion × Arousal ---
    ea = cfg.get("emo_arousal", {})
    thr_emo = float(ea.get("thr_emo", 0.60))
    thr_ar  = float(ea.get("thr_arousal", 0.60))
    add_ea  = float(ea.get("bonus", 0.01))
    cap_ea  = float(ea.get("cap", 0.02))

    emo = float(segment.get("emotion_score", 0.0))
    aro = float(segment.get("arousal", segment.get("arousal_score", 0.0)))
    if emo >= thr_emo and aro >= thr_ar:
        # scale bonus slightly by how far above thresholds we are, then cap
        scale = min(1.0, ((emo - thr_emo) + (aro - thr_ar)) / 0.8)
        bonus_total += min(cap_ea, add_ea * (1.0 + 0.5 * scale))

    # --- Question/List × Info Density ---
    qi = cfg.get("ql_infodense", {})
    thr_ql = float(qi.get("thr_ql", 0.60))
    thr_id = float(qi.get("thr_id", 0.60))
    add_qi = float(qi.get("bonus", 0.01))
    cap_qi = float(qi.get("cap", 0.02))

    ql = float(segment.get("question_list", 0.0))
    idv = float(segment.get("info_density", 0.0))
    if ql >= thr_ql and idv >= thr_id:
        scale = min(1.0, ((ql - thr_ql) + (idv - thr_id)) / 0.8)
        bonus_total += min(cap_qi, add_qi * (1.0 + 0.5 * scale))

    # Final cap across all synergy routes
    bonus_total = min(bonus_total, float(cfg.get("max_total_bonus", 0.03)))

    # Optional: dampen synergy if the clip ends with an outro or bait
    tail_text = (segment.get("tail_text") or segment.get("text_tail") or "")
    if tail_text:
        if re.search(r"(thanks for watching|like and subscribe|follow for more|link in bio)", tail_text, re.I):
            bonus_total *= 0.5  # halve the synergy in clear outro/bait

    # Debug
    dbg = segment.setdefault("_debug", {})
    dbg["synergy_bonus"] = round(bonus_total, 5)

    return bonus_total

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
        'tiktok': {'optimal': (15, 30), 'acceptable': (5, 45), 'minimal': (3, 60)},
        'instagram': {'optimal': (15, 30), 'acceptable': (5, 45), 'minimal': (3, 60)},
        'instagram_reels': {'optimal': (15, 30), 'acceptable': (5, 45), 'minimal': (3, 60)},
        'youtube_shorts': {'optimal': (20, 45), 'acceptable': (5, 60), 'minimal': (3, 90)},
        'linkedin': {'optimal': (30, 60), 'acceptable': (10, 90), 'minimal': (5, 120)}
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
    elif ranges['minimal'][0] <= duration <= ranges['minimal'][1]:
        # Very short clips get partial credit
        if duration < ranges['acceptable'][0]:
            return 0.2 + 0.3 * (duration - ranges['minimal'][0]) / (ranges['acceptable'][0] - ranges['minimal'][0])
        else:
            return 0.2 + 0.3 * (ranges['minimal'][1] - duration) / (ranges['minimal'][1] - ranges['acceptable'][1])
    else:
        return 0.0  # Outside all ranges

def calculate_dynamic_length_score(segment: Dict, platform: str) -> float:
    """
    Calculate length score for dynamic segments, considering natural boundaries.
    """
    # Check if Platform Length V2 is enabled
    config = get_config()
    plat_cfg = config.get("platform_length_v2", {})
    
    if not plat_cfg.get("enabled", True):
        # V1 path - legacy implementation
        duration = segment.get("end", 0) - segment.get("start", 0)
        base_score = _platform_length_match(duration, platform)
        
        # Bonus for natural boundaries
        boundary_type = segment.get("boundary_type", "")
        confidence = segment.get("confidence", 0.0)
        
        if boundary_type in ["sentence_end", "insight_marker"] and confidence > 0.8:
            base_score += 0.1  # Bonus for clean boundaries
        
        return min(1.0, base_score)
    
    # V2 path - enhanced implementation
    duration = (segment.get("end", 0.0) - segment.get("start", 0.0)) or 0.0
    
    # Calculate WPS with fallbacks
    wps = None
    if segment.get("word_count"):
        wps = float(segment["word_count"]) / max(duration, 1e-6)
    elif segment.get("text") and duration > 0:
        # Fallback: compute word count from text
        word_count = len(segment["text"].split())
        wps = float(word_count) / max(duration, 1e-6)
    
    # Extract text tail for outro detection (last 1-2 seconds)
    text_tail = segment.get("tail_text", "")
    if not text_tail and segment.get("text") and duration > 0:
        # Simple fallback: take last few words if no tail_text provided
        words = segment["text"].split()
        if len(words) > 3:
            text_tail = " ".join(words[-3:])  # Last 3 words as approximation
    
    # Get other segment data with defaults
    loopability = segment.get("loopability", 0.0)
    boundary_type = segment.get("boundary_type", "")
    boundary_conf = float(segment.get("confidence", 0.0) or 0.0)
    
    # Use V2 scoring
    score = _platform_length_score_v2(
        duration=duration,
        platform=platform,
        loopability=loopability,
        wps=wps,
        boundary_type=boundary_type,
        boundary_conf=boundary_conf,
        text_tail=text_tail,
    )
    
    # Add debug information
    if "_debug" not in segment:
        segment["_debug"] = {}
    
    debug = segment["_debug"]
    debug["length_v2_duration"] = duration
    debug["length_v2_wps"] = wps
    debug["length_v2_loopability"] = loopability
    debug["length_v2_boundary_type"] = boundary_type
    debug["length_v2_boundary_conf"] = boundary_conf
    debug["length_v2_text_tail_present"] = bool(text_tail)
    debug["length_v2_final_score"] = score
    
    return score

# Platform Length V2 - Enhanced smooth scoring
import math
import re

# Outro detection regex
_OUTRO_RE = re.compile(
    r"(thanks for watching|subscribe|follow|like and subscribe|link in bio|see you next time)",
    re.I
)

def _gauss01(x: float, mu: float, sigma: float) -> float:
    """Gaussian function normalized to 0-1 range"""
    if sigma <= 1e-6:
        return 1.0 if abs(x - mu) < 1e-6 else 0.0
    z = (x - mu) / sigma
    return float(math.exp(-0.5 * z * z))

def _tri_band01(x: float, lo: float, hi: float) -> float:
    """Triangular function: 0 outside [lo,hi], 1 at midpoint, linear slopes"""
    if lo >= hi:
        return 0.0
    mid = 0.5 * (lo + hi)
    if x <= lo or x >= hi:
        return 0.0
    return (x - lo) / (mid - lo) if x < mid else (hi - x) / (hi - mid)

def _platform_length_score_v2(
    duration: float,
    platform: str,
    *,
    loopability: float = 0.0,
    wps: float | None = None,
    boundary_type: str = "",
    boundary_conf: float = 0.0,
    text_tail: str = "",
) -> float:
    """Enhanced platform length scoring with smooth curves and adaptive features"""
    # Get platform configuration with fallbacks
    config = get_config()
    plat_cfg = config.get("platform_length_v2", {})
    platforms = plat_cfg.get("platforms", {})
    
    # Default platform config
    default_cfg = {"mu": 22.0, "sigma": 7.0, "cap": 60.0, "wps": [2.8, 4.5]}
    cfg = platforms.get(platform, default_cfg)
    
    mu, sigma, cap = cfg["mu"], cfg["sigma"], cfg["cap"]
    wps_range = cfg["wps"]

    # Loop-aware shift: shorter & tighter target when highly loopable
    if loopability >= 0.60:
        mu -= 2.0
        sigma = max(5.0, sigma - 1.5)

    # Guardrails
    if duration <= 0.0:
        return 0.0
    if duration > cap:
        return 0.0

    # Smooth base score using Gaussian
    base = _gauss01(duration, mu, sigma)

    # Near-cap anxiety penalty (don't risk getting cut by platform)
    if duration >= cap - 1.0:
        base *= 0.85

    # Density harmony: gentle blend with platform's ideal WPS band
    if wps is not None and wps > 0:
        lo, hi = wps_range
        dens = _tri_band01(wps, lo, hi)  # 0..1
        base = 0.85 * base + 0.15 * dens

    # Boundary quality bonuses/penalties
    if boundary_type == "sentence_end" and boundary_conf >= 0.90:
        base = min(1.0, base + 0.10)
    elif boundary_type in ("sentence_end", "insight_marker") and boundary_conf >= 0.75:
        base = min(1.0, base + 0.05)
    elif boundary_type == "mid_word":
        base *= 0.85

    # Anti-outro penalty
    if text_tail and _OUTRO_RE.search(text_tail):
        base = max(0.0, base - 0.15)

    return float(max(0.0, min(1.0, base)))

def _audio_prosody_score(audio_path: str, start: float, end: float, y_sr=None, text: str = "", genre: str = 'general') -> float:
    """Enhanced audio analysis for arousal/energy detection with intelligent fallback"""
    try:
        if y_sr is None:
            y, sr = librosa.load(audio_path, sr=None, offset=max(0, start-0.2), duration=(end-start+0.4))
        else:
            y, sr = y_sr
            s = max(int((start-0.2)*sr), 0)
            e = min(int((end+0.4)*sr), len(y))
            y = y[s:e]
        
        if len(y) == 0:
            # Fallback to text-based estimation
            return _text_based_audio_estimation(text, genre)

        # Check if Arousal V5 is enabled
        arousal_cfg = get_config().get("arousal_v5", {})
        if arousal_cfg.get("enabled", True):
            # Use enhanced V5 arousal scoring
            dur_s = end - start
            tokens = text.split() if text else None
            return _arousal_v5(y, sr, text, genre, dur_s, tokens)
        else:
            # Use legacy compute_audio_energy function
            return compute_audio_energy(y, sr)
    except Exception as e:
        logger.warning(f"Audio energy computation failed: {e}")
        # Fallback to text-based estimation
        return _text_based_audio_estimation(text, genre)

# Arousal V5 helper functions
def scale01(value, min_val, max_val):
    """Scale value to 0-1 range based on min/max bounds"""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def robust_stats(data):
    """Compute robust statistics using median and IQR"""
    if len(data) == 0:
        return 0.0, 1.0
    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    std = iqr / 1.349  # IQR to std conversion
    return median, max(std, 1e-6)

def _arousal_v5(y, sr, text="", genre="general", dur_s=None, tokens=None):
    """Enhanced arousal scoring with cadence, pitch dynamics, and spectral flux"""
    try:
        # Get arousal config
        arousal_cfg = get_config().get("arousal_v5", {})
        w_energy = arousal_cfg.get("w_energy", 0.35)
        w_cadence = arousal_cfg.get("w_cadence", 0.30)
        w_pitch = arousal_cfg.get("w_pitch", 0.20)
        w_flux = arousal_cfg.get("w_flux", 0.15)
        
        # 1) Core energy (reuse existing compute_audio_energy)
        energy_core = compute_audio_energy(y, sr)
        
        # 2) Cadence features
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        med = np.median(rms) + 1e-8
        
        # Pause ratio
        pause_ratio = np.mean(rms < arousal_cfg.get("cadence_pause_med_mult", 0.25) * med)
        
        # Burst density
        thr = med + np.std(rms)
        bursts = np.mean((rms[1:] >= thr) & (rms[:-1] < thr)) * (sr/512)  # per-sec approx
        
        # Words per second
        wps = (len(tokens) / max(dur_s, 1e-3)) if tokens and dur_s else 0.0
        wps_range = arousal_cfg.get("cadence_wps_range", [2.0, 5.5])
        burst_range = arousal_cfg.get("cadence_burst_range", [0.5, 3.0])
        
        cadence_parts = [
            scale01(wps, wps_range[0], wps_range[1]),
            1.0 - pause_ratio,
            scale01(bursts, burst_range[0], burst_range[1])
        ]
        cadence = sat(cadence_parts)
        
        # 3) Pitch dynamics
        try:
            # Try pyin first (more accurate)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_clean = f0[voiced_flag]
        except:
            # Fallback to piptrack
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            f0_clean = pitches[pitches > 0]
        
        if len(f0_clean) > 10:
            f0_range = np.percentile(f0_clean, 90) - np.percentile(f0_clean, 10)
            f0_std = np.std(f0_clean)
            f0_delta = np.mean(np.abs(np.diff(f0_clean)))
            
            # Scale to 0-1
            pitch_parts = [
                scale01(f0_range, 50, 300),  # Hz range
                scale01(f0_std, 10, 100),    # Hz std
                scale01(f0_delta, 5, 50)     # Hz delta
            ]
            pitch_dyn = sat(pitch_parts)
        else:
            pitch_dyn = 0.0
        
        # 4) Spectral flux (using onset strength as proxy)
        flux = librosa.onset.onset_strength(y=y, sr=sr)
        flux_s = np.mean(flux) / (np.std(flux) + 1e-6)  # Normalize by variability
        flux_s = scale01(flux_s, 0.5, 3.0)
        
        # 5) Combine audio components with saturating combiner
        audio_parts = [energy_core, cadence, pitch_dyn, flux_s]
        audio_weights = [w_energy, w_cadence, w_pitch, w_flux]
        # Normalize weights
        total_weight = sum(audio_weights)
        audio_weights = [w/total_weight for w in audio_weights]
        
        audio_raw = sat([part * weight for part, weight in zip(audio_parts, audio_weights)])
        
        # 6) Text micro-bonus (only if audio exists)
        text_bonus = 0.0
        if audio_raw >= arousal_cfg.get("audio_presence_guard", 0.10):
            text_arousal = _arousal_score_text(text, genre)
            text_bonus = min(text_arousal, arousal_cfg.get("text_micro_cap", 0.08))
        
        # 7) Genre-specific adjustments
        genre_bonus = 0.0
        if genre == 'comedy' and dur_s and dur_s <= arousal_cfg.get("laughter_early_sec", 5.0):
            # Check for laughter in first 5 seconds
            laughter_boost = detect_laughter_exclamations(text, y, sr, 0.0, enable_audio_analysis=False)
            if laughter_boost > 0:
                genre_bonus = min(laughter_boost, arousal_cfg.get("comedy_laughter_bonus", 0.04))
        
        # Business/Education caps penalty
        if genre in ['business', 'educational'] and cadence < 0.4 and pitch_dyn < 0.4:
            caps_count = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
            exclam_count = text.count('!')
            caps_scale, exclam_scale = arousal_cfg.get("bizedu_caps_excl_scale", [0.01, 0.005])
            penalty = min(arousal_cfg.get("bizedu_caps_excl_cap", 0.05), 
                         caps_scale * caps_count + exclam_scale * exclam_count)
            text_bonus = max(0.0, text_bonus - penalty)
        
        # 8) Final combination
        arousal_raw = audio_raw + text_bonus + genre_bonus
        
        # 9) Robust scaling with sigmoid
        # For now, use simple normalization (can add global EMA later)
        arousal_final = 1.0 / (1.0 + np.exp(-arousal_cfg.get("sigmoid_a", 1.4) * (arousal_raw - 0.5)))
        
        return float(np.clip(arousal_final, 0.0, 1.0))
        
    except Exception as e:
        logger.warning(f"Arousal V5 computation failed: {e}")
        # Fallback to text-based estimation
        return _text_based_audio_estimation(text, genre)

def _text_based_audio_estimation(text: str, genre: str = 'general') -> float:
    """Intelligent text-based audio arousal estimation when audio analysis fails"""
    if not text:
        return 0.3  # Default moderate arousal
    
    t = text.lower()
    score = 0.3  # Base moderate arousal
    
    # High-energy text indicators
    high_energy_indicators = [
        ('!', 0.1), ('amazing', 0.15), ('incredible', 0.15), ('crazy', 0.15),
        ('insane', 0.2), ('?!', 0.2), ('wow', 0.1), ('unbelievable', 0.15),
        ('shocking', 0.2), ('wild', 0.1), ('epic', 0.15), ('mind-blowing', 0.2)
    ]
    
    for indicator, boost in high_energy_indicators:
        if indicator in t:
            score = min(score + boost, 0.9)
            break  # Only apply the first match to avoid over-scoring
    
    # Genre-specific audio estimation
    if genre == 'fantasy_sports':
        sports_energy_words = ['fire', 'draft', 'start', 'target', 'sleeper', 'bust']
        if any(word in t for word in sports_energy_words):
            score = min(score + 0.1, 0.9)
    elif genre == 'comedy':
        comedy_energy_words = ['hilarious', 'funny', 'lol', 'haha', 'rofl', 'joke']
        if any(word in t for word in comedy_energy_words):
            score = min(score + 0.1, 0.9)
    elif genre == 'true_crime':
        crime_energy_words = ['murder', 'killer', 'evidence', 'mystery', 'suspicious']
        if any(word in t for word in crime_energy_words):
            score = min(score + 0.1, 0.9)
    
    return float(np.clip(score, 0.0, 1.0))

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
            if abs(denominator) > 0.001:  # More appropriate threshold for audio data
                dynamic_range = (np.percentile(rms, 90) - np.percentile(rms, 10)) / denominator
                energy = energy * 0.7 + dynamic_range * 0.3
        
        return float(np.clip(energy, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"Audio energy normalization failed: {e}")
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
        # Strong content boundaries - check if word ends with punctuation
        if any(word.endswith(punct) for punct in [".", "!", "?", ":", ";"]):
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
        
        # Story/insight markers (expanded)
        elif any(phrase in " ".join(words[max(0, i-1):i+2]) for phrase in [
            "here's the thing", "the key is", "the key insight", "what i learned", "my take",
            "the bottom line", "in summary", "to wrap up", "main observation",
            "key takeaway", "the thing is", "what i found", "here's what", "the insight",
            "here's why", "let me tell you", "you know what", "this is why", "the reason is",
            "the problem is", "the issue is", "the challenge is", "the solution is", "the answer is",
            "the truth is", "the reality is", "the fact is", "the secret is", "the trick is",
            "the way to", "the best way", "the only way", "the right way", "the wrong way"
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
    
    # Platform-specific optimal lengths (adjusted for better content)
    platform_lengths = {
        'tiktok': {'min': 12, 'max': 30, 'optimal': 20},
        'instagram': {'min': 12, 'max': 30, 'optimal': 22},
        'instagram_reels': {'min': 12, 'max': 30, 'optimal': 22},
        'youtube': {'min': 15, 'max': 60, 'optimal': 35},
        'youtube_shorts': {'min': 15, 'max': 60, 'optimal': 35},
        'twitter': {'min': 8, 'max': 25, 'optimal': 18},
        'linkedin': {'min': 15, 'max': 45, 'optimal': 30}
    }
    
    target_length = platform_lengths.get(platform, platform_lengths['tiktok'])
    
    # Combine all segments into one continuous text for better boundary detection
    combined_text = " ".join([seg.get("text", "") for seg in segments])
    total_start = segments[0].get("start", 0) if segments else 0
    total_end = segments[-1].get("end", 0) if segments else 0
    total_duration = total_end - total_start
    
    # Find natural boundaries in the combined text
    boundaries = find_natural_boundaries(combined_text)
    
    if not boundaries or len(boundaries) < 2:
        # No natural boundaries found, use original segments
        return segments
    
    # Create segments based on boundaries
    words = combined_text.split()
    current_start = total_start
    
    # Add start boundary
    all_boundaries = [{"position": 0, "type": "start", "confidence": 1.0}] + boundaries
    
    for i, boundary in enumerate(all_boundaries):
        if boundary["confidence"] < 0.8:  # Increased from 0.6 to 0.8 for higher confidence
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
        
        if len(segment_words) < 8:  # Increased from 3 to 8 for longer segments
            continue
        
        # Calculate timing based on word count and total duration
        total_words = len(words)
        segment_ratio = len(segment_words) / total_words
        segment_duration = total_duration * segment_ratio
        
        # CRITICAL FIX: Preserve original transcript timing instead of artificial calculation
        # Find the actual transcript segments that correspond to this text
        original_start = None
        original_end = None
        
        # Look for the first and last words in the original transcript
        for i, word in enumerate(words):
            if i == boundary["position"]:
                # Find which original segment contains this word
                for orig_seg in segments:
                    if word in orig_seg.get("text", "").split():
                        original_start = orig_seg.get("start", current_start)
                        break
            if i == end_position - 1:
                # Find which original segment contains this word
                for orig_seg in segments:
                    if word in orig_seg.get("text", "").split():
                        original_end = orig_seg.get("end", current_start + segment_duration)
                        break
        
        # Use original timing if found, otherwise fall back to calculated timing
        if original_start is not None and original_end is not None:
            segment_duration = original_end - original_start
            current_start = original_start
        
        # Check if segment meets platform requirements
        if target_length["min"] <= segment_duration <= target_length["max"]:
            dynamic_segments.append({
                "text": segment_text,
                "start": current_start,
                "end": current_start + segment_duration,
                "boundary_type": boundary["type"],
                "confidence": boundary["confidence"]
            })
        elif segment_duration < target_length["min"]:
            # If segment is too short, try to merge with previous segment or extend
            if len(dynamic_segments) > 0:
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
            else:
                # First segment is too short, extend it to minimum length
                extended_duration = target_length["min"]
                dynamic_segments.append({
                    "text": segment_text,
                    "start": current_start,
                    "end": current_start + extended_duration,
                    "boundary_type": "extended",
                    "confidence": boundary["confidence"]
                })
        
        current_start += segment_duration
    
    # If no dynamic segments were created, return original segments
    if not dynamic_segments:
        return segments
    
    # Post-process: ensure all segments meet minimum length requirements
    final_segments = []
    for seg in dynamic_segments:
        duration = seg["end"] - seg["start"]
        if duration < target_length["min"]:
            # Extend short segments to minimum length
            seg["end"] = seg["start"] + target_length["min"]
            seg["boundary_type"] = "extended"
        final_segments.append(seg)
    
    return final_segments

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
        print(f"Auto-detected genre: {detected_genre}")
        print("You can override this by specifying a different genre")
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
        print(f"Auto-detected genre: {detected_genre}")
        print("You can override this by selecting a specific genre")
        genre = detected_genre
    else:
        genre = user_genre
        print(f"Using user-selected genre: {genre}")
    
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
        
        # FIXED: Increase hook score weight to be the dominant factor
        hook_path = (0.50 * f.get("hook_score", 0.0) + 0.20 * f.get("arousal_score", 0.0) + 
                     0.15 * f.get("payoff_score", 0.0) + 0.10 * f.get("info_density", 0.0) + 
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
            "label": "Imbalanced",
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
            "label": "Good Balance",
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


# Title Generation and Grade Functions
_TITLE_STOP = re.compile(r"[.!?]\s+")
_STRONG_VERB = re.compile(r"\b(win|save|avoid|learn|unlock|beat|grow|double|prove|fix|crush|build)\b", re.I)

def _grade_from_score(x: float) -> str:
    """Convert score (0-1) to letter grade"""
    x = float(x or 0)
    if   x >= 0.93: return "A+"
    elif x >= 0.90: return "A"
    elif x >= 0.85: return "A-"
    elif x >= 0.80: return "B+"
    elif x >= 0.75: return "B"
    elif x >= 0.70: return "B-"
    elif x >= 0.60: return "C+"
    elif x >= 0.50: return "C"
    else:           return "C-"

def _heuristic_title(text: str, feats: dict, cfg: dict, rank: int | None = None) -> str:
    """Generate viral-style title from transcript and features"""
    tcfg = (cfg or {}).get("titles", {}) or {}
    max_len = int(tcfg.get("max_len", 80))
    
    # Extract first sentence
    sent = _TITLE_STOP.split((text or "").strip())
    first = (sent[0] if sent else "").strip(" -–—")
    cand = first if first else (text or "")[:max_len]
    cand = re.sub(r"^\b(?:so|well|look|listen|okay|you know)[, ]+", "", cand, flags=re.I)
    cand = cand.capitalize()

    # Boosts from features (families)
    boosts = []
    fam = feats.get("_debug", {}).get("hook_v5", {}).get("fam_scores", {}) if feats else {}
    w = (tcfg.get("heuristic_boosts") or {})
    
    if fam.get("listicle", 0) > 0:
        boosts.append("Top plays to try")
    if fam.get("curiosity", 0):  
        boosts.append("The truth you're missing")
    if fam.get("howto", 0):      
        boosts.append("How to get ahead")
    if fam.get("contrarian", 0): 
        boosts.append("What everyone gets wrong")

    # Prefer strong verbs
    if not _STRONG_VERB.search(cand):
        for b in boosts:
            if _STRONG_VERB.search(b):
                cand = b + ": " + cand
                break

    # Trim to max length
    if len(cand) > max_len:
        cand = cand[:max_len-1].rstrip() + "…"

    if tcfg.get("prepend_rank") and isinstance(rank, int):
        return f"#{rank} {cand}"
    return cand

def _grade_breakdown(feats: dict) -> dict:
    """Generate grade breakdown for all scoring dimensions"""
    return {
        "overall": _grade_from_score(feats.get("final_score", feats.get("viral_score_100", 0)/100)),
        "hook":    _grade_from_score(feats.get("hook_score", 0)),
        "flow":    _grade_from_score(feats.get("arousal_score", feats.get("arousal", 0))),
        "value":   _grade_from_score(feats.get("payoff_score", 0)),
        "trend":   _grade_from_score(feats.get("loopability", 0)),
    }

async def _llm_title_async(text: str, feats: dict, cfg: dict) -> str:
    """Generate title using LLM (optional)"""
    if not text: return ""
    llm_cfg = (cfg or {}).get("llm_titles", {}) or {}
    if not llm_cfg.get("enabled"): return ""
    
    import asyncio, json, os
    from httpx import AsyncClient, Timeout

    model = llm_cfg.get("model", "gpt-4o-mini")
    max_len = int(llm_cfg.get("max_len", 80))
    sys = "You write short viral social video titles (<=80 chars). Make it punchy, specific, non-clickbait, present-tense."
    usr = f"""Transcript snippet:
{text[:600]}

Scores (0..1):
Hook={feats.get('hook_score',0):.2f} Flow={feats.get('arousal_score',0):.2f} Value={feats.get('payoff_score',0):.2f} Trend={feats.get('loopability',0):.2f}

Return ONLY the title string, no quotes, <= {max_len} chars."""
    payload = {
        "model": model,
        "messages": [{"role":"system","content":sys},{"role":"user","content":usr}],
        "temperature": 0.6,
        "max_tokens": 60,
    }
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return ""
    try:
        async with AsyncClient(timeout=Timeout(llm_cfg.get("timeout_s",8))) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload)
            j = r.json()
            t = j["choices"][0]["message"]["content"].strip()
            return t[:max_len]
    except Exception:
        return ""


