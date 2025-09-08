from typing import Dict
import re
from .genres import GenreProfile

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
