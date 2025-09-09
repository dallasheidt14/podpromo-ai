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

class GenreAwareScorer:
    """Main genre-aware scoring system"""
    def __init__(self):
        from .genre_profiles import FantasySportsGenreProfile, ComedyGenreProfile
        
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
            'health_wellness': ['doctor', 'doctors', 'health', 'wellness', 'fitness', 'nutrition', 'mental', 'transformation', 'before after', 'myth', 'tip', 'advice', 'natural', 'remedy', 'cure', 'treatment', 'therapy', 'healing', 'wellness', 'lifestyle'],
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
            'true_crime': ['case', 'evidence', 'murder', 'investigation', 'police', 'detective', 'victim', 'suspect', 'crime', 'mystery', 'disappeared', 'missing'],
            'business': ['company', 'startup', 'revenue', 'investment', 'business', 'entrepreneur', 'million', 'dollar', 'profit', 'strategy', 'market', 'industry'],
            'news_politics': ['breaking', 'announcement', 'government', 'political', 'election', 'policy', 'controversy', 'scandal', 'official', 'statement'],
            'education': ['science', 'research', 'study', 'discovered', 'learn', 'fact', 'scientist', 'experiment', 'theory', 'mind-blowing', 'did you know'],
            'health_wellness': ['doctor', 'doctors', 'health', 'wellness', 'fitness', 'nutrition', 'mental', 'transformation', 'before after', 'myth', 'tip', 'advice', 'natural', 'remedy', 'cure', 'treatment', 'therapy', 'healing', 'wellness', 'lifestyle'],
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
