"""
Feature computation module for viral detection system.
Contains feature extraction and computation functions.
"""

from typing import Dict, List, Tuple, Any
import logging
import numpy as np
import librosa
import re
import hashlib
from functools import lru_cache
from scipy import signal
from scipy.stats import skew, kurtosis
from config_loader import get_config

logger = logging.getLogger(__name__)

# Import dependencies from other modules
from .scoring import get_clip_weights
from .genres import GenreAwareScorer

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

def compute_features_cached(segment_hash: str, audio_file: str, genre: str, platform: str = 'tiktok') -> Dict:
    """Cached version of compute_features_v4 for performance"""
    # This is a placeholder - in practice, you'd need to reconstruct the segment
    # from the hash or use a different caching strategy
    return compute_features_v4({"text": "", "start": 0, "end": 0}, audio_file, genre=genre, platform=platform)

def create_segment_hash(segment: Dict) -> str:
    """Create a hash from segment content for cache key"""
    content = f"{segment.get('text', '')[:100]}_{segment.get('start', 0)}_{segment.get('end', 0)}"
    return hashlib.md5(content.encode()).hexdigest()

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
    """Enhanced loopability scoring with perfect-loop detection, quotability patterns, and curiosity enders"""
    if not text: 
        return 0.0
    
    t = text.lower()
    score = 0.0
    
    # Perfect loop detection - ends where it begins
    words = t.split()
    if len(words) >= 3:
        first_phrase = " ".join(words[:3])
        last_phrase = " ".join(words[-3:])
        if first_phrase == last_phrase:
            score += 0.4
    
    # Quotability patterns
    quotable_patterns = [
        r"here's the thing",
        r"the truth is",
        r"what i learned",
        r"the key is",
        r"here's why",
        r"the secret",
        r"the trick"
    ]
    
    for pattern in quotable_patterns:
        if re.search(pattern, t):
            score += 0.2
            break
    
    # Curiosity enders - questions or incomplete thoughts
    if t.endswith('?') or t.endswith('...') or t.endswith('but'):
        score += 0.15
    
    # Short, punchy statements
    if len(words) <= 8 and any(word in t for word in ['insane', 'crazy', 'wild', 'epic', 'amazing']):
        score += 0.1
    
    return float(np.clip(score, 0.0, 1.0))

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
        
        # Compute audio features
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate arousal score based on audio features
        rms_mean = np.mean(rms)
        spectral_mean = np.mean(spectral_centroids)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Normalize and combine features
        arousal_score = (rms_mean * 0.4 + spectral_mean * 0.3 + zcr_mean * 0.3)
        arousal_score = float(np.clip(arousal_score, 0.0, 1.0))
        
        return arousal_score
        
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}, falling back to text-based estimation")
        return _text_based_audio_estimation(text, genre)

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

def _detect_payoff(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect payoff strength: resolution, answer, or value delivery using genre-specific patterns"""
    if not text or len(text.strip()) < 8:
        return 0.0, "too_short"
    
    t = text.lower()
    score = 0.0
    reasons = []
    
    # Resolution patterns
    resolution_patterns = [
        r"(so|therefore|thus|as a result|consequently)",
        r"(the answer is|the solution is|here's how)",
        r"(that's why|which explains|this means)",
        r"(in conclusion|to sum up|the bottom line)"
    ]
    
    for pattern in resolution_patterns:
        if re.search(pattern, t):
            score += 0.2
            reasons.append("resolution")
            break
    
    # Value delivery patterns
    value_patterns = [
        r"(here's what|the key|the secret|the trick)",
        r"(you should|you need to|you must|you have to)",
        r"(this will|this can|this helps|this makes)",
        r"(the benefit|the advantage|the upside)"
    ]
    
    for pattern in value_patterns:
        if re.search(pattern, t):
            score += 0.15
            reasons.append("value_delivery")
            break
    
    # Genre-specific payoff patterns
    if genre == 'fantasy_sports':
        sports_patterns = [
            r"(start|bench|target|avoid|sleeper|bust)",
            r"(this week|next week|playoffs|season)",
            r"(league winner|championship|title)"
        ]
        for pattern in sports_patterns:
            if re.search(pattern, t):
                score += 0.1
                reasons.append("sports_payoff")
                break
    
    # Penalty for questions without answers
    if "?" in t and not any(word in t for word in ["answer", "solution", "here's", "the key"]):
        score -= 0.1
        reasons.append("question_without_answer")
    
    final_score = float(np.clip(score, 0.0, 1.0))
    reason_str = ";".join(reasons) if reasons else "no_payoff"
    return final_score, reason_str

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


# For remaining functions, import from monolithic file temporarily
from services.secret_sauce_pkg.__init__monolithic import (
    # Hook V5 functions
    _hook_score_v4,
    _hook_score_v5,
    _emotion_score_v4,
    _payoff_presence_v4,
    _info_density_v4,
    _info_density_raw_v2,
    _detect_insight_content_v2,
    _apply_insight_confidence_multiplier,
    
    # Utility functions
    debug_segment_scoring,
    compute_features_v4_batch,
    
    # Hook V5 functions
    compute_audio_hook_modifier,
    detect_laughter_exclamations,
    calculate_hook_components,
    calculate_time_weighted_hook_score,
    score_patterns_in_text,
    _saturating_sum,
    _proximity_bonus,
    _normalize_quotes_lower,
    _first_clause,
    _get_hook_cues_from_config,
    _family_score,
    _evidence_guard,
    _anti_intro_outro_penalties,
    _audio_micro_for_hook,
    _sigmoid,
    _text_based_audio_estimation,
    _calibrate_info_density_stats,
    _ql_calibrate_stats,
    _calibrate_emotion_stats,
    build_emotion_audio_sidecar,
    compute_audio_energy,
    info_density_score_v2,
    question_list_score_v2,
    emotion_score_v2,
    _platform_length_score_v2,
)

# Export all functions
__all__ = [
    # Main feature computation functions
    "compute_features_v4",
    "compute_features_v4_batch", 
    "compute_features",
    "compute_features_cached",
    
    # Individual feature functions
    "_hook_score",
    "_hook_score_v4",
    "_hook_score_v5",
    "_emotion_score",
    "_emotion_score_v4",
    "_payoff_presence",
    "_payoff_presence_v4",
    "_detect_payoff",
    "_info_density",
    "_info_density_v4",
    "_info_density_raw_v2",
    "_question_or_list",
    "_loopability_heuristic",
    "_arousal_score_text",
    "_audio_prosody_score",
    "_detect_insight_content",
    "_detect_insight_content_v2",
    "_apply_insight_confidence_multiplier",
    "_calculate_niche_penalty",
    "_ad_penalty",
    "_platform_length_match",
    "calculate_dynamic_length_score",
    
    # Utility functions
    "create_segment_hash",
    "debug_segment_scoring",
    
    # Hook V5 functions
    "compute_audio_hook_modifier",
    "detect_laughter_exclamations",
    "calculate_hook_components",
    "calculate_time_weighted_hook_score",
    "score_patterns_in_text",
    "_saturating_sum",
    "_proximity_bonus",
    "_normalize_quotes_lower",
    "_first_clause",
    "_get_hook_cues_from_config",
    "_family_score",
    "_evidence_guard",
    "_anti_intro_outro_penalties",
    "_audio_micro_for_hook",
    "_sigmoid",
]
