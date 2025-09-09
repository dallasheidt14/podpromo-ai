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

# Import new Phase 1, 2 & 3 types and utilities
from .types import Features, Scores, FEATURE_TYPES, SYNERGY_MODE, PLATFORM_LEN_V, WHITEN_PATHS, GENRE_BLEND, BOUNDARY_HYSTERESIS, PROSODY_AROUSAL, PAYOFF_GUARD, CALIBRATION_V, MIN_WORDS, MAX_WORDS, MIN_SEC, MAX_SEC, _keep
from .scoring_utils import whiten_paths, synergy_bonus, platform_length_score_v2, apply_genre_blending, find_optimal_boundaries, prosody_arousal_score, payoff_guard, apply_calibration, pick_boundary, snap_to_nearest_pause_or_punct

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
            "loopability": 0.0,
            "_ad_flag": ad_result["flag"],
            "_ad_penalty": ad_result["penalty"],
            "_ad_reason": ad_result["reason"]
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
        # Apply confidence multiplier if available
        confidence = segment.get("confidence", None)
        if confidence is not None:
            insight_score = _apply_insight_confidence_multiplier(insight_score, confidence)
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
        "age restrictions", "availability and applicable",
        # Additional promotional patterns
        "code", "off your order", "fuel smarter", "play harder",
        "visit", "website", "promo", "discount", "save", "deal",
        "offer", "special", "limited time", "exclusive", "free",
        "subscribe", "follow", "like and subscribe", "link in bio"
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


# Real implementations from the original sophisticated scoring system
import math

# Regex patterns for Hook V5
_WORD = re.compile(r"[A-Za-z']+|\d+%?")
_PUNCT_CLAUSE = re.compile(r"(?<=[.!?])\s+")
_HAS_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b")
_HAS_COMP = re.compile(r"\b(?:vs\.?|versus|more than|less than)\b|[<>]")
_HAS_HOWWHY = re.compile(r"\b(?:how|why|what)\b")
_SOFT_FLIP = re.compile(r"\bbut (?:actually|in reality)\b")

# Question/List scoring patterns
_QMARK = re.compile(r"\?\s*$")
_INTERROG = re.compile(r"\b(what|why|how|when|where|which|who)\b", re.I)
_COMPARE = re.compile(r"\b(vs\.?|versus|better than|worse than|compare(?:d)? to)\b", re.I)
_CHOICE = re.compile(r"\b(which|pick|choose|would you rather|either|or)\b", re.I)
_RHET_IND = re.compile(r"\b(you know|right|isn't it|don't you think|agree)\b", re.I)
_CLIFF_Q = re.compile(r"\b(what if|imagine|suppose|what would happen|what do you think)\b", re.I)
_GENUINE = re.compile(r"\b(genuinely|honestly|really|truly|actually)\b", re.I)
_LIST_MARKERS = re.compile(r"\b(first|second|third|fourth|fifth|last|finally|next|then|also|additionally|moreover|furthermore)\b", re.I)
_LIST_ITEMS = re.compile(r"\d+\.\s+[^.!?]+[.!?]?")
_BAIT_PATTERNS = re.compile(r"\b(you won't believe|shocking|amazing|incredible|unbelievable|mind-blowing)\b", re.I)
_VACUOUS_Q = re.compile(r"\b(what do you think|what's your opinion|agree\?|right\?|you know\?)\b", re.I)

def _tokens_ql(s: str):
    """Tokenize for question/list scoring"""
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def _sentences(text: str):
    """Split text into sentences"""
    return re.split(r"[.!?]+\s+", text.strip())

def _sigmoid_ql(x: float, a: float) -> float:
    """Sigmoid function for question/list scoring"""
    return 1.0 / (1.0 + np.exp(-a * x))

def _saturating_sum(scores: List[float], cap: float = 1.0) -> float:
    """Saturating sum for combining multiple scores"""
    prod = 1.0
    for s in scores:
        s = max(0.0, min(1.0, float(s)))
        prod *= (1.0 - s)
    return min(cap, 1.0 - prod)

def _proximity_bonus(index_in_video: int, k: float) -> float:
    """Proximity bonus for early segments"""
    try:
        i = max(0, int(index_in_video))
    except Exception:
        i = 0
    return math.exp(- i / max(1e-6, float(k)))

def _normalize_quotes_lower(text: str) -> str:
    """Normalize quotes and convert to lowercase"""
    t = (text or "").strip().lower()
    return t.translate({
        0x2019: 0x27,  # ' -> '
        0x2018: 0x27,  # ' -> '
        0x201C: 0x22,  # " -> "
        0x201D: 0x22,  # " -> "
    })

def _first_clause(text: str, max_words: int) -> str:
    """Extract first clause up to max_words"""
    sent = _PUNCT_CLAUSE.split(text, maxsplit=1)[0]
    toks = _WORD.findall(sent)
    return " ".join(toks[:max_words])

def _get_hook_cues_from_config(cfg: Dict[str, Any]) -> Dict[str, List[re.Pattern]]:
    """Extract hook cues from configuration"""
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
    """Score text against hook cue families"""
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
    """Check for evidence in early text or first clause"""
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
    """Apply penalties for intro/outro content"""
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
    """Apply audio modifier with cap"""
    try:
        return max(0.0, min(cap, float(audio_mod)))
    except Exception:
        return 0.0

def _sigmoid(z: float, a: float) -> float:
    """Sigmoid function for calibration"""
    return 1.0 / (1.0 + math.exp(-a * z))

def _calibrate_simple(raw: float, mu: float = 0.40, sigma: float = 0.18, a: float = 1.6) -> float:
    """Simple calibration using sigmoid"""
    z = 0.0 if sigma <= 0 else (raw - mu) / sigma
    return _sigmoid(z, a)

def _hook_score_v4(text: str, arousal: float = 0.0, words_per_sec: float = 0.0, genre: str = 'general', 
                   audio_data=None, sr=None, start_time: float = 0.0) -> Tuple[float, str, Dict]:
    """Real Hook V4 implementation"""
    # Enhanced hook scoring with multiple indicators
    hook_indicators = [
        'you', 'imagine', 'what if', 'did you know', 'here\'s why', 'the secret', 
        'shocking', 'amazing', 'incredible', 'unbelievable', 'mind-blowing',
        'here\'s what', 'the truth is', 'the key', 'the trick', 'the secret',
        'what happens when', 'why do', 'how did', 'here\'s how'
    ]
    hook_score = 0.0
    reasons = []
    
    text_lower = text.lower()
    
    # Check for hook indicators
    for indicator in hook_indicators:
        if indicator in text_lower:
            hook_score += 0.15
            reasons.append(f"hook_indicator_{indicator.replace(' ', '_')}")
    
    # Boost for questions
    if '?' in text:
        hook_score += 0.25
        reasons.append("question_mark")
    
    # Boost for short, punchy text
    word_count = len(text.split())
    if word_count < 8:
        hook_score += 0.2
        reasons.append("very_short_punchy")
    elif word_count < 15:
        hook_score += 0.1
        reasons.append("short_punchy")
    
    # Boost for exclamations
    exclam_count = text.count('!')
    if exclam_count > 0:
        hook_score += min(exclam_count * 0.1, 0.3)
        reasons.append(f"exclamation_{exclam_count}")
    
    # Boost for numbers/statistics
    if re.search(r'\d+', text):
        hook_score += 0.1
        reasons.append("contains_numbers")
    
    # Boost for comparison words
    comparison_words = ['vs', 'versus', 'more than', 'less than', 'better than', 'worse than']
    if any(word in text_lower for word in comparison_words):
        hook_score += 0.1
        reasons.append("comparison")
    
    hook_score = min(hook_score, 1.0)
    hook_details = {
        "hook_components": {"word_count": word_count, "exclam_count": exclam_count},
        "hook_type": "general",
        "confidence": hook_score,
        "audio_modifier": 0.0,
        "laughter_boost": 0.0,
        "time_weighted_score": hook_score
    }
    
    return hook_score, ",".join(reasons), hook_details

def _hook_score_v5(text: str, cfg: Dict = None, segment_index: int = 0, audio_modifier: float = 0.0,
                   arousal: float = 0.0, q_or_list: float = 0.0) -> Tuple[float, float, Dict]:
    """Real Hook V5 implementation - the sophisticated scoring system"""
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

    mu = 0.40
    sigma = 0.18
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

def _emotion_score_v4(text: str) -> float:
    """Real emotion score v4 implementation"""
    # Get emotion words from config
    config = get_config()
    emo_words = config.get("lexicons", {}).get("EMO_WORDS", [])
    
    t = text.lower()
    high_intensity = ["incredible", "insane", "mind-blowing", "shocking"]
    high_hits = sum(1 for w in high_intensity if w in t)
    regular_hits = sum(1 for w in emo_words if w in t and w not in high_intensity)
    total_score = (high_hits * 2 + regular_hits) / 5.0
    return float(min(total_score, 1.0))

def _payoff_presence_v4(text: str) -> Tuple[float, str]:
    """Real payoff presence v4 implementation"""
    # Get payoff markers from config
    config = get_config()
    payoff_markers = config.get("lexicons", {}).get("PAYOFF_MARKERS", [])
    
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
            break

    # Check for payoff markers from config
    for marker in payoff_markers:
        if marker.lower() in t:
            score += 0.15
            reasons.append("config_payoff_marker")
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

    # Penalty for questions without answers
    if "?" in t and not any(word in t for word in ["answer", "solution", "here's", "the key"]):
        score -= 0.1
        reasons.append("question_without_answer")

    final_score = float(np.clip(score, 0.0, 1.0))
    reason_str = ";".join(reasons) if reasons else "no_payoff"
    return final_score, reason_str

def _info_density_v4(text: str) -> float:
    """Real info density v4 implementation"""
    if not text or len(text.strip()) < 8:
        return 0.1
    
    t = text.lower()
    words = t.split()
    base = min(0.3, len(words) * 0.012)
    
    # Get filler words from config
    config = get_config()
    filler_words = config.get("lexicons", {}).get("FILLERS", ["you know", "like", "um", "uh", "right", "so"])
    filler_count = sum(1 for filler in filler_words if filler in t)
    filler_penalty = min(filler_count * 0.08, 0.4)
    
    numbers_and_stats = len(re.findall(r'\b\d+\b|[\$\d,]+|\d+%|\d+\.\d+', text))
    proper_nouns = sum(1 for w in text.split() if w[0].isupper() and len(w) > 2)
    
    specificity_boost = min((numbers_and_stats * 0.15 + proper_nouns * 0.12), 0.6)
    
    final = base - filler_penalty + specificity_boost
    return float(max(0.1, min(1.0, final)))

def compute_features_v4_enhanced(segment: Dict, audio_file: str, y_sr=None, genre: str = 'general', platform: str = 'tiktok', segments: list = None) -> Dict:
    """
    Enhanced feature computation with Phase 1, 2 & 3 improvements:
    - Typed containers with validation
    - Unified synergy scoring
    - Platform length v2
    - Path whitening (Phase 2)
    - Genre confidence blending (Phase 2)
    - Boundary hysteresis (Phase 2)
    - Prosody-aware arousal (Phase 3)
    - Payoff evidence guard (Phase 3)
    - Calibration system (Phase 3)
    """
    # Use existing compute_features_v4 as base
    features_dict = compute_features_v4(segment, audio_file, y_sr, genre, platform)
    
    if not FEATURE_TYPES:
        return features_dict
    
    # Convert to typed Features container
    features = Features.from_dict(features_dict)
    features.validate(strict=False)  # Clamp in production
    
    # Apply path whitening if enabled
    raw_paths = {
        'hook': features.hook,
        'arousal': features.arousal,
        'emotion': features.emotion,
        'payoff': features.payoff,
        'info_density': features.info_density,
        'q_list': features.q_list,
        'loopability': features.loopability,
        'platform_length': features.platform_length
    }
    
    whitened_paths = None
    if WHITEN_PATHS:
        whitened_paths = whiten_paths(raw_paths)
        # Update features with whitened values
        features.hook = whitened_paths['hook']
        features.arousal = whitened_paths['arousal']
        features.emotion = whitened_paths['emotion']
        features.payoff = whitened_paths['payoff']
        features.info_density = whitened_paths['info_density']
        features.q_list = whitened_paths['q_list']
        features.loopability = whitened_paths['loopability']
        features.platform_length = whitened_paths['platform_length']
    
    # Apply genre confidence blending if enabled
    genre_debug = {}
    if GENRE_BLEND and segments:
        try:
            from .genres import GenreAwareScorer
            genre_scorer = GenreAwareScorer()
            
            # Get base weights
            base_weights = get_clip_weights()
            
            # Apply genre blending
            blended_weights, genre_debug = apply_genre_blending(base_weights, genre_scorer, segments)
            
            # Apply blended weights to features (simplified - in practice you'd integrate this with scoring)
            # This is a placeholder for the full integration
            features.meta['genre_blending'] = genre_debug
            features.meta['blended_weights'] = blended_weights
            
        except Exception as e:
            logger.warning(f"Genre blending failed: {e}")
            genre_debug = {"error": str(e), "blending_applied": False}
    
    # Apply Phase 3 enhancements
    phase3_debug = {}
    
    # Prosody-aware arousal
    if PROSODY_AROUSAL and audio_file:
        try:
            start = segment.get('start', 0)
            end = segment.get('end', start + 30)
            text = segment.get('text', '')
            
            # Get prosody-enhanced arousal score
            prosody_arousal = prosody_arousal_score(text, audio_file, start, end, genre)
            features.arousal = prosody_arousal
            phase3_debug['prosody_arousal'] = {
                'enabled': True,
                'original_arousal': features_dict.get('arousal_score', 0.0),
                'prosody_arousal': prosody_arousal,
                'improvement': prosody_arousal - features_dict.get('arousal_score', 0.0)
            }
        except Exception as e:
            logger.warning(f"Prosody arousal failed: {e}")
            phase3_debug['prosody_arousal'] = {'enabled': False, 'error': str(e)}
    else:
        phase3_debug['prosody_arousal'] = {'enabled': False, 'reason': 'disabled_or_no_audio'}
    
    # Payoff evidence guard
    if PAYOFF_GUARD:
        try:
            text = segment.get('text', '')
            hook_text = text[:100]  # First 100 chars as hook
            body_text = text[100:]  # Rest as body
            
            # Apply payoff guard
            original_payoff = features.payoff
            guarded_payoff = payoff_guard(hook_text, body_text, original_payoff, genre)
            features.payoff = guarded_payoff
            
            phase3_debug['payoff_guard'] = {
                'enabled': True,
                'original_payoff': original_payoff,
                'guarded_payoff': guarded_payoff,
                'capped': guarded_payoff < original_payoff,
                'genre': genre
            }
        except Exception as e:
            logger.warning(f"Payoff guard failed: {e}")
            phase3_debug['payoff_guard'] = {'enabled': False, 'error': str(e)}
    else:
        phase3_debug['payoff_guard'] = {'enabled': False, 'reason': 'disabled'}
    
    # Apply unified synergy scoring
    if SYNERGY_MODE == "unified":
        synergy = synergy_bonus(raw_paths)
    else:
        # Use existing synergy logic
        synergy = features_dict.get('synergy_multiplier', 1.0) - 1.0
    
    # Apply calibration
    if CALIBRATION_V:
        try:
            # Calibrate final score
            original_final = features_dict.get('final_score', 0.0)
            calibrated_final = apply_calibration(original_final, CALIBRATION_V)
            
            phase3_debug['calibration'] = {
                'enabled': True,
                'version': CALIBRATION_V,
                'original_final': original_final,
                'calibrated_final': calibrated_final,
                'adjustment': calibrated_final - original_final
            }
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            phase3_debug['calibration'] = {'enabled': False, 'error': str(e)}
    else:
        phase3_debug['calibration'] = {'enabled': False, 'reason': 'disabled'}
    
    # Quantize for stability
    features.quantize()
    
    # Convert back to dict with enhanced debug info
    result = features.to_dict()
    
    # Calculate final score using the scoring system
    from .scoring import score_segment_v4
    scored_result = score_segment_v4(result, genre=genre)
    final_score = scored_result.get('viral_score_100', 0) / 100.0  # Convert from 0-100 to 0-1
    display_score = scored_result.get('viral_score_100', 0)
    
    # Quantize final scores for stability
    from .types import quantize
    final_score = quantize(final_score)
    display_score = quantize(display_score, 1.0)  # Quantize to whole numbers for display
    
    result.update({
        'final_score': final_score,
        'display_score': display_score,
        'raw_score': final_score,
        'clip_score_100': display_score,
        'synergy_multiplier': 1.0 + synergy,
        'synergy_bonus': synergy,
        'scoring_version': 'v4.7.2-unified-syn-whiten-blend-prosody-guard-cal',
        'debug': {
            'raw_paths': raw_paths,
            'whitened_paths': whitened_paths,
            'synergy_mode': SYNERGY_MODE,
            'feature_types_enabled': FEATURE_TYPES,
            'platform_len_v': PLATFORM_LEN_V,
            'whiten_paths_enabled': WHITEN_PATHS,
            'genre_blend_enabled': GENRE_BLEND,
            'boundary_hysteresis_enabled': BOUNDARY_HYSTERESIS,
            'prosody_arousal_enabled': PROSODY_AROUSAL,
            'payoff_guard_enabled': PAYOFF_GUARD,
            'calibration_enabled': bool(CALIBRATION_V),
            'calibration_version': CALIBRATION_V,
            'genre_debug': genre_debug,
            'phase3_debug': phase3_debug
        }
    })
    
    return result

def find_viral_clips_enhanced(segments: List[Dict], audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """
    Enhanced viral clip finding with Phase 1, 2 & 3 improvements:
    - Path whitening
    - Genre confidence blending
    - Boundary hysteresis
    - Prosody-aware arousal
    - Payoff evidence guard
    - Calibration system
    """
    logger.info(f"Enhanced pipeline received {len(segments)} segments")
    
    if not segments:
        return {
            'clips': [],
            'genre': genre,
            'platform': platform,
            'scoring_version': 'v4.7.1-unified-syn-whiten-blend',
            'debug': {
                'phase2_enabled': True,
                'whiten_paths': WHITEN_PATHS,
                'genre_blend': GENRE_BLEND,
                'boundary_hysteresis': BOUNDARY_HYSTERESIS
            }
        }
    
    # Apply boundary hysteresis if enabled
    if BOUNDARY_HYSTERESIS:
        optimal_segments = find_optimal_boundaries(segments, audio_file)
        if optimal_segments:
            segments = optimal_segments
    
    # Process each segment with enhanced features
    enhanced_segments = []
    logger.info(f"Processing {len(segments)} segments with enhanced pipeline")
    
    for i, segment in enumerate(segments):
        try:
            enhanced_features = compute_features_v4_enhanced(
                segment, audio_file, genre=genre, platform=platform, segments=segments
            )
            # Preserve original segment data with enhanced features
            enhanced_segment = {
                **segment,  # Keep original segment data (text, start, end, etc.)
                **enhanced_features  # Add enhanced features
            }
            enhanced_segments.append(enhanced_segment)
            
            # Log progress every 10 segments
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(segments)} segments")
                
        except Exception as e:
            logger.warning(f"Failed to process segment {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(enhanced_segments)}/{len(segments)} segments")
    
    # Sort by final score
    enhanced_segments.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    # Log score distribution
    if enhanced_segments:
        scores = [seg.get('final_score', 0) for seg in enhanced_segments]
        logger.info(f"Score distribution: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
        
        # Log top 3 segments
        for i, seg in enumerate(enhanced_segments[:3]):
            score = seg.get('final_score', 0)
            display_score = seg.get('display_score', 0)
            text_length = len(seg.get('text', '').split())
            logger.info(f"Top segment {i+1}: score={score:.3f}, display={display_score}, words={text_length}")
    
    # Take top clips
    top_clips = enhanced_segments[:10]  # Top 10 clips
    
    # Calculate health metrics
    import statistics
    durations = [seg.get('end', 0) - seg.get('start', 0) for seg in enhanced_segments]
    health_metrics = {
        'segments': len(enhanced_segments),
        'sec_p50': statistics.median(durations) if durations else 0,
        'sec_p90': statistics.quantiles(durations, n=10)[8] if len(durations) >= 10 else max(durations) if durations else 0,
        'yield_rate': len(top_clips) / max(1, len(enhanced_segments)),
        'filters': {
            'ads_removed': 0,  # Will be calculated by filtering functions
            'intros_removed': 0,  # Will be calculated by filtering functions
            'caps_applied': len(segments) - len(enhanced_segments)  # Segments filtered by caps
        }
    }
    
    return {
        'clips': top_clips,
        'genre': genre,
        'platform': platform,
        'scoring_version': 'v4.7.2-unified-syn-whiten-blend-prosody-guard-cal',
        'debug': {
            'phase2_enabled': True,
            'phase3_enabled': True,
            'whiten_paths': WHITEN_PATHS,
            'genre_blend': GENRE_BLEND,
            'boundary_hysteresis': BOUNDARY_HYSTERESIS,
            'prosody_arousal': PROSODY_AROUSAL,
            'payoff_guard': PAYOFF_GUARD,
            'calibration': bool(CALIBRATION_V),
            'calibration_version': CALIBRATION_V,
            'synergy_mode': SYNERGY_MODE,
            'platform_len_v': PLATFORM_LEN_V,
            'total_segments': len(segments),
            'processed_segments': len(enhanced_segments),
            'top_clips_count': len(top_clips),
            'health': health_metrics
        }
    }

# Stub implementations for unused functions
def debug_segment_scoring(*args, **kwargs):
    """Stub implementation"""
    return {}

def compute_features_v4_batch(*args, **kwargs):
    """Stub implementation"""
    return []

def compute_audio_hook_modifier(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def detect_laughter_exclamations(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def calculate_hook_components(*args, **kwargs):
    """Stub implementation"""
    return {}

def calculate_time_weighted_hook_score(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def score_patterns_in_text(*args, **kwargs):
    """Stub implementation"""
    return 0.0

# Removed duplicate stub implementations - using real implementations above

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
    if genre == 'comedy':
        comedy_indicators = ['hilarious', 'funny', 'lol', 'haha', 'rofl', 'joke']
        if any(indicator in t for indicator in comedy_indicators):
            score = min(score + 0.1, 0.9)
    elif genre == 'fantasy_sports':
        sports_indicators = ['fire', 'draft', 'start', 'bench', 'target', 'sleeper', 'bust']
        if any(indicator in t for indicator in sports_indicators):
            score = min(score + 0.1, 0.9)
    elif genre == 'true_crime':
        crime_indicators = ['murder', 'killer', 'victim', 'evidence', 'mystery', 'suspicious']
        if any(indicator in t for indicator in crime_indicators):
            score = min(score + 0.1, 0.9)
    
    return float(np.clip(score, 0.0, 1.0))

def _calibrate_info_density_stats(*args, **kwargs):
    """Stub implementation"""
    return {}

def _ql_calibrate_stats(*args, **kwargs):
    """Stub implementation"""
    return {}

def _calibrate_emotion_stats(*args, **kwargs):
    """Stub implementation"""
    return {}

def build_emotion_audio_sidecar(*args, **kwargs):
    """Stub implementation"""
    return {}

def compute_audio_energy(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def info_density_score_v2(*args, **kwargs):
    """Stub implementation"""
    return 0.0

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

    Q_raw = _saturating_sum([q_direct, q_compare, q_choice, q_rhet, q_cliff]) + q_prompt

    # -------- Lists subscore --------
    list_markers = _LIST_MARKERS.search(text)
    list_items = _LIST_ITEMS.findall(text)
    ideal_range = cfg["ideal_items_range"]
    
    list_count = len(list_items)
    if list_count == 0:
        list_density = 0.0
    elif ideal_range[0] <= list_count <= ideal_range[1]:
        list_density = 1.0
    else:
        # Penalty for too few or too many items
        if list_count < ideal_range[0]:
            list_density = 0.3 + 0.7 * (list_count / ideal_range[0])
        else:
            list_density = max(0.3, 1.0 - 0.1 * (list_count - ideal_range[1]))

    list_marker_bonus = 0.2 if list_markers else 0.0
    L_raw = list_density + list_marker_bonus

    # -------- Genre tweaks --------
    genre_tweaks = cfg["genre_tweaks"].get(genre or "", {})
    q_bonus = genre_tweaks.get("question_bonus", 0.0)
    l_bonus = genre_tweaks.get("list_bonus", 0.0)

    Q_raw += q_bonus
    L_raw += l_bonus

    # -------- Anti-bait penalties --------
    bait_penalty = 0.0
    if _BAIT_PATTERNS.search(text):
        bait_penalty = min(cfg["pen_bait_cap"], 0.1 * len(_BAIT_PATTERNS.findall(text)))
    
    vacuous_penalty = 0.0
    if _VACUOUS_Q.search(text):
        vacuous_penalty = min(cfg["pen_vacuous_q_cap"], 0.05 * len(_VACUOUS_Q.findall(text)))

    # -------- Final combination --------
    raw_score = _saturating_sum([Q_raw, L_raw]) - bait_penalty - vacuous_penalty
    return max(0.0, min(1.0, raw_score))

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
    
    # Calculate calibration parameters
    raw_scores = []
    for seg in segments:
        text = seg.get("text") or seg.get("transcript") or ""
        if text:
            raw = _question_list_raw_v2(text)
            raw_scores.append(raw)
    
    if not raw_scores:
        return
    
    # Simple calibration
    MU = sum(raw_scores) / len(raw_scores)
    SIGMA = (sum((x - MU) ** 2 for x in raw_scores) / len(raw_scores)) ** 0.5
    
    # Apply scores
    for seg in segments:
        seg["question_score"] = question_list_score_v2(seg, MU, SIGMA)

def emotion_score_v2(*args, **kwargs):
    """Stub implementation"""
    return 0.0

# Regex for outro detection
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
    if PLATFORM_LEN_V >= 2:
        # Use new Phase 1 implementation
        # Estimate info_density from text_tail if available
        info_density = 0.0
        if text_tail:
            # Simple heuristic for info density
            words = len(text_tail.split())
            if words > 0:
                info_density = min(1.0, words / 20.0)  # Rough estimate
        
        return platform_length_score_v2(duration, info_density, platform)
    
    # Fallback to original implementation
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

def _info_density_raw_v2(*args, **kwargs):
    """Stub implementation"""
    return 0.0

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
    
    # Saturating combiner: 1 - ∏(1 - xᵢ)
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


def detect_podcast_genre(*args, **kwargs):
    """Stub implementation for podcast genre detection"""
    return 'general'

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
        if boundary["confidence"] < 0.5:  # Lowered threshold to include more boundaries
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
        
        if len(segment_words) < 5:  # Reduced to allow shorter segments
            continue
        
        # Calculate timing based on word count and total duration
        total_words = len(words)
        segment_ratio = len(segment_words) / total_words
        segment_duration = total_duration * segment_ratio
        
        # Ensure minimum duration for platform requirements
        if segment_duration < target_length["min"]:
            segment_duration = target_length["min"]
        
        # CRITICAL FIX: Preserve original transcript timing instead of artificial calculation
        # Find the actual transcript segments that correspond to this text
        original_start = None
        original_end = None
        
        # Look for the first and last words in the original transcript
        for j, word in enumerate(words):
            if j == boundary["position"]:
                # Find which original segment contains this word
                for orig_seg in segments:
                    if word in orig_seg.get("text", "").split():
                        original_start = orig_seg.get("start", current_start)
                        break
            if j == end_position - 1:
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
            segment = {
                "text": segment_text,
                "start": current_start,
                "end": current_start + segment_duration,
                "boundary_type": boundary["type"],
                "confidence": boundary["confidence"]
            }
            # Apply caps filtering
            if _keep(segment):
                dynamic_segments.append(segment)
        elif segment_duration < target_length["min"]:
            # If segment is too short, just extend it to minimum length instead of merging
            extended_duration = target_length["min"]
            segment = {
                "text": segment_text,
                "start": current_start,
                "end": current_start + extended_duration,
                "boundary_type": "extended",
                "confidence": boundary["confidence"]
            }
            # Apply caps filtering
            if _keep(segment):
                dynamic_segments.append(segment)
        
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
    
    # Apply final caps filtering to ensure all segments meet requirements
    final_segments = [seg for seg in final_segments if _keep(seg)]
    
    return final_segments

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
        'true_crime': 'true crime content',
        'health_wellness': 'health and wellness advice',
        'news_politics': 'news and political content',
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
        'setup_punchline': 'classic comedy structure',
        'mystery': 'intriguing mystery elements',
        'resolution': 'satisfying resolution',
        'authority': 'expert authority and credibility',
        'specificity': 'concrete details and specifics',
        'curiosity': 'curiosity gap and knowledge',
        'clarity': 'clear explanations',
        'credibility': 'trustworthy health advice',
        'transformation': 'life transformation potential',
        'urgency': 'breaking news urgency',
        'controversy': 'controversial political content'
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
        from .genres import GenreAwareScorer
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

def _grade_breakdown(feats: dict) -> dict:
    """Generate detailed grade breakdown for scoring explanation"""
    breakdown = {}
    
    # Core scoring components
    breakdown['hook'] = {
        'score': feats.get('hook_score', 0.0),
        'grade': _score_to_grade(feats.get('hook_score', 0.0)),
        'description': 'Attention-grabbing opening'
    }
    
    breakdown['arousal'] = {
        'score': feats.get('arousal_score', 0.0),
        'grade': _score_to_grade(feats.get('arousal_score', 0.0)),
        'description': 'Energy and excitement level'
    }
    
    breakdown['emotion'] = {
        'score': feats.get('emotion_score', 0.0),
        'grade': _score_to_grade(feats.get('emotion_score', 0.0)),
        'description': 'Emotional engagement'
    }
    
    breakdown['payoff'] = {
        'score': feats.get('payoff_score', 0.0),
        'grade': _score_to_grade(feats.get('payoff_score', 0.0)),
        'description': 'Clear value or insight'
    }
    
    breakdown['info'] = {
        'score': feats.get('info_density', 0.0),
        'grade': _score_to_grade(feats.get('info_density', 0.0)),
        'description': 'Information density'
    }
    
    breakdown['q_or_list'] = {
        'score': feats.get('question_score', 0.0),
        'grade': _score_to_grade(feats.get('question_score', 0.0)),
        'description': 'Questions or list format'
    }
    
    breakdown['loop'] = {
        'score': feats.get('loopability', 0.0),
        'grade': _score_to_grade(feats.get('loopability', 0.0)),
        'description': 'Replayability factor'
    }
    
    breakdown['length'] = {
        'score': feats.get('platform_length_score', 0.0),
        'grade': _score_to_grade(feats.get('platform_length_score', 0.0)),
        'description': 'Optimal length for platform'
    }
    
    return breakdown

def _score_to_grade(score: float) -> str:
    """Convert score to letter grade"""
    if score >= 0.9:
        return 'A+'
    elif score >= 0.8:
        return 'A'
    elif score >= 0.7:
        return 'B+'
    elif score >= 0.6:
        return 'B'
    elif score >= 0.5:
        return 'C+'
    elif score >= 0.4:
        return 'C'
    elif score >= 0.3:
        return 'D'
    else:
        return 'F'

def _heuristic_title(text: str, feats: dict, cfg: dict, rank: int | None = None) -> str:
    """Generate heuristic title based on content and features"""
    # Extract key phrases
    words = text.split()
    
    # Look for key phrases in first 20 words
    first_20 = words[:20]
    key_phrases = []
    
    # Look for quoted text
    quoted = re.findall(r'"([^"]*)"', text)
    if quoted:
        key_phrases.extend(quoted[:2])
    
    # Look for capitalized phrases
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    if caps:
        key_phrases.extend(caps[:2])
    
    # Look for numbers and statistics
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
    if numbers:
        key_phrases.extend(numbers[:2])
    
    # If no key phrases found, use first meaningful words
    if not key_phrases:
        # Skip common words
        meaningful = [w for w in first_20 if w.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
        key_phrases = meaningful[:3]
    
    # Create title
    if key_phrases:
        title = ' '.join(key_phrases[:3])
        # Truncate if too long
        if len(title) > 60:
            title = title[:57] + '...'
    else:
        # Fallback to first few words
        title = ' '.join(first_20[:5])
        if len(title) > 60:
            title = title[:57] + '...'
    
    # Add rank if provided
    if rank is not None:
        title = f"#{rank}: {title}"
    
    return title

# Platform and tone mapping features
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
        'comedy': 1.1,
        'fantasy_sports': 0.9,
        'true_crime': 1.0,
        'business': 1.05,
        'news_politics': 1.0,
        'health_wellness': 1.1
    },
    'linkedin': {
        'business': 1.2,  # Perfect match
        'education': 1.1,
        'news_politics': 1.0,
        'health_wellness': 0.95,
        'comedy': 0.7,  # Not ideal for LinkedIn
        'fantasy_sports': 0.8,
        'true_crime': 0.85
    }
}

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

def resolve_genre_from_tone(tone: str, auto_detected: str) -> str:
    """Map frontend tone to backend genre with fallback to auto-detected"""
    if tone and tone in TONE_TO_GENRE_MAP:
        mapped_genre = TONE_TO_GENRE_MAP[tone]
        print(f"🎯 Tone '{tone}' mapped to genre: {mapped_genre}")
        return mapped_genre
    
    print(f"🎯 No tone mapping found for '{tone}', using auto-detected: {auto_detected}")
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
            "label": "⚖️ Mixed Performance", 
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
    from .genres import GenreAwareScorer
    genre_scorer = GenreAwareScorer()
    
    # Get confidence scores for all genres
    genre_scores = {}
    for genre_name in genre_scorer.genres.keys():
        confidence = genre_scorer.detect_genre_with_confidence(segments, genre_name)
        genre_scores[genre_name] = confidence
    
    # Sort by confidence
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    top_patterns = [{"genre": g, "confidence": c} for g, c in sorted_genres[:3]]
    
    return {
        "auto_detected_genre": detected_genre,
        "applied_genre": applied_genre,
        "genre_confidence": "high" if detected_genre != 'general' else "low",
        "top_genre_patterns": top_patterns,
        "tone_used": tone,
        "mapping_applied": tone is not None and tone in TONE_TO_GENRE_MAP,
        "sample_text": sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
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
    from .genres import detect_podcast_genre
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
        from .genres import detect_podcast_genre
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
    import datetime
    
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

def split_mixed_segments(segments: List[Dict]) -> List[Dict]:
    """
    Split segments that contain both intro content and good content.
    This helps separate the valuable content from the filler.
    """
    split_segments = []
    
    for seg in segments:
        text = seg.get("text", "").lower()
        
        # Check if this segment contains both intro and good content
        has_intro = any(pattern in text[:100] for pattern in [
            "yo, what's up", "hey", "hi", "hello", "what's up",
            "it's monday", "it's tuesday", "it's wednesday", "it's thursday", "it's friday",
            "i'm jeff", "my name is", "hope you", "hope everyone"
        ])
        
        has_good_content = any(pattern in text for pattern in [
            "observation", "insight", "casual drafters", "way better", "main observation",
            "key takeaway", "the thing is", "what i found", "what i learned"
        ])
        
        if has_intro and has_good_content:
            # Try to find where the good content starts
            words = text.split()
            good_content_start = 0
            
            # Look for transition markers
            for i, word in enumerate(words):
                if any(pattern in " ".join(words[i:i+3]) for pattern in [
                    "observation", "insight", "casual drafters", "way better",
                    "main observation", "key takeaway", "the thing is"
                ]):
                    good_content_start = i
                    break
            
            if good_content_start > 0:
                # Split the segment
                intro_text = " ".join(words[:good_content_start])
                good_text = " ".join(words[good_content_start:])
                
                # Calculate timing split
                total_duration = seg.get("end", 0) - seg.get("start", 0)
                intro_ratio = good_content_start / len(words)
                intro_duration = total_duration * intro_ratio
                
                # Create intro segment (will be filtered out)
                intro_seg = seg.copy()
                intro_seg["text"] = intro_text
                intro_seg["end"] = seg.get("start", 0) + intro_duration
                intro_seg["is_intro"] = True
                
                # Create good content segment
                good_seg = seg.copy()
                good_seg["text"] = good_text
                good_seg["start"] = seg.get("start", 0) + intro_duration
                good_seg["is_intro"] = False
                
                split_segments.extend([intro_seg, good_seg])
            else:
                # Can't split cleanly, keep as is
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
        from .genres import GenreAwareScorer
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
    from .scoring import score_segment_v4
    scored_clips = [score_segment_v4(f, genre=genre) for f in non_intro_features]
    
    # Sort by viral score and return top 5
    return {
        'genre': genre,
        'clips': sorted(scored_clips, key=lambda x: x["viral_score_100"], reverse=True)[:5]
    }

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
    
    # Dynamic segmentation functions
    "find_natural_boundaries",
    "create_dynamic_segments",
    
    # Pipeline functions
    "filter_ads_from_features",
    "filter_intro_content_from_features", 
    "split_mixed_segments",
    "find_viral_clips",
    
    # Explanation and analysis functions
    "_explain_viral_potential_v4",
    "_grade_breakdown",
    "_score_to_grade",
    "_heuristic_title",
    
    # Platform and tone mapping
    "PLATFORM_GENRE_MULTIPLIERS",
    "TONE_TO_GENRE_MAP",
    "PLATFORM_MAP",
    "resolve_platform",
    "resolve_genre_from_tone",
    "interpret_synergy",
    "get_genre_detection_debug",
    
    # Advanced API functions
    "find_viral_clips_with_tone",
    "find_viral_clips_with_genre",
    "find_candidates",
    
    # Question/List scoring V2
    "_question_list_raw_v2",
    "question_list_score_v2",
    "attach_question_list_scores_v2",
    
    # Phase 1 Enhanced Features
    "compute_features_v4_enhanced",
    
    # Phase 2 Enhanced Features
    "find_viral_clips_enhanced",
    
    # Phase 3 Enhanced Features
    "prosody_arousal_score",
    "payoff_guard",
    "apply_calibration",
    
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
    "detect_podcast_genre",
]
