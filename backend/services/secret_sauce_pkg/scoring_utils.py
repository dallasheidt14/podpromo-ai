"""
Advanced scoring utilities for Phase 1 improvements.
"""

import numpy as np
from typing import Dict, Tuple
from .types import _c01, quantize

# Path constants
PATHS = ("hook", "arousal", "emotion", "payoff", "info_density", "q_list", "loopability", "platform_length")

def whiten_paths(path_scores: Dict[str, float]) -> Dict[str, float]:
    """
    De-correlate paths to prevent double-counting.
    Whitens the path vector before applying weights.
    """
    v = np.array([_c01(path_scores.get(p, 0.0)) for p in PATHS], dtype=np.float32)
    
    # Subtract mean to reduce global inflation
    v = v - v.mean()
    
    # Scale by L2 to avoid one path dominating
    denom = np.linalg.norm(v) or 1.0
    v = (v / denom) * 0.5 + 0.5  # Map back to ~[0,1]
    
    return {p: _c01(float(x)) for p, x in zip(PATHS, v)}

def synergy_bonus(path_scores: Dict[str, float]) -> float:
    """
    Unified additive synergy bonus with anti-bait protection.
    Replaces multiplicative damping with bounded additive bonus.
    """
    # Reward co-occurrence of core elements
    core = (path_scores.get("hook", 0.0) + path_scores.get("payoff", 0.0) + path_scores.get("arousal", 0.0)) / 3
    structure = (path_scores.get("info_density", 0.0) + path_scores.get("q_list", 0.0) + path_scores.get("loopability", 0.0)) / 3
    
    bonus = 0.10 * core + 0.05 * structure
    
    # Anti-bait: high hook with low payoff/info_density
    if path_scores.get("hook", 0.0) > 0.8 and (path_scores.get("payoff", 0.0) < 0.3 or path_scores.get("info_density", 0.0) < 0.3):
        bonus -= 0.08
    
    # Anti-bait: high arousal with low payoff
    if path_scores.get("arousal", 0.0) > 0.8 and path_scores.get("payoff", 0.0) < 0.3:
        bonus -= 0.05
    
    return max(-0.10, min(0.15, bonus))

def platform_length_score_v2(seconds: float, info_density: float, platform: str) -> float:
    """
    Platform-aware length scoring with density consideration.
    Rewards fit + density rather than just duration.
    """
    # Platform-specific targets and weight mixes
    PL_TARGET = {
        "tiktok": 22, "instagram": 24, "youtube": 28, "linkedin": 30
    }
    PL_MIX = {
        "tiktok": (0.65, 0.35), "instagram": (0.65, 0.35), 
        "youtube": (0.55, 0.45), "linkedin": (0.55, 0.45)
    }
    
    target = PL_TARGET.get(platform, 24)
    bell_w, dens_w = PL_MIX.get(platform, (0.6, 0.4))
    
    # Bell curve around target, widened by density
    width = max(6.0, 12.0 - 4.0 * info_density)  # Tighter window when dense
    d = abs(seconds - target)
    bell = max(0.0, 1.0 - (d / width) ** 2)
    
    return _c01(bell_w * bell + dens_w * info_density)

def genre_blend_weights(base_weights: Dict[str, float], profile_a: Dict[str, float], 
                       ca: float, profile_b: Dict[str, float], cb: float) -> Dict[str, float]:
    """
    Blend genre weights based on confidence scores.
    Used for top-2 genre blending.
    """
    # Renormalize confidences
    s = max(1e-6, ca + cb)
    wa = ca / s
    wb = cb / s
    
    return {
        k: base_weights[k] * (wa * profile_a.get(k, 1.0) + wb * profile_b.get(k, 1.0))
        for k in base_weights
    }

def piecewise_calibrate(x: float) -> float:
    """
    Piecewise calibration to make scores more interpretable.
    Brightens mid-range, compresses extremes.
    """
    if x < 0.2:
        return x * 0.8
    elif x < 0.5:
        return 0.16 + (x - 0.2) * 0.9
    elif x < 0.8:
        return 0.43 + (x - 0.5) * 0.8
    else:
        return 0.67 + (x - 0.8) * 0.6

def boundary_hysteresis(candidates: list, current: dict, min_delta: float = 0.03) -> dict:
    """
    Apply hysteresis to boundary selection to reduce jitter.
    Prefer existing boundary unless new one is significantly better.
    """
    best = current
    stability_bonus = 0.005  # Small bonus for keeping existing boundary
    
    for c in candidates:
        # Add stability bonus to current boundary
        current_score = current.get('score', 0.0) + stability_bonus
        candidate_score = c.get('score', 0.0)
        
        if candidate_score > current_score + min_delta:
            best = c
    
    return best

def snap_to_boundary(timestamp: float, silence_dips: list, punctuation_marks: list, 
                    window_ms: float = 250.0) -> float:
    """
    Snap boundary to nearest silence dip or punctuation mark within window.
    """
    window_s = window_ms / 1000.0
    all_markers = silence_dips + punctuation_marks
    
    if not all_markers:
        return timestamp
    
    # Find closest marker within window
    closest = min(all_markers, key=lambda x: abs(x - timestamp))
    
    if abs(closest - timestamp) <= window_s:
        return closest
    
    return timestamp

def detect_genre_with_confidence_enhanced(segments: list, genre_scorer) -> tuple[str, float, dict]:
    """
    Enhanced genre detection with top-2 confidence blending.
    Returns (primary_genre, confidence, debug_info)
    """
    if not segments or not segments:
        return "general", 0.0, {"top_genres": [], "blending_applied": False}
    
    # Get confidence scores for all genres
    genre_scores = {}
    for genre_name in genre_scorer.genres.keys():
        # Combine segments into text for genre detection
        combined_text = " ".join([seg.get('text', '') for seg in segments])
        detected_genre, confidence = genre_scorer.detect_genre_with_confidence(combined_text)
        genre_scores[genre_name] = confidence
    
    # Sort by confidence
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_genres:
        return "general", 0.0, {"top_genres": [], "blending_applied": False}
    
    primary_genre, primary_conf = sorted_genres[0]
    
    # Check if we should apply blending
    if len(sorted_genres) < 2 or primary_conf >= 0.85:
        # Hard assign - high confidence or no second option
        return primary_genre, primary_conf, {
            "top_genres": sorted_genres[:3],
            "blending_applied": False,
            "reason": "high_confidence" if primary_conf >= 0.85 else "insufficient_options"
        }
    
    secondary_genre, secondary_conf = sorted_genres[1]
    
    # Only blend if secondary confidence is above threshold
    if secondary_conf < 0.25:
        return primary_genre, primary_conf, {
            "top_genres": sorted_genres[:3],
            "blending_applied": False,
            "reason": "low_secondary_confidence"
        }
    
    # Apply blending
    return primary_genre, primary_conf, {
        "top_genres": sorted_genres[:3],
        "blending_applied": True,
        "secondary_genre": secondary_genre,
        "secondary_confidence": secondary_conf,
        "blended_weights": True
    }

def apply_genre_blending(base_weights: dict, genre_scorer, segments: list) -> tuple[dict, dict]:
    """
    Apply genre confidence blending to weights.
    Returns (blended_weights, debug_info)
    """
    primary_genre, primary_conf, debug_info = detect_genre_with_confidence_enhanced(segments, genre_scorer)
    
    if not debug_info.get("blending_applied", False):
        # Use single genre
        genre_profile = genre_scorer.genres.get(primary_genre)
        if genre_profile:
            blended_weights = {k: base_weights[k] * genre_profile.weights.get(k, 1.0) for k in base_weights}
        else:
            blended_weights = base_weights.copy()
        
        return blended_weights, debug_info
    
    # Apply blending
    primary_profile = genre_scorer.genres.get(primary_genre)
    secondary_profile = genre_scorer.genres.get(debug_info["secondary_genre"])
    
    if not primary_profile or not secondary_profile:
        return base_weights.copy(), debug_info
    
    # Blend weights
    blended_weights = blend_weights(
        base_weights,
        primary_profile.weights,
        primary_conf,
        secondary_profile.weights,
        debug_info["secondary_confidence"]
    )
    
    debug_info["primary_weights"] = primary_profile.weights
    debug_info["secondary_weights"] = secondary_profile.weights
    debug_info["final_weights"] = blended_weights
    
    return blended_weights, debug_info

def find_optimal_boundaries(segments: list, audio_file: str, min_delta: float = 0.03) -> list:
    """
    Find optimal boundaries with hysteresis to reduce jitter.
    """
    if not segments:
        return []
    
    # This is a simplified implementation - in practice, you'd integrate with
    # the existing boundary detection logic
    optimal_segments = []
    current_boundary = None
    
    for i, segment in enumerate(segments):
        # Calculate boundary score for this segment
        boundary_score = calculate_boundary_score(segment, audio_file)
        
        if current_boundary is None:
            # First segment
            current_boundary = {
                'segment': segment,
                'score': boundary_score,
                'index': i
            }
        else:
            # Apply hysteresis
            if boundary_score > current_boundary['score'] + min_delta:
                # New boundary is significantly better
                optimal_segments.append(current_boundary['segment'])
                current_boundary = {
                    'segment': segment,
                    'score': boundary_score,
                    'index': i
                }
            else:
                # Keep current boundary, update if slightly better
                if boundary_score > current_boundary['score']:
                    current_boundary['segment'] = segment
                    current_boundary['score'] = boundary_score
                    current_boundary['index'] = i
    
    # Add the last boundary
    if current_boundary:
        optimal_segments.append(current_boundary['segment'])
    
    return optimal_segments

def calculate_boundary_score(segment: dict, audio_file: str) -> float:
    """
    Calculate boundary quality score for a segment.
    This is a simplified implementation - integrate with existing logic.
    """
    # Placeholder implementation
    # In practice, this would use silence detection, punctuation analysis, etc.
    duration = segment.get('end', 0) - segment.get('start', 0)
    text = segment.get('text', '')
    
    # Simple heuristic based on duration and text quality
    duration_score = min(1.0, duration / 30.0)  # Prefer segments around 30s
    text_score = min(1.0, len(text.split()) / 50.0)  # Prefer substantial text
    
    return (duration_score + text_score) / 2.0

def _arousal_score_text_audio(text_score: float, rms: float, zcr: float, pitch_std: float) -> float:
    """
    Fuse text arousal with normalized audio features for prosody-aware arousal.
    Combines text analysis with RMS, zero-crossing rate, and pitch variance.
    """
    # Normalize audio features to [0,1] via robust scaling
    # These would typically come from audio analysis
    rms_norm = min(1.0, max(0.0, rms))  # RMS energy (0-1)
    zcr_norm = min(1.0, max(0.0, zcr))  # Zero-crossing rate (0-1)
    pitch_norm = min(1.0, max(0.0, pitch_std))  # Pitch standard deviation (0-1)
    
    # Weighted combination: text + audio features
    arousal = 0.5 * text_score + 0.25 * rms_norm + 0.15 * zcr_norm + 0.10 * pitch_norm
    
    return _c01(arousal)

def extract_prosody_features(audio_file: str, start: float, end: float, sr: int = 22050) -> dict:
    """
    Extract prosody features from audio segment.
    Returns normalized features for arousal scoring.
    """
    try:
        import librosa
        import numpy as np
        
        # Load audio segment
        y, sr = librosa.load(audio_file, sr=sr, offset=start, duration=end-start)
        
        if len(y) == 0:
            return {"rms": 0.0, "zcr": 0.0, "pitch_std": 0.0, "error": "empty_audio"}
        
        # RMS energy (loudness)
        rms = np.sqrt(np.mean(y**2))
        rms_norm = min(1.0, rms / 0.1)  # Normalize assuming max RMS ~0.1
        
        # Zero-crossing rate (speech rate/articulation)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        zcr_norm = min(1.0, zcr * 10)  # Scale up for normalization
        
        # Pitch standard deviation (intonation variation)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            pitch_std = np.std(pitch_values)
            pitch_std_norm = min(1.0, pitch_std / 100)  # Normalize assuming max std ~100Hz
        else:
            pitch_std_norm = 0.0
        
        return {
            "rms": rms_norm,
            "zcr": zcr_norm,
            "pitch_std": pitch_std_norm,
            "error": None
        }
        
    except Exception as e:
        return {"rms": 0.0, "zcr": 0.0, "pitch_std": 0.0, "error": str(e)}

def payoff_guard(hook_text: str, body_text: str, payoff_score: float, genre: str = 'general') -> float:
    """
    Prevent promissory hooks from scoring high without delivery.
    Applies genre-specific caps based on evidence strength.
    """
    # Genre-specific caps
    genre_caps = {
        'education': 0.55,  # Stricter for educational content
        'business': 0.60,   # Stricter for business content
        'comedy': 0.70,     # Gentler for comedy (punchline-style payoff)
        'general': 0.65     # Default cap
    }
    
    cap = genre_caps.get(genre, 0.65)
    
    # Check for evidence of delivery
    has_answer = _detect_insight_content_v2(body_text) > 0.5
    
    if not has_answer and payoff_score > cap:
        return cap  # Cap optimistic payoff without evidence
    
    return payoff_score

def _detect_insight_content_v2(text: str) -> float:
    """
    Detect insight content strength (simplified version).
    In practice, this would use the full implementation from features.py
    """
    if not text:
        return 0.0
    
    # Simple heuristic for insight detection
    insight_indicators = [
        'because', 'therefore', 'as a result', 'consequently', 'thus',
        'the key is', 'the secret is', 'here\'s how', 'the reason',
        'explains', 'reveals', 'shows', 'demonstrates', 'proves'
    ]
    
    text_lower = text.lower()
    insight_count = sum(1 for indicator in insight_indicators if indicator in text_lower)
    
    # Normalize by text length
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    
    insight_density = insight_count / (word_count / 50)  # Normalize to ~50 words
    return min(1.0, insight_density)

def piecewise_calibrate(x: float) -> float:
    """
    Piecewise calibration to make scores more interpretable.
    Brightens mid-range, compresses extremes.
    """
    if x < 0.2:
        return x * 0.8
    elif x < 0.5:
        return 0.16 + (x - 0.2) * 0.9
    elif x < 0.8:
        return 0.43 + (x - 0.5) * 0.8
    else:
        return 0.67 + (x - 0.8) * 0.6

def apply_calibration(score: float, calibration_version: str = "2025.09.1") -> float:
    """
    Apply calibration to make scores more interpretable.
    """
    if calibration_version == "2025.09.1":
        return piecewise_calibrate(score)
    else:
        return score  # No calibration for unknown versions

def prosody_arousal_score(text: str, audio_file: str, start: float, end: float, genre: str = 'general') -> float:
    """
    Complete prosody-aware arousal scoring.
    Combines text analysis with audio features.
    """
    # Get text arousal score (existing function)
    from .features import _arousal_score_text
    text_arousal = _arousal_score_text(text, genre)
    
    # Extract audio features
    audio_features = extract_prosody_features(audio_file, start, end)
    
    if audio_features.get("error"):
        # Fallback to text-only if audio analysis fails
        return text_arousal
    
    # Combine text and audio
    return _arousal_score_text_audio(
        text_arousal,
        audio_features["rms"],
        audio_features["zcr"],
        audio_features["pitch_std"]
    )
