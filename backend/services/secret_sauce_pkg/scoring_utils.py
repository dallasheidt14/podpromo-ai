"""
Advanced scoring utilities for Phase 1 improvements.
"""

import numpy as np
import re
import os
from typing import Dict, List, Tuple
from .types import _c01, quantize
# blend_weights will be defined locally to avoid circular import

def _best_audio_for_features(audio_path: str) -> str:
    """Prefer clean WAV if available to eliminate mpg123 errors"""
    base, ext = os.path.splitext(audio_path)
    wav_candidate = f"{base}.__asr16k.wav"
    if os.path.exists(wav_candidate):
        return wav_candidate
    else:
        return audio_path

def blend_weights(
    genre_conf: Dict[str, float],
    weights_by_genre: Dict[str, Dict[str, float]],
    *,
    top_k: int = 2,
    min_conf: float = 0.20,
    default_genre: str = "general",
) -> Dict[str, float]:
    """
    Blend per-genre weight vectors by confidence. Falls back to `default_genre`.
    - genre_conf: {"business": 0.55, "comedy": 0.25, ...}
    - weights_by_genre: {"business": {...}, "comedy": {...}, "general": {...}}
    Returns a normalized weight dict over paths.
    """
    if not genre_conf:
        return dict(weights_by_genre.get(default_genre, {}))

    # take top_k genres above threshold
    items = [(g, c) for g, c in genre_conf.items() if c >= min_conf and g in weights_by_genre]
    if not items:
        return dict(weights_by_genre.get(default_genre, {}))

    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_k]

    total_conf = sum(c for _, c in items)
    if total_conf <= 0:
        return dict(weights_by_genre.get(default_genre, {}))

    # normalized confidences as blend coefficients
    coeffs = {g: (c / total_conf) for g, c in items}

    # union of all paths present in the selected genres
    all_keys = set().union(*[weights_by_genre[g].keys() for g, _ in items])
    blended = {k: 0.0 for k in all_keys}

    for g, a in coeffs.items():
        for k, w in weights_by_genre[g].items():
            blended[k] += a * w

    # renormalize to sum~=1 for safety
    s = sum(blended.values())
    if s > 0:
        blended = {k: v / s for k, v in blended.items()}
    return blended

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
    
    # Blend weights using genre confidence
    genre_conf = {
        primary_genre: primary_conf,
        debug_info["secondary_genre"]: debug_info["secondary_confidence"]
    }
    weights_by_genre = {
        primary_genre: primary_profile.weights,
        debug_info["secondary_genre"]: secondary_profile.weights
    }
    
    try:
        blended_weights = blend_weights(genre_conf, weights_by_genre, top_k=2, min_conf=0.20)
    except Exception as e:
        logger.warning(f"Genre blending failed: {e}; falling back to primary genre weights")
        blended_weights = primary_profile.weights.copy()
    
    debug_info["primary_weights"] = primary_profile.weights
    debug_info["secondary_weights"] = secondary_profile.weights
    debug_info["final_weights"] = blended_weights
    
    return blended_weights, debug_info

def grow_to_bins(seg: dict, audio_file: str, targets: list = [12, 18, 24, 30], max_jitter: float = 1.0, max_score_drop: float = 0.02) -> dict:
    """
    Grow segment to target platform bins by extending to natural cuts.
    Returns the best variant by platform_length_score_v2.
    """
    variants = [seg]
    current_dur = seg.get("end", 0) - seg.get("start", 0)
    
    for target in targets:
        if current_dur >= target - 0.5:  # already long enough
            continue
            
        # Try to extend to target duration
        extended = try_extend_to(seg, audio_file, target, max_jitter)
        if extended and (seg.get("final_score", 0) - extended.get("final_score", 0)) <= max_score_drop:
            variants.append(extended)
    
    # Return best variant by platform_length_score_v2
    return max(variants, key=lambda x: x.get("platform_length_score_v2", 0))

def try_extend_to(seg: dict, audio_file: str, target_duration: float, max_jitter: float = 1.0) -> dict:
    """
    Try to extend segment to target duration by finding natural cuts.
    """
    start = seg.get("start", 0)
    end = seg.get("end", 0)
    current_dur = end - start
    target_end = start + target_duration
    
    # Look for natural cuts within jitter range
    snap_points = find_natural_cuts(audio_file, end, target_end, max_jitter)
    
    if not snap_points:
        return None
        
    # Try each snap point
    best_variant = None
    best_score = -1
    
    for snap_end in snap_points:
        variant = seg.copy()
        variant["end"] = snap_end
        variant["text"] = get_text_for_segment(audio_file, start, snap_end)
        
        # Re-score the variant
        variant_score = score_variant(variant)
        if variant_score > best_score:
            best_score = variant_score
            best_variant = variant
    
    return best_variant

def find_natural_cuts(audio_file: str, start_time: float, end_time: float, max_jitter: float) -> list:
    """
    Find natural cut points (punctuation, silence) within the time range.
    """
    # This is a simplified version - in practice you'd use audio analysis
    # For now, return evenly spaced points as placeholders
    cuts = []
    step = 0.5  # 0.5 second steps
    current = start_time + step
    while current <= end_time:
        cuts.append(current)
        current += step
    return cuts

def get_text_for_segment(audio_file: str, start: float, end: float) -> str:
    """
    Extract text for the given time segment.
    """
    # This would integrate with your transcription service
    # For now, return placeholder
    return f"Extended segment {start:.1f}-{end:.1f}"

def score_variant(variant: dict) -> float:
    """
    Score a variant segment.
    """
    # This would call your scoring function
    return variant.get("final_score", 0)

def find_optimal_boundaries(segments: list, audio_file: str, min_delta: float = 0.03) -> list:
    """
    Find optimal boundaries with hysteresis to reduce jitter.
    Modified to be less aggressive and preserve more segments.
    """
    if not segments:
        return []
    
    # For now, just return all segments with minor boundary adjustments
    # This prevents the aggressive filtering that was causing 0 clips
    optimal_segments = []
    
    for i, segment in enumerate(segments):
        # Apply minor boundary adjustments if needed
        adjusted_segment = segment.copy()
        
        # Snap to nearest pause or punctuation if available
        adjusted_segment = snap_to_nearest_pause_or_punct(adjusted_segment)
        
        optimal_segments.append(adjusted_segment)
    
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
        
        # Load audio segment - prefer clean WAV if available
        audio_for_features = _best_audio_for_features(audio_file)
        y, sr = librosa.load(audio_for_features, sr=sr, offset=start, duration=end-start)
        
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
    elif calibration_version == "2025.09.2":
        return piecewise_calibrate_v2(score)
    elif calibration_version == "2025.09.3":
        return calibrate_mild(score)
    else:
        return score

def looks_like_question(t: str) -> bool:
    """Robust question detector that handles clipped punctuation and ASR cases"""
    if not t: 
        return False
    t = t.strip().lower()
    if t.endswith("?"): 
        return True
    # Handle clipped punctuation / ASR cases
    return bool(re.match(
        r"^(what|why|how|when|where|who|whom|whose|which|do|does|did|can|could|should|would|is|are|was|were|will|won't|isn't|aren't|didn't)\b",
        t
    ))

def apply_anti_bait_cap(features: dict) -> float:
    """
    Apply anti-bait cap to prevent micro-clips from hitting max scores.
    HARD CAPS must be applied AFTER all boosts and BEFORE calibration.
    """
    final_score = features.get("final_score", 0.0)
    hook_score = features.get("hook_score", 0.0)
    payoff_score = features.get("payoff_score", 0.0)
    words = features.get("words", 0)
    ad_penalty = features.get("ad_penalty", 0.0)
    text = features.get("text", "").strip()
    
    # Track applied caps for debugging
    applied = []
    
    # Check for ad-like content
    looks_ad = ad_penalty >= 0.3 or any(phrase in text.lower() for phrase in [
        "chase sapphire", "amex platinum", "credit card", "points per dollar"
    ])
    
    # HARD CAPS (must be AFTER boosts, BEFORE calibration)
    is_q = looks_like_question(text)
    is_micro = words < 12
    
    # Micro anti-bait cap
    if is_micro and payoff_score < 0.20 and (is_q or looks_ad):
        final_score = min(final_score, 0.55)
        applied.append("anti_bait_micro")
    
    # Long super-hook but no payoff
    if words >= 18 and hook_score >= 0.90 and payoff_score < 0.12:
        final_score = min(final_score, 0.72)
        applied.append("no_payoff_ceiling")
    
    # Question-only cap (if no answer stitched)
    if is_q and is_micro and payoff_score < 0.20 and not features.get("_has_answer", False):
        final_score = min(final_score, 0.55)
        applied.append("question_cap")
    
    # Record applied caps for debugging
    if applied:
        features.setdefault("flags", {})["caps_applied"] = applied
    
    # Sanity check: if we capped, we should still see it
    if "question_cap" in features.get("flags", {}).get("caps_applied", []):
        assert final_score <= 0.55 + 1e-6, f"Question cap leaked: {final_score}"
    
    # Small payoff evidence bonus (keeps balance)
    if payoff_score >= 0.30:
        final_score += 0.02   # tiny +2%
    
    # Calibration ceiling: never show 100/100
    final_score = min(final_score, 0.98)
    final_score = round(final_score, 2)  # stabilize ordering
    
    return final_score

def try_stitch_answer(seg: dict, next_seg: dict) -> dict:
    """Try to stitch a question with its answer for better context"""
    if not seg.get("text", "").strip().endswith("?"):
        return None
    if next_seg is None: 
        return None
    
    t = (next_seg.get("text") or "").lower()
    # Simple payoff cues
    if (next_seg.get("payoff_score", 0.0) >= 0.25 or
        re.search(r"\b(so|that means|here('s)? (how|why)|the (answer|takeaway) is|because)\b", t)):
        # Merge segments with max 8 second extension
        merged = {
            "text": seg.get("text", "") + " " + next_seg.get("text", ""),
            "start": seg.get("start", 0),
            "end": min(seg.get("end", 0) + 8.0, next_seg.get("end", seg.get("end", 0) + 8.0)),
            "words": seg.get("words", 0) + next_seg.get("words", 0),
            "_has_answer": True
        }
        # Copy other fields from the question segment
        for key, value in seg.items():
            if key not in merged:
                merged[key] = value
        return merged
    return None

def collapse_question_runs(candidates: list, iou_thresh: float = 0.35) -> list:
    """
    Collapse consecutive question segments, keeping only the strongest one.
    Treats adjacency by time, not IoU; keeps best of local question-only run.
    """
    def is_q(c):
        return (c.get("text", "").strip().endswith("?")) and c.get("payoff_score", 0.0) < 0.20
    
    out, run = [], []
    
    # Sort candidates by start time
    for c in sorted(candidates, key=lambda x: x.get("start", 0.0)):
        if is_q(c):
            run.append(c)
        else:
            if run:
                # Keep the best question from the run
                best = max(run, key=lambda r: r.get("final_score", 0.0))
                out.append(best)
                run.clear()
            out.append(c)
    
    # Handle any remaining run
    if run:
        out.append(max(run, key=lambda r: r.get("final_score", 0.0)))
    
    return out

def piecewise_calibrate_v2(score: float) -> float:
    """
    Enhanced piecewise calibration to stretch the 0.35-0.55 band.
    Brightens mids by ~10-15% and compresses extremes slightly.
    """
    if score <= 0.35:
        # Compress low scores slightly
        return score * 0.95
    elif score <= 0.55:
        # Brighten mid-range scores by 12%
        return score * 1.12
    elif score <= 0.75:
        # Slight brightening for upper-mid scores
        return score * 1.05
    else:
        # Compress high scores slightly to prevent ceiling effects
        return min(0.95, score * 0.98)

def calibrate_mild(score: float) -> float:
    """
    Mild calibration to widen mid-range contrasts.
    Brightens 0.35-0.55, compresses extremes slightly.
    """
    if score < 0.20:
        return 0.8 * score
    elif score < 0.50:
        return 0.16 + 0.9 * (score - 0.20)
    elif score < 0.80:
        return 0.43 + 0.8 * (score - 0.50)
    else:
        return 0.67 + 0.6 * (score - 0.80)

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

# Phase 1: Hysteresis + Snap-to Boundary Selection
MIN_DELTA = 0.03
STICKY_BONUS = 0.005

def pick_boundary(current: Dict, candidates: List[Dict]) -> Dict:
    """
    Pick the best boundary using hysteresis to prevent jitter.
    Only moves to a new boundary if score improvement exceeds MIN_DELTA.
    """
    if not candidates:
        return current
    
    # Add sticky bonus to current boundary to prevent unnecessary changes
    best_score = current.get('score', 0) + STICKY_BONUS
    best_boundary = current
    
    for candidate in candidates:
        candidate_score = candidate.get('score', 0)
        if candidate_score > best_score + MIN_DELTA:
            best_score = candidate_score
            best_boundary = candidate
    
    return best_boundary

def snap_to_nearest_pause_or_punct(boundary: Dict, window_ms: float = 250) -> Dict:
    """
    Snap boundary to nearest punctuation or pause within window_ms.
    """
    # This is a simplified implementation - in practice you'd analyze audio
    # for silence/pause detection or look for punctuation in text
    return boundary
