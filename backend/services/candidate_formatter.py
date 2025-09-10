from typing import Dict, List
import logging

from config_loader import get_config
from services.secret_sauce_pkg import _grade_breakdown, _heuristic_title

logger = logging.getLogger(__name__)

# helper to read numeric fields safely
def _num(d, key, default=0.0):
    v = d.get(key)
    try:
        return float(v)
    except Exception:
        return default

def _get_text(feats: dict) -> str:
    for k in ("text", "clean_text", "raw_text", "display_text"):
        v = feats.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _word_count_from_text(t: str) -> int:
    return len(t.split()) if t else 0

def _text_length(feats: dict) -> int:
    # prefer numeric words field if produced by features
    if isinstance(feats.get("words"), (int, float)):
        w = int(feats["words"])
        if w > 0:
            return w
    # fallback to counting words in text
    return _word_count_from_text(_get_text(feats))

def _display_text(t: str) -> str:
    if not t: return t
    t = t.strip()
    return t[0].upper() + t[1:] if t[0].islower() else t

def format_candidate(seg_features: dict, *, platform: str, genre: str, full_episode_transcript: str = None) -> dict:
    """
    seg_features is the dict returned by compute_features_v4_enhanced (or v4 fallback).
    Build a stable, backward-compatible candidate payload.
    """
    # existing fields
    text = _get_text(seg_features)
    text_len = _text_length(seg_features)
    
    cand = {
        "start": seg_features.get("start"),
        "end": seg_features.get("end"),
        "text": _display_text(text),  # auto-capitalize for display
        "raw_text": text,  # original transcription text for audio matching
        "full_transcript": full_episode_transcript or text,  # full episode transcript for detail view
        "text_length": text_len,
        "final_score": _num(seg_features, "final_score", 0.0),
        "display_score": int(round(_num(seg_features, "final_score", 0.0) * 100)),
        "platform": platform,
        "genre": genre,
    }

    # always present scoring components (or 0.0 defaults)
    cand["hook_score"]      = _num(seg_features, "hook_score", 0.0)
    cand["arousal_score"]   = _num(seg_features, "arousal_score", 0.0)
    cand["payoff_score"]    = _num(seg_features, "payoff_score", 0.0)
    cand["info_density"]    = _num(seg_features, "info_density", 0.0)
    cand["loopability"]     = _num(seg_features, "loopability", 0.0)
    cand["insight_score"]   = _num(seg_features, "insight_score", 0.0)

    # platform length (v1 + v2). keep both for transparency; scorer already prefers v2 via flag.
    cand["platform_len_match"]        = _num(seg_features, "platform_len_match", 0.0)
    cand["platform_length_score_v2"]  = _num(seg_features, "platform_length_score_v2", 0.0)

    # enhanced extras (optional)
    cand["q_list_score"]    = _num(seg_features, "q_list_score", 0.0)
    cand["prosody_arousal"] = _num(seg_features, "prosody_arousal", 0.0)
    cand["emotion_score"]   = _num(seg_features, "emotion_score", 0.0)
    cand["insight_conf"]    = _num(seg_features, "insight_conf", 0.0)

    # penalties/flags if you surface them elsewhere
    cand["ad_penalty"]      = _num(seg_features, "ad_penalty", 0.0)
    cand["niche_penalty"]   = _num(seg_features, "niche_penalty", 0.0)
    cand["should_exclude"]  = bool(seg_features.get("should_exclude", False))

    # optional versioning for transparency
    if "scoring_version" in seg_features:
        cand["scoring_version"] = seg_features["scoring_version"]
    if "weights_version" in seg_features:
        cand["weights_version"] = seg_features["weights_version"]
    if "flags" in seg_features:
        cand["flags"] = seg_features["flags"]

    return cand

def format_candidates(
    ranked_segments: List[Dict],
    final_genre: str,
    backend_platform: str,
    episode_id: str,
    full_episode_transcript: str = None,
) -> List[Dict]:
    """Convert ranked segments into candidate dictionaries"""
    candidates: List[Dict] = []
    config = get_config()
    for i, seg in enumerate(ranked_segments):
        # For enhanced pipeline, features are directly on the segment object
        # For legacy pipeline, they might be under "features" key
        features = seg.get("features", {})
        if not features:
            # Enhanced pipeline: features are directly on segment
            features = {k: v for k, v in seg.items() if k not in ["start", "end", "text", "id"]}
        
        enhanced_features = {**features, "final_score": seg.get("raw_score", 0.0)}
        # Use display text (auto-capitalized) for title generation to match what user sees
        display_text = _display_text(seg["text"])
        title = _heuristic_title(display_text, enhanced_features, config, rank=i + 1)
        grades = _grade_breakdown(enhanced_features)
        
        # Create full segment data for enhanced formatter
        full_segment_data = {
            "start": seg.get("start"),
            "end": seg.get("end"), 
            "text": seg.get("text", ""),
            **enhanced_features
        }
        
        # Use the new enhanced formatter for individual features
        enhanced_candidate = format_candidate(full_segment_data, platform=backend_platform, genre=final_genre, full_episode_transcript=full_episode_transcript)
        
        # Log the presence of enhanced fields
        logger.debug(
            "candidate features: pl_v2=%.2f q_list=%.2f prosody=%.2f insight_conf=%.2f",
            enhanced_candidate.get("platform_length_score_v2", 0.0),
            enhanced_candidate.get("q_list_score", 0.0),
            enhanced_candidate.get("prosody_arousal", 0.0),
            enhanced_candidate.get("insight_conf", 0.0),
        )
        
        candidate = {
            "id": f"clip_{episode_id}_{i}",
            "title": title,
            "features": features,
            "grades": grades,
            "score": seg.get("raw_score", 0.0),
            "raw_score": seg.get("raw_score", 0),
            "display_score": seg.get("display_score", 0),
            "clip_score_100": seg.get("clip_score_100", 0),
            "confidence": seg.get("confidence", "Low"),
            "confidence_color": seg.get("confidence_color", "gray"),
            "synergy_mult": seg.get("synergy_multiplier", 1.0),
            "winning_path": seg.get("winning_path", "unknown"),
            "path_scores": seg.get("path_scores", {}),
            "moment_type": seg.get("type", "general"),
            "moment_confidence": seg.get("confidence", 0.5),
            "status": "completed",
            
            # Enhanced features from the new formatter
            **enhanced_candidate,
            
            # Legacy display fields for backward compatibility
            "Viral Potential": seg.get("display_score", 0),
            "Hook Power": enhanced_candidate.get("hook_score", 0.0) * 100,
            "Energy Level": enhanced_candidate.get("arousal_score", 0.0) * 100,
            "Payoff Strength": enhanced_candidate.get("payoff_score", 0.0) * 100,
            "Emotion Impact": enhanced_candidate.get("emotion_score", 0.0) * 100,
            "Question Engagement": enhanced_candidate.get("q_list_score", 0.0) * 100,
            "Information Density": enhanced_candidate.get("info_density", 0.0) * 100,
            "Loop Potential": enhanced_candidate.get("loopability", 0.0) * 100,
            "Platform Match": enhanced_candidate.get("platform_length_score_v2", enhanced_candidate.get("platform_len_match", 0.0)) * 100,
        }
        candidates.append(candidate)
    return candidates
