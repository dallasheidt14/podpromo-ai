from typing import Dict, List
import logging

from config_loader import get_config
from services.secret_sauce_pkg import _grade_breakdown, _heuristic_title
from services.title_gen import normalize_platform
from services.title_service import generate_titles

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

def _w_time(w):
    """Extract timestamp from word object, handling multiple field names"""
    for k in ("ts", "start", "start_time"):
        v = w.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None

def _w_text(w):
    """Extract text from word object, handling multiple field names"""
    return (w.get("text") or w.get("word") or "").strip()

def stitch_text_from_words(words, start_s: float, end_s: float) -> str:
    """Stitch full transcript from words within [start_s, end_s] time range"""
    if not words:
        return ""
    
    parts = []
    for w in words:
        time_val = _w_time(w)
        if time_val is not None and start_s <= time_val <= end_s:
            text_val = _w_text(w)
            if text_val:
                parts.append(text_val)
    
    txt = " ".join(parts).strip()
    # Clean up multiple spaces
    return " ".join(txt.split())

def _apply_hook_honesty_cap(features: dict, meta: dict) -> dict:
    """Apply hook honesty cap in fallback mode when clip has unfinished_tail_penalty"""
    is_fallback = bool((meta or {}).get("is_fallback"))
    caps = set((meta or {}).get("caps") or []) | set((meta or {}).get("flags") or [])
    if is_fallback and ("unfinished_tail_penalty" in caps):
        features["hook"] = min(features.get("hook", 0.0), 0.85)  # cap to 85%
    return features

def stitch_full_transcript(episode, start: float, end: float) -> str:
    """Stitch full transcript from episode words/segments within [start, end] time range"""
    # prefer word-level stitching
    words = getattr(episode, "words", None) or []
    buf = []
    if words:
        pad = 0.25  # small guard for boundary rounding
        for w in words:
            ts = _w_time(w)
            if ts is None:
                continue
            if (start - pad) <= ts <= (end + pad):
                t = _w_text(w)
                if t:
                    buf.append(t)
        if buf:
            return " ".join(buf).strip()

    # fallback to segment text overlap
    segs = getattr(episode, "segments", None) or []
    if segs:
        parts = []
        for s in segs:
            s_start = s.get("start", 0.0)
            s_end = s.get("end", 0.0)
            if s_start < end and s_end > start:
                txt = (s.get("text") or "").strip()
                if txt:
                    parts.append(txt)
        if parts:
            return " ".join(parts).strip()

    return ""

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
    episode = None,
) -> List[Dict]:
    """Convert ranked segments into candidate dictionaries"""
    candidates: List[Dict] = []
    config = get_config()
    
    # Episode-level deduplication to prevent title repetition
    used_title_keys: set[str] = set()
    
    def _pick_title_for_candidate(txt: str, platform: str | None, clip_id: str = None, start: float = None, end: float = None):
        from services.title_service import generate_titles, normalize_platform
        import re
        
        # Debug logging to catch parameter issues
        logger.debug(f"_pick_title_for_candidate called with: platform={platform}, clip_id={clip_id}, start={start}, end={end}")
        
        plat = normalize_platform(platform)
        
        # Ensure start and end are valid floats
        try:
            start_float = float(start) if start is not None else 0.0
        except (ValueError, TypeError):
            start_float = 0.0
            
        try:
            end_float = float(end) if end is not None else 0.0
        except (ValueError, TypeError):
            end_float = 0.0
        
        variants = generate_titles(
            txt, 
            platform=plat, 
            n=6, 
            avoid_titles={t for t in used_title_keys},
            episode_id=episode_id,
            clip_id=clip_id
        )
        title = variants[0]["title"] if variants else "Most Leaders Solve the Wrong Problem"
        # remember to avoid repeats for later clips in the same episode
        key = re.sub(r"[^a-z0-9]+","", title.lower())
        used_title_keys.add(key)
        return title
    
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
        
        # Extract timing information and ensure they're floats
        start_time = float(seg.get("start", 0))
        end_time = float(seg.get("end", 0))
        
        # Use new unified title generator with episode-level deduplication
        clip_id = f"clip_{episode_id}_{i}"
        
        # Debug: Check what backend_platform contains
        logger.debug(f"DEBUG: backend_platform={backend_platform}, type={type(backend_platform)}")
        logger.debug(f"DEBUG: start_time={start_time}, end_time={end_time}, type(start)={type(start_time)}")
        
        title = _pick_title_for_candidate(
            txt=display_text, 
            platform=backend_platform, 
            clip_id=clip_id, 
            start=start_time, 
            end=end_time
        )
        
        # Optional: attach variants for UI (if needed later)
        # candidate["title_variants"] = titles
        grades = _grade_breakdown(enhanced_features)
        
        # Create full segment data for enhanced formatter
        full_segment_data = {
            "start": seg.get("start"),
            "end": seg.get("end"), 
            "text": seg.get("text", ""),
            **enhanced_features
        }
        
        # Apply hook honesty cap before formatting
        enhanced_features = _apply_hook_honesty_cap(enhanced_features, seg.get("meta", {}))
        
        # Use the new enhanced formatter for individual features
        enhanced_candidate = format_candidate(full_segment_data, platform=backend_platform, genre=final_genre, full_episode_transcript=full_episode_transcript)
        
        # Stitch full transcript from episode words/segments within [start, end] time range
        if episode:
            full_transcript = stitch_full_transcript(episode, start_time, end_time)
        else:
            # Fallback to existing segment text if no episode data
            full_transcript = seg.get("text", "")
        
        # Always use full transcript for title generation (prefer stitched over display_text)
        title_text = full_transcript if full_transcript else display_text
        
        # Debug: Log what text is being used for title generation
        logger.debug(f"TITLE_GEN_DEBUG: episode={episode_id}, clip={clip_id}, using_transcript={bool(full_transcript)}, text_len={len(title_text)}, text_preview='{title_text[:100]}...'")
        
        title = _pick_title_for_candidate(
            txt=title_text, 
            platform=backend_platform, 
            clip_id=clip_id, 
            start=start_time, 
            end=end_time
        )
        
        # Apply hook honesty cap in fallback mode
        # Check if we're in fallback mode and if this clip has unfinished_tail_penalty
        is_fallback = seg.get("episode_fallback_mode", False) or seg.get("ft_status") in ["sparse_finished", "unresolved"]
        caps = seg.get("caps", []) or seg.get("meta", {}).get("caps", [])
        if is_fallback and "unfinished_tail_penalty" in caps:
            # Cap hook score to avoid "Hook 100%" when tail is clearly soft
            if "hook" in features:
                features["hook"] = min(features.get("hook", 0.0), 0.85)
        
        # Log the presence of enhanced fields
        logger.debug(
            "candidate features: pl_v2=%.2f q_list=%.2f prosody=%.2f insight_conf=%.2f",
            enhanced_candidate.get("platform_length_score_v2", 0.0),
            enhanced_candidate.get("q_list_score", 0.0),
            enhanced_candidate.get("prosody_arousal", 0.0),
            enhanced_candidate.get("insight_conf", 0.0),
        )
        
        # Create snippet for card display
        snippet = full_transcript if len(full_transcript) <= 240 else full_transcript[:240] + "…"
        
        candidate = {
            "id": f"clip_{episode_id}_{i}",
            "title": title,
            "transcript": full_transcript,  # Full stitched transcript
            "text": full_transcript,  # Ensure text field is set
            "snippet": snippet,  # Short snippet for cards
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
            
            # Carry through ft_status from enhanced pipeline
            "ft_status": seg.get("ft_status"),   # <-- carry through
            "ft_meta": seg.get("ft_meta", {}),   # optional but useful for logging
            
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
