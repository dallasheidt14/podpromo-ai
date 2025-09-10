import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

def calculate_resolution_delta(candidate: Dict[str, Any]) -> float:
    """Calculate resolution_delta: payoff curve increase over the last ~25% of the clip window"""
    payoff_score = candidate.get("payoff_score", 0.0)
    duration = candidate.get("end", 0) - candidate.get("start", 0)
    
    if duration <= 0:
        return 0.0
    
    # For now, use a simple heuristic: if payoff is high and duration is reasonable, assume some resolution
    # In a full implementation, this would analyze the actual payoff curve over time
    if payoff_score >= 0.25 and duration >= 8.0:
        return min(0.25, payoff_score * 0.5)  # Cap at 0.25, scale with payoff
    
    return 0.0

def passes_text_gates(candidate: Dict[str, Any]) -> bool:
    """Rate-based text quality gates with finished-thought exception"""
    duration = max(candidate.get("end", 0) - candidate.get("start", 0), 0.001)
    
    # Get raw counts (these should be passed from scoring stage)
    filler_count = candidate.get("filler_count", 0)
    umuh_count = candidate.get("umuh_count", 0) 
    silence_ms = candidate.get("silence_ms", 0)
    
    # Calculate rates
    filler_rate = filler_count / duration
    umuh_rate = umuh_count / duration
    silence_rate = silence_ms / (duration * 1000.0)
    
    # Base rate thresholds
    ok = (filler_rate <= 0.18 and umuh_rate <= 0.12 and silence_rate <= 0.18)
    
    # Finished-thought exception: allow +25% tolerance
    if not ok and (candidate.get("finished_thought") or candidate.get("resolution_delta", 0) > 0.12):
        tol = 1.25
        ok = (filler_rate <= 0.18 * tol and umuh_rate <= 0.12 * tol and silence_rate <= 0.18 * tol)
    
    return ok

# --- helpers (place near top) -----------------------------------------------

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _extract_times(obj):
    """
    Accepts a candidate dict or a raw features dict and tries multiple keys.
    Returns (start, end) in SECONDS or (None, None) if unavailable.
    """
    # common keys in your pipeline
    keys = [
        ("start", "end"),
        ("start_time", "end_time"),
        ("segment_start", "segment_end"),
        ("t_start", "t_end"),
    ]
    for k0, k1 in keys:
        s = _as_float(obj.get(k0)) if isinstance(obj, dict) else None
        e = _as_float(obj.get(k1)) if isinstance(obj, dict) else None
        if s is not None and e is not None:
            return s, e
    return None, None

def _normalize_range(s, e):
    """
    Returns a normalized (start, end) or None if invalid.
    Handles swapped order and zero/negative spans.
    """
    if s is None or e is None:
        return None
    # fix swapped
    if e < s:
        s, e = e, s
    # discard degenerate
    if e - s <= 0.0:
        return None
    return s, e

def iou_time(a, b):
    """
    Safe IoU on 1D time intervals.
    - Returns 0.0 if either interval is missing/invalid.
    - Never raises on None.
    """
    times_a = _extract_times(a)
    times_b = _extract_times(b)
    if times_a is None or times_b is None:
        return 0.0
    
    sa, ea = times_a
    sb, eb = times_b
    norm_a = _normalize_range(sa, ea)
    norm_b = _normalize_range(sb, eb)
    if norm_a is None or norm_b is None:
        return 0.0
    
    sa, ea = norm_a
    sb, eb = norm_b
    inter = max(0.0, min(ea, eb) - max(sa, sb))
    if inter <= 0.0:
        return 0.0
    union = (ea - sa) + (eb - sb) - inter
    # guard: union should be > 0, but be defensive
    return inter / union if union > 0.0 else 0.0


def fails_quality(feats: dict) -> str | None:
    """Check if segment fails quality gates (soft reject)"""
    payoff = feats.get("payoff_score", 0.0)
    arousal = feats.get("arousal_score", 0.0)
    question = feats.get("question_score", 0.0)
    if payoff >= 0.6 and arousal >= 0.45:
        hook_threshold = 0.06
    elif payoff >= 0.4 and arousal >= 0.35:
        hook_threshold = 0.10
    else:
        hook_threshold = 0.15
    hook = feats.get("hook_score", 0.0)
    weak_hook = hook < hook_threshold
    has_early_question = question >= 0.50
    no_payoff = payoff < 0.25
    ad_like = feats.get("_ad_flag", False) or (feats.get("_ad_penalty", 0.0) >= 0.3)
    if hook < 0.08 and (ad_like or no_payoff):
        if ad_like and no_payoff:
            return "ad_like;weak_hook;no_payoff"
        if ad_like:
            return "ad_like;weak_hook"
        return "weak_hook;no_payoff"
    if hook < 0.08:
        return "weak_hook_very_soft"
    if weak_hook:
        return "weak_hook_mild_soft"
    if no_payoff and not has_early_question:
        return "no_payoff"
    if arousal < 0.20:
        return "low_energy"
    return None


def prefer_finished(a, b):
    """Family-aware NMS: prefer finished thoughts over unfinished ones from same family"""
    # Check if they're from the same family (overlapping time ranges)
    if iou_time(a, b) >= 0.5:  # High overlap suggests same family
        a_finished = a.get("finished_thought", 0) == 1
        b_finished = b.get("finished_thought", 0) == 1
        
        # Prefer finished thoughts
        if a_finished != b_finished:
            return a if a_finished else b
    
    # Fallback: higher final score wins
    return a if a.get("final_score", 0) >= b.get("final_score", 0) else b

def nms_by_time(candidates, iou_thresh=0.55, logger=None):
    """
    Keep highest-scoring in overlapping groups using time IoU.
    - Ignores invalid intervals (treats as IoU=0 with others, so they won't knock others out).
    - Family-aware tie-breaks: prefer resolved longer variants within same family.
    - Never returns < min_keep if you pass min_keep afterward in your pipeline.
    """
    if not candidates:
        return []

    # stable sort by tie-breakers - favor longer clips on ties
    def key(c):
        fs = float(c.get("final_score", 0.0))
        po = float(c.get("payoff_score", 0.0))
        pl_v2 = float(c.get("platform_length_score_v2", 0.0))
        length = float(c.get("end", 0) - c.get("start", 0)) if c.get("end") and c.get("start") else 0.0
        st = c.get("start")
        st = float(st) if st is not None else 1e12  # push Nones to the end
        return (-fs, -pl_v2, -length, -po, st)

    def nms_tiebreak(a, b):
        """Family-aware tie-breaking for NMS"""
        # Use the prefer_finished function for family-aware logic
        preferred = prefer_finished(a, b)
        return -1 if preferred == a else 1

    sorted_cands = sorted(candidates, key=key)
    kept = []

    for c in sorted_cands:
        drop = False
        for k in kept:
            if iou_time(c, k) >= iou_thresh:
                # Use family-aware tie-breaking
                if nms_tiebreak(c, k) > 0:  # k wins, drop c
                    drop = True
                    break
                else:  # c wins, drop k
                    kept.remove(k)
                    if logger:
                        sc = k.get("final_score", 0.0)
                        logger.debug("NMS family-aware drop: score=%.3f start=%s end=%s", sc, k.get("start"), k.get("end"))
        if not drop:
            kept.append(c)
        else:
            if logger:
                sc = c.get("final_score", 0.0)
                logger.debug("NMS drop: score=%.3f start=%s end=%s", sc, c.get("start"), c.get("end"))

    if logger:
        logger.info("NMS: %d -> %d (thresh=%.2f)", len(candidates), len(kept), iou_thresh)
    return kept

def filter_overlapping_candidates(candidates: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Non-Maximum Suppression for overlapping candidates using IoU (legacy wrapper)"""
    return nms_by_time(candidates, iou_thresh=iou_threshold, logger=logger)


def _scores(cands):
    """Extract final_score floats from candidates, ensuring they're in [0,1] range"""
    vals = []
    for c in cands:
        v = c.get("final_score")  # THE canonical field in [0,1]
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals

def filter_low_quality(candidates: List[Dict], min_score: int = 20) -> List[Dict]:
    """Filter out low-quality candidates using episode-relative percentile + rank gating"""
    if not candidates:
        return candidates
    
    import numpy as np
    
    # Extract final scores for percentile calculation (ONLY final_score)
    scores = _scores(candidates)
    if not scores:
        # Nothing usable; just keep top-k as a failsafe
        return sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)[:3]
    
    # Duration-based percentile filtering
    micro_candidates = [c for c in candidates if (c.get('end', 0) - c.get('start', 0)) < 14.0]
    normal_candidates = [c for c in candidates if 14.0 <= (c.get('end', 0) - c.get('start', 0)) <= 28.0]
    
    kept_by_percentile = []
    
    # Micro clips: use p70
    if micro_candidates:
        micro_scores = _scores(micro_candidates)
        if micro_scores:
            k = max(1, int(0.10 * len(micro_scores)))
            trimmed_micro = micro_scores[:-k] if len(micro_scores) > 10 else micro_scores
            micro_cutoff = float(np.percentile(trimmed_micro, 70))
            micro_cutoff = min(micro_cutoff, 0.80)
            kept_by_percentile.extend([c for c in micro_candidates if c.get("final_score", 0.0) >= micro_cutoff])
    
    # Normal clips: use p60 (or p55 if episode is long)
    if normal_candidates:
        normal_scores = _scores(normal_candidates)
        if normal_scores:
            # Check if episode is long (total duration > 25 minutes)
            total_duration = max(c.get('end', 0) for c in candidates) - min(c.get('start', 0) for c in candidates)
            percentile_threshold = 55 if total_duration > 25 * 60 else 60
            
            k = max(1, int(0.10 * len(normal_scores)))
            trimmed_normal = normal_scores[:-k] if len(normal_scores) > 10 else normal_scores
            normal_cutoff = float(np.percentile(trimmed_normal, percentile_threshold))
            normal_cutoff = min(normal_cutoff, 0.80)
            kept_by_percentile.extend([c for c in normal_candidates if c.get("final_score", 0.0) >= normal_cutoff])
    
    # Fallback: if no duration-based filtering worked, use original logic
    if not kept_by_percentile:
        mn, mx = min(scores), max(scores)
        spread = mx - mn
        percentile_threshold = 60 if spread < 0.20 else 70
        k = max(1, int(0.10 * len(scores)))
        trimmed_scores = scores[:-k] if len(scores) > 10 else scores
        cutoff_score = float(np.percentile(trimmed_scores, percentile_threshold))
        cutoff_score = min(cutoff_score, 0.80)
        kept_by_percentile = [c for c in candidates if c.get("final_score", 0.0) >= cutoff_score]
    
    # Always keep top-3 by final_score
    top3 = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)[:3]
    top3_ids = {id(c) for c in top3}
    kept_ids = {id(c) for c in kept_by_percentile}
    union_ids = kept_ids | top3_ids
    
    # Protected slot for excellent platform fit (never drop best long, well-fit candidate)
    def sort_key(c):
        return (round(c.get('final_score', 0), 3),
                round(c.get('platform_length_score_v2', 0.0), 3),
                round(c.get('end', 0) - c.get('start', 0), 2))
    
    # Protected slot for excellent platform fit (never drop best long, well-fit candidate)
    protected = [c for c in candidates 
                if c.get('platform_length_score_v2', 0) >= 0.90 
                and (c.get('end', 0) - c.get('start', 0)) >= 18.0]
    if protected:
        best_protected = sorted(protected, key=sort_key, reverse=True)[0]
        union_ids.add(id(best_protected))
        logger.info(f"PROTECTED: keeping long platform-fit clip dur={best_protected.get('end', 0) - best_protected.get('start', 0):.1f}s pl_v2={best_protected.get('platform_length_score_v2', 0):.2f} score={best_protected.get('final_score', 0):.3f}")
    
    # Protected resolver: rescue one "well-fit resolver" with good payoff
    protected_resolver = [c for c in candidates
                         if c.get('platform_length_score_v2', 0.0) >= 0.90
                         and c.get('payoff_score', 0.0) >= 0.35
                         and (c.get('end', 0) - c.get('start', 0)) >= 18.0]
    if protected_resolver:
        best_resolver = max(protected_resolver, key=lambda c: c.get('final_score', 0))
        best_resolver["protected"] = True  # Mark as protected
        union_ids.add(id(best_resolver))
        logger.info(f"PROTECTED_RESOLVER: keeping long resolving clip dur={best_resolver.get('end', 0) - best_resolver.get('start', 0):.1f}s pl_v2={best_resolver.get('platform_length_score_v2', 0):.2f} payoff={best_resolver.get('payoff_score', 0):.2f}")
    
    # Resolver protection: rescue one completed sentence with good payoff
    resolvers = [c for c in candidates
                if c.get('payoff_score', 0.0) >= 0.35
                and c.get('text', '').strip().endswith(('.', '!', '?'))
                and 18.0 <= (c.get('end', 0) - c.get('start', 0)) <= 28.0
                and c.get('platform_length_score_v2', 0.0) >= 0.85]
    if resolvers:
        best_resolver = max(resolvers, key=lambda c: c.get('final_score', 0))
        best_resolver["protected"] = True  # Mark as protected
        union_ids.add(id(best_resolver))
        logger.info(f"PROTECTED_RESOLVER: kept resolved cut dur={best_resolver.get('end', 0) - best_resolver.get('start', 0):.1f}s payoff={best_resolver.get('payoff_score', 0):.2f}")
    
    kept = [c for c in candidates if id(c) in union_ids]
    
    # Add resolution_delta to candidates if not present
    for candidate in kept:
        if "resolution_delta" not in candidate:
            candidate["resolution_delta"] = calculate_resolution_delta(candidate)
    
    # Additional quality gates (text-based) with rate-based metrics
    text_filtered = []
    drop_reasons = {}
    
    for candidate in kept:
        text = candidate.get("text", "")
        words = len(text.split())
        score = candidate.get("final_score", 0)
        hook_score = candidate.get("hook_score", 0)
        payoff_score = candidate.get("payoff_score", 0)
        info_density = candidate.get("info_density", 0)
        platform_length_score_v2 = candidate.get("platform_length_score_v2", 0)
        
        # Check rejection criteria
        reasons = []
        penalties = 0.0
        notes = []
        
        # Basic length check (keep this as absolute)
        if words < 8:
            reasons.append("too_short")
        
        # Check for bait without payoff
        if hook_score > 0.80 and payoff_score < 0.10:
            reasons.append("bait_no_payoff")
        
        # Require "shorts with substance" - for words < 10, demand either payoff >= 0.25 or info_density >= 0.60
        if words < 10 and payoff_score < 0.25 and info_density < 0.60:
            # Relax threshold for high-info content
            if info_density >= 0.78 and words >= 8:
                notes.append("short_but_high_info")
            else:
                reasons.append("too_short_low_value")
        
        # Check stopword ratio
        if text:
            stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            words_list = text.lower().split()
            stopword_count = sum(1 for word in words_list if word in stopwords)
            stopword_ratio = stopword_count / len(words_list) if words_list else 0
            if stopword_ratio > 0.65:
                reasons.append("too_stopwordy")
        
        # NEW: Rate-based text quality gates with finished-thought exception
        if not passes_text_gates(candidate):
            reasons.append("rate_based_quality_fail")
        
        # Apply minor penalties for style issues
        if text and text[0].islower():
            penalties += 0.02     # tiny nudge
            notes.append("starts_lowercase")
        
        # Relax thresholds for high-quality content
        if info_density >= 0.65 or platform_length_score_v2 >= 0.6:
            # Relax thresholds for high-quality content
            if "too_short" in reasons and words >= 6:
                reasons.remove("too_short")
            if "too_stopwordy" in reasons and stopword_ratio <= 0.70:
                reasons.remove("too_stopwordy")
            if "rate_based_quality_fail" in reasons:
                # Allow finished-thought clips to pass even with higher rates
                if candidate.get("finished_thought") or candidate.get("resolution_delta", 0) > 0.12:
                    reasons.remove("rate_based_quality_fail")
                    notes.append("rate_exception_applied")
        
        if not reasons:
            # Apply penalties to final score
            if penalties > 0:
                candidate['final_score'] = max(0.0, candidate.get('final_score', 0.0) - penalties)
            text_filtered.append(candidate)
        else:
            # Track rejection reasons
            for reason in reasons:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    
    # Log rejection summary
    if drop_reasons:
        reason_summary = ", ".join([f"{reason}: {count}" for reason, count in drop_reasons.items()])
        logger.info(f"Quality filter rejections: {reason_summary}")
    
    # Enforce minimum keep
    if len(text_filtered) < 3:
        text_filtered = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)[:3]
        logger.warning(f"No candidates passed text quality gates, keeping top {len(text_filtered)} candidates")
    
    # Dynamic soft floor based on episode size
    min_keep = 3  # Default minimum
    
    # Adjust based on episode size
    if len(candidates) >= 8:
        min_keep = 4
    if len(candidates) >= 60:  # Large episode
        min_keep = max(min_keep, 5)
    
    # Check if we need to add more candidates
    if len(text_filtered) < min_keep:
        additional_needed = min_keep - len(text_filtered)
        additional_candidates = [c for c in sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True) if c not in text_filtered][:additional_needed]
        text_filtered.extend(additional_candidates)
        logger.info(f"Applied soft floor: added {len(additional_candidates)} candidates to reach minimum {min_keep}")
    
    # Helpful debug - log a compact sample of the actual numbers we used
    try:
        sample = [round(s, 3) for s in (scores[:5] if len(scores) <= 5 else scores[:3] + scores[-2:])]
        logger.info(f"Quality filter: spread={spread:.3f}, perc={percentile_threshold}, cutoff={cutoff_score:.3f}, sample={sample}")
    except Exception:
        pass
    
    logger.info(f"Quality filter result: {len(candidates)} → {len(text_filtered)} candidates")
    return text_filtered
