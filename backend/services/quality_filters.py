import logging
from typing import Dict, List, Any
import numpy as np
import re
import math
from collections import Counter
from math import sqrt

# Import unfinished policy constants
from services.secret_sauce_pkg.features import (
    FINISHED_THOUGHT_REQUIRED,
    UNFINISHED_HARD_DROP, UNFINISHED_MALUS, UNFINISHED_CONF_HARD,
    END_GAP_MS_OK, END_CONF_OK, SOFT_ENDING_MALUS,
)

# Deprecated: Legacy ad detection patterns - canonical ad detection now in services/ads.py
# These constants are kept for backward compatibility but are no longer used.
_AD_PATTERNS = [
    r"\b(visit|go to|use code|promo code|sponsored by|brought to you by)\b",
    r"\b(dot com|\.com|/pricing|/app)\b",
    r"\b(terms apply|learn more|sign up|limited time)\b",
    r"\b(call|text)\s+\d{3}[-\s]\d{3}[-\s]\d{4}\b",
    r"\b(start your free trial|shop now|click (the )?link)\b",
    r"\b(our ai agent|24/7 support|money back guarantee)\b",
]

# Brand list for ad detection (deprecated)
_AD_BRANDS = {
    "Mint Mobile", "Granger", "Wise", "Shopify", "BetterHelp", "Squarespace", 
    "NordVPN", "HelloFresh", "Raycon", "SeatGeek", "Audible", "Cash App"
}

# Product-y phrases (deprecated)
_AD_PRODUCT_PHRASES = {
    "fiber blend", "aromatic spices", "formulation", "subscription", 
    "workspace", "AI platform"
}
_AD_RE = re.compile("|".join(_AD_PATTERNS), re.IGNORECASE)

def _ad_like_score(text: str) -> float:
    """Calculate ad likelihood score - delegates to centralized detector"""
    from services.ads import ad_like_score
    return ad_like_score(text)

def is_viable_clip(clip: dict, *, relax: dict | None = None) -> tuple[bool, str]:
    """
    Central viability gate to avoid scattered boolean drops.
    Returns (ok, reason_if_dropped).
    'relax' can override specific thresholds, e.g. {"finished_thresh": 0.60}
    """
    relax = relax or {}
    text = (clip.get("text") or "").strip()
    if not text:
        return False, "empty_text"

    # Ad / promo
    if _ad_like_score(text) >= 0.60 or clip.get("is_advertisement") or clip.get("features", {}).get("is_advertisement"):
        return False, "ad_like"

    # Finished thought gating
    finished_like = bool(clip.get("finished_thought") or clip.get("ft_status") in ("finished", "sparse_finished"))
    ft_cov = float(clip.get("ft_coverage_ratio") or 0.0)
    req_cov = float(relax.get("finished_thresh", 0.66))
    if not finished_like or ft_cov < req_cov:
        return False, "unfinished"

    # Safety (soft example, keep your existing checks if any)
    if clip.get("safety_flag") in ("hate", "sexual_minor", "self_harm"):
        return False, "safety_block"

    return True, ""

logger = logging.getLogger(__name__)

# Safe defaults for when config is not passed
_DEFAULTS = {
    # finish/terminal detection helpers
    "REFINE_MIN_TAIL_SILENCE": 0.12,  # seconds of (near) silence to consider "finished"
    "REFINE_SNAP_MAX_NUDGE":   0.35,  # max nudge when snapping to punct/boundary

    # "question-only" drop rule (no payoff, mostly question/list)
    "DROP_Q_ONLY_QMIN":       0.50,   # q_or_list >= 0.5
    "DROP_Q_ONLY_PAYOFF_MAX": 0.15,   # AND payoff < 0.15

    # hook repetition clamp (only if you're clamping hook for repeated openers)
    "REP_HOOK_FIRST_N_CHARS": 200,
    "REP_HOOK_CLAMP_MAX":     0.65,   # cap normalized hook if repetition is high

    # ad gating (quality gates compare against ad_likelihood computed in features)
    "ALLOW_ADS":   False,     # default: block ads
    "AD_SIG_MIN":  0.60,      # treat >= 0.60 as ad-ish; tune as you like
}

_warned_cfg_none = False
def _cfg(cfg, key, default=None):
    global _warned_cfg_none
    if cfg is None and not _warned_cfg_none:
        logger.warning("quality_filters: cfg=None; using built-in defaults")
        _warned_cfg_none = True
    if default is None:
        default = _DEFAULTS.get(key)
    return getattr(cfg, key, default) if cfg is not None else default

# Tunables
_LONG_SEC_1 = 20.0   # minimum for "long"
_LONG_SEC_2 = 24.0   # encourage at least one in 24–30s range when available
_LONG_MIX_MIN = 2    # target at least 2 long clips in finals

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

def _apply_unfinished_policy(cand) -> bool:
    """
    Returns True to keep (possibly with penalty), False to hard-drop.
    Expects these booleans/floats already present on cand (as you currently compute them):
      - cand['has_terminal_punct']   -> bool
      - cand['last_conf']            -> float in [0,1]
    """
    if not FINISHED_THOUGHT_REQUIRED:
        return True

    has_punct = bool(cand.get('has_terminal_punct', False))
    last_conf = float(cand.get('last_conf', 0.0))

    if has_punct:
        # finished enough — keep with no penalty
        return True

    # No terminal punctuation
    # Only *truly* bad tails are hard-dropped (kill-switch + very low conf)
    if UNFINISHED_HARD_DROP and last_conf < UNFINISHED_CONF_HARD:
        cand.setdefault('gate_flags', []).append('DROP_UNFINISHED_HARD')
        return False

    # Otherwise, keep and attach a small penalty to virality
    cand.setdefault('penalties', {})
    cand['penalties']['unfinished_pen'] = UNFINISHED_MALUS
    cand.setdefault('gate_flags', []).append('UNFINISHED_SOFT')
    return True

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

def filter_low_quality(candidates, mode=None, **_):
    """Backwards-compatible wrapper. `mode` is optional hint ('strict'|'balanced')."""
    # ignore mode for now or map it internally
    return _filter_low_quality_impl(candidates, {})

def _filter_low_quality_impl(candidates, gate_mode):
    """Internal implementation of quality filtering"""
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
        
        # Ad detection - block sponsor reads
        ad_like = candidate.get("ad_like", None)
        if ad_like is None:
            ad_like = _ad_like_score(text)
            candidate["ad_like"] = ad_like
        
        if ad_like >= 0.60:
            reasons.append("ad_like")
        
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
        
        # Relax gates for longer clips (20s+) with strong payoff/arousal
        duration = candidate.get('end', 0) - candidate.get('start', 0)
        if duration >= 20.0:  # Lowered threshold for soft ending
            payoff_score = candidate.get('payoff_score', 0.0)
            arousal_score = candidate.get('arousal_score', 0.0)
            
            # If long clip has strong payoff/arousal, relax strict gates
            if payoff_score >= 0.4 or arousal_score >= 0.4:
                if "too_short" in reasons:
                    reasons.remove("too_short")
                    notes.append("long_clip_relaxation")
                if "too_stopwordy" in reasons and stopword_ratio <= 0.75:
                    reasons.remove("too_stopwordy")
                    notes.append("long_clip_relaxation")
                if "rate_based_quality_fail" in reasons:
                    reasons.remove("rate_based_quality_fail")
                    notes.append("long_clip_relaxation")
                
                # Soft ending criterion: allow conjunction endings for high-value long clips
                if duration >= 20.0 and payoff_score >= 0.5:
                    # Allow clips that end on conjunctions if payoff is very high
                    if not candidate.get('finished_thought', False) and payoff_score >= 0.6:
                        notes.append("soft_ending_high_payoff")
                        # Don't penalize for not being "finished" if payoff is exceptional
                        # Override finished_thought for high-value long clips
                        candidate['finished_thought'] = True
                        candidate['soft_ending'] = True
        
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
    
    # Enforce minimum keep with payoff preference
    if len(text_filtered) < 3:
        # Prioritize clips with payoff > 0 when falling back
        survivors = sorted(candidates, key=lambda c: (
            -(c.get("features", {}).get("payoff_score", 0.0) > 0.0),
            -c.get("final_score", c.get("display_score", 0.0))
        ))
        text_filtered = survivors[:3]
        logger.warning(f"No candidates passed text quality gates, keeping top {len(text_filtered)} candidates (payoff-prioritized)")
    
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
        # Use same payoff-prioritized sorting for additional candidates
        remaining = [c for c in candidates if c not in text_filtered]
        additional_candidates = sorted(remaining, key=lambda c: (
            -(c.get("features", {}).get("payoff_score", 0.0) > 0.0),
            -c.get("final_score", c.get("display_score", 0.0))
        ))[:additional_needed]
        text_filtered.extend(additional_candidates)
        logger.info(f"Applied soft floor: added {len(additional_candidates)} candidates to reach minimum {min_keep} (payoff-prioritized)")
    
    # Helpful debug - log a compact sample of the actual numbers we used
    try:
        sample = [round(s, 3) for s in (scores[:5] if len(scores) <= 5 else scores[:3] + scores[-2:])]
        logger.info(f"Quality filter: spread={spread:.3f}, perc={percentile_threshold}, cutoff={cutoff_score:.3f}, sample={sample}")
    except Exception:
        pass
    
    # ---- FINISHED-THOUGHT BACKSTOP -----------------------------------------
    # If none of the finals are "finished_thought", try to promote one that is.
    # We scan the pre-dedupe survivors so behavior stays stable and bounded.
    try:
        has_finished = any(bool(c.get("finished_thought")) for c in text_filtered)
        if not has_finished:
            # Search among prior survivors (same batch) for the best finished-thought item
            pool = [c for c in candidates if c.get("finished_thought")]
            if pool:
                # Highest score finished-thought that doesn't overlap heavily with an existing final
                pool.sort(key=lambda c: float(c.get("final_score", c.get("score", 0.0))), reverse=True)
                candidate = pool[0]
                # Replace the weakest final to keep count stable
                if text_filtered:
                    text_filtered.sort(key=lambda c: float(c.get("final_score", c.get("score", 0.0))))
                    weakest = text_filtered[0]
                    # Only swap if the finished-thought is not worse than the weakest by a big margin
                    if float(candidate.get("final_score", candidate.get("score", 0.0))) >= float(weakest.get("final_score", weakest.get("score", 0.0))) - 0.05:
                        text_filtered[0] = candidate
                        logger.info("QUALITY_BACKSTOP: promoted finished_thought clip %s", candidate.get("id"))
    except Exception as e:
        logger.error("QUALITY_BACKSTOP_ERROR: %s", e)
    # -----------------------------------------------------------------------

    # ---- DURATION MIX GUARD ------------------------------------------------
    # Ensure at least 2 long (>=20s) clips in finals when such survivors exist.
    try:
        have_long = sum(1 for c in text_filtered if float(c.get("dur", 0.0)) >= _LONG_SEC_1)
        if have_long < _LONG_MIX_MIN:
            pool = [c for c in candidates if float(c.get("dur", 0.0)) >= _LONG_SEC_1]
            # prefer finished_thought and payoff_ok, then by score and length
            pool.sort(key=lambda c: (
                1 if c.get("finished_thought") else 0,
                1 if c.get("payoff_ok") else 0,
                float(c.get("final_score", c.get("score", 0.0))),
                float(c.get("dur", 0.0))
            ), reverse=True)
            while have_long < _LONG_MIX_MIN and pool:
                candidate = pool.pop(0)
                # swap out weakest short
                shorts = [c for c in text_filtered if float(c.get("dur", 0.0)) < _LONG_SEC_1]
                if not shorts:
                    break
                weakest = min(shorts, key=lambda c: float(c.get("final_score", c.get("score", 0.0))))
                idx = text_filtered.index(weakest)
                text_filtered[idx] = candidate
                have_long += 1
                log.info("DIVERSITY_GUARD_LONG: promoted long clip id=%s dur=%.1fs", candidate.get("id"), candidate.get("dur",0.0))
    except Exception as e:
        log.warning("DIVERSITY_GUARD_LONG_ERROR: %s", e)
    # -----------------------------------------------------------------------

    logger.info(f"Quality filter result: {len(candidates)} → {len(text_filtered)} candidates")
    return text_filtered

# --- Fallback-aware gates (drop-in) ------------------------------------------
from collections import Counter
import logging

logger = logging.getLogger(__name__)

_FINISHED_OK = {"finished"}
_SPARSE_OK = {"sparse_finished"}
_END_PUNCT = (".", "!", "?", "…", '"', "'", '"', "'")

def _get_tail_gap_sec(cand) -> float:
    meta = cand.get("meta", {}) if isinstance(cand, dict) else {}
    for k in ("tail_to_next_eos_sec", "tail_gap_sec", "tail_gap_to_next_eos"):
        v = meta.get(k, cand.get(k) if isinstance(cand, dict) else None)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return 1e9  # unknown -> very large

def _ends_with_punct(cand) -> bool:
    txt = cand.get("text") or cand.get("preview") or ""
    return str(txt).strip().endswith(_END_PUNCT)

def _is_finished_like(cand: dict, *, fallback: bool, tail_close_sec: float = 0.75, cfg=None):
    # 1) Hard-drop ads unless explicitly allowed
    allow_ads = _cfg(cfg, "ALLOW_ADS", False)
    if cand.get("ad") and not allow_ads:
        return False, "DROP_AD"
    
    # 2) Drop question-only bait (configurable threshold)
    qmin = _cfg(cfg, "DROP_Q_ONLY_QMIN", 0.50)
    pmax = _cfg(cfg, "DROP_Q_ONLY_PAYOFF_MAX", 0.15)
    if (cand.get("q_list", 0.0) >= qmin and
        cand.get("payoff", 0.0)  <  pmax):
        return False, "DROP_Q_ONLY"
    
    # 3) Classic bait: huge hook, no payoff
    if cand.get("hook", 0.0) >= 0.80 and cand.get("payoff", 0.0) < 0.10:
        return False, "DROP_BAIT_NO_PAYOFF"
    
    # ft_classifier can be "finished" / "sparse_finished" / "missing"
    ft = (cand.get("ft_classifier") or "").lower()
    if ft in ("finished", "sparse_finished"):
        cand["finished_thought"] = True
        cand["finish_reason"] = "ft_classifier"
        return True, "KEEP_FT"

    # explicit finished flag from upstream scoring, if present
    if cand.get("finished_thought") is True:
        return True, "KEEP_FLAG_FINISHED"

    # Enhanced finished thought detection
    text = cand.get("text", "")
    dur = cand.get("dur", 0.0)
    gap = cand.get("tail_gap_sec")
    
    # Log diagnostic info
    close_to_eos_ms = int((gap or 999.0) * 1000)
    pause_at_tail_ms = 0  # Would need actual pause detection
    ended_with_punct = bool(text and text.strip()[-1] in ".?!")
    
    logger.debug(f"CLOSE_TO_EOS_MS={close_to_eos_ms}, PAUSE_AT_TAIL_MS={pause_at_tail_ms}, ENDED_WITH_PUNCT_BOOL={ended_with_punct}")
    
    # Accept if end is within ≤0.6–0.8s of a strong EOS or pause≥350ms
    if gap is not None and gap <= tail_close_sec:
        cand["finished_thought"] = True
        cand["finish_reason"] = "eos_hit"
        return True, "KEEP_TAIL_CLOSE"
    
    # Check for discourse closers first (highest priority)
    discourse_closers = ["and that's why", "so we", "that's why", "which is why", "this is why"]
    text_lower = text.lower()
    if any(closer in text_lower for closer in discourse_closers):
        cand["finished_thought"] = True
        cand["finish_reason"] = "discourse_closer"
        return True, "KEEP_DISCOURSE_CLOSER"
    
    # Don't drop strong clips that barely miss EOS (platform-neutral)
    if (cand.get("payoff_ok") and
        gap is not None and gap <= 0.45):
        cand["finished_thought"] = True
        cand["finish_reason"] = "close_eos"
        return True, "KEEP_TAIL_CLOSE"

    # Confidence-based finished thought detection (soft decision)
    from services.util import calculate_finish_confidence
    words = cand.get("words") or []
    confidence = calculate_finish_confidence(cand, words)
    
    # Convert gap to milliseconds for comparison
    tail_gap_ms = (gap or 0.0) * 1000.0
    
    # Priority 1: Terminal punctuation → accept, no malus
    if ended_with_punct:
        cand["finished_thought"] = True
        cand["finish_reason"] = "terminal_punct"
        return True, "KEEP_TERMINAL_PUNCT"
    
    # Priority 2: Soft ending via gap or confidence
    if tail_gap_ms >= END_GAP_MS_OK or (confidence is not None and confidence >= END_CONF_OK):
        cand["finished_thought"] = True
        cand["finish_reason"] = "soft_ending"
        # Apply soft ending malus
        cand.setdefault('penalties', {})
        cand['penalties']['soft_ending'] = SOFT_ENDING_MALUS
        cand.setdefault('gate_flags', []).append('SOFT_ENDING_OK')
        
        # Track which path triggered for logging
        if tail_gap_ms >= END_GAP_MS_OK:
            cand.setdefault('gate_flags', []).append('SOFT_ENDING_VIA_GAP')
        if confidence is not None and confidence >= END_CONF_OK:
            cand.setdefault('gate_flags', []).append('SOFT_ENDING_VIA_CONF')
            
        return True, "SOFT_ENDING_OK"
    
    # Priority 3: Unfinished policy (unchanged from previous implementation)
    # Prepare the candidate with required fields for the policy
    cand["has_terminal_punct"] = ended_with_punct
    cand["last_conf"] = confidence
    
    # Hard drop only if (no punctuation) AND (confidence < UNFINISHED_CONF_HARD)
    if (confidence is not None) and (confidence < UNFINISHED_CONF_HARD):
        cand.setdefault('gate_flags', []).append('DROP_UNFINISHED_HARD')
        logger.debug("DROP_UNFINISHED_HARD (conf=%.2f < %.2f)", confidence, UNFINISHED_CONF_HARD)
        return False, "DROP_UNFINISHED_HARD"
    
    # Otherwise soft unfinished → keep with unfinished malus
    cand["finished_thought"] = True
    cand["finish_reason"] = "unfinished_soft"
    cand.setdefault('penalties', {})
    cand['penalties']['unfinished_pen'] = UNFINISHED_MALUS
    cand.setdefault('gate_flags', []).append('UNFINISHED_SOFT')
    return True, "UNFINISHED_SOFT"

def _apply_quality_gates_fallback_aware(candidates, *, fallback: bool, tail_close_sec: float, cfg=None):
    kept = []
    reasons = Counter()
    for c in candidates:
        ok, tag = _is_finished_like(c, fallback=fallback, tail_close_sec=tail_close_sec, cfg=cfg)
        reasons[tag] += 1
        if ok:
            kept.append(c)
    logger.info("GATES: kept=%d/%d, reasons=%s", len(kept), len(candidates), dict(reasons))
    
    # Log unfinished policy statistics
    soft = sum(1 for c in kept if 'UNFINISHED_SOFT' in c.get('gate_flags', []))
    hard = reasons.get('DROP_UNFINISHED_HARD', 0)
    logger.info("UNFINISHED_POLICY: soft=%d hard=%d malus=%.3f", soft, hard, UNFINISHED_MALUS)
    
    # Log soft ending policy statistics
    soft_ending_ok = sum(1 for c in kept if 'SOFT_ENDING_OK' in c.get('gate_flags', []))
    via_gap = sum(1 for c in kept if 'SOFT_ENDING_VIA_GAP' in c.get('gate_flags', []))
    via_conf = sum(1 for c in kept if 'SOFT_ENDING_VIA_CONF' in c.get('gate_flags', []))
    logger.info("SOFT_ENDING_POLICY: ok=%d via_gap=%d via_conf=%d malus=%.3f", 
                soft_ending_ok, via_gap, via_conf, SOFT_ENDING_MALUS)
    
    return kept, dict(reasons)

# Public API expected by clip_score.py (compat layer)
def essential_gates(candidates, *, fallback: bool = False, tail_close_sec: float = 1.5, cfg=None):
    """
    Fallback-aware replacement for the old gating function.
    - Accepts 'finished' always.
    - When fallback=True, also accepts 'sparse_finished'.
    - Allows 'unfinished_tail_penalty' if very near boundary or strong punctuation.
    Returns (kept_candidates, reasons_dict).
    """
    if not candidates:
        logger.info("GATES: kept=0/0, reasons=%s", {})
        return [], {}
    return _apply_quality_gates_fallback_aware(
        candidates, fallback=fallback, tail_close_sec=tail_close_sec, cfg=cfg
    )

def _tfidf_embed(texts: list[str]) -> list[dict[str,float]]:
    """Create TF-IDF embeddings for semantic similarity"""
    docs = [Counter(re.findall(r"[a-z0-9]+", t.lower())) for t in texts]
    df = Counter()
    for d in docs: df.update(d.keys())
    N = len(texts)
    vecs = []
    for d in docs:
        denom = float(sum(d.values()) or 1)
        v = {}
        for w,c in d.items():
            idf = math.log((N+1)/(1+df[w])) + 1.0
            v[w] = (c/denom) * idf
        vecs.append(v)
    return vecs

def _cos(a: dict, b: dict) -> float:
    """Cosine similarity between two TF-IDF vectors"""
    common = set(a) & set(b)
    num = sum(a[w]*b[w] for w in common)
    da = sqrt(sum(x*x for x in a.values())); db = sqrt(sum(x*x for x in b.values()))
    return 0.0 if da==0 or db==0 else (num/(da*db))

def mmr_select_semantic(cands: list[dict], K: int, lam: float = 0.7) -> list[dict]:
    """Semantic MMR selection using TF-IDF embeddings"""
    if not cands: return []
    texts = [c["text"] for c in cands]
    E = _tfidf_embed(texts)
    selected = [0]  # assume input is relevance-sorted
    while len(selected) < min(K, len(cands)):
        best_idx, best_score = None, -1e9
        for i in range(len(cands)):
            if i in selected: continue
            rel = cands[i].get("rank_score", cands[i].get("display_score", 0.0))
            div = max(_cos(E[i], E[j]) for j in selected) if selected else 0.0
            mmr_val = lam*rel - (1.0-lam)*div
            if mmr_val > best_score:
                best_idx, best_score = i, mmr_val
        selected.append(best_idx)
    return [cands[i] for i in selected]
