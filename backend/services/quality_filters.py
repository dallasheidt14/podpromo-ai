import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


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


def filter_overlapping_candidates(candidates: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Non-Maximum Suppression for overlapping candidates using IoU"""
    if not candidates:
        return candidates
    
    # Sort by final_score (highest first)
    sorted_candidates = sorted(candidates, key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)
    
    def calculate_iou(cand1: Dict, cand2: Dict) -> float:
        """Calculate Intersection over Union for time overlap"""
        start1, end1 = cand1.get("start", 0), cand1.get("end", 0)
        start2, end2 = cand2.get("start", 0), cand2.get("end", 0)
        
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        # Calculate union
        union = (end1 - start1) + (end2 - start2) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    filtered: List[Dict] = []
    for candidate in sorted_candidates:
        # Check if this candidate overlaps significantly with any already selected
        should_keep = True
        for selected in filtered:
            iou = calculate_iou(candidate, selected)
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            filtered.append(candidate)
    
    # Safety net: if no candidates survived, keep the best one
    if not filtered and candidates:
        filtered = [candidates[0]]
    
    return filtered


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
    
    # Calculate episode-relative thresholds
    mn, mx = min(scores), max(scores)
    spread = mx - mn
    
    # Low-contrast relaxer: if spread < 0.20, use p60 instead of p70
    percentile_threshold = 60 if spread < 0.20 else 70
    cutoff_score = float(np.percentile(scores, percentile_threshold))
    
    # Keep by percentile
    kept_by_percentile = [c for c in candidates if c.get("final_score", 0.0) >= cutoff_score]
    
    # Always keep top-3 by final_score
    top3 = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)[:3]
    top3_ids = {id(c) for c in top3}
    kept_ids = {id(c) for c in kept_by_percentile}
    union_ids = kept_ids | top3_ids
    kept = [c for c in candidates if id(c) in union_ids]
    
    # Additional quality gates (text-based)
    text_filtered = []
    for candidate in kept:
        text = candidate.get("text", "")
        if len(text.split()) < 8:  # Reduced from 10 to be less strict
            continue
        if text and text[0].islower():
            continue
        text_filtered.append(candidate)
    
    # Enforce minimum keep
    if len(text_filtered) < 3:
        text_filtered = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)[:3]
        logger.warning(f"No candidates passed text quality gates, keeping top {len(text_filtered)} candidates")
    
    # Soft floor: if we started with many segments but ended with few, bump to min 3
    if len(text_filtered) < 3 and len(candidates) >= 8:
        # Add more from top candidates
        additional_needed = 3 - len(text_filtered)
        additional_candidates = [c for c in sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True) if c not in text_filtered][:additional_needed]
        text_filtered.extend(additional_candidates)
        logger.info(f"Applied soft floor: added {len(additional_candidates)} candidates to reach minimum 3")
    
    # Helpful debug - log a compact sample of the actual numbers we used
    try:
        sample = [round(s, 3) for s in (scores[:5] if len(scores) <= 5 else scores[:3] + scores[-2:])]
        logger.info(f"Quality filter: spread={spread:.3f}, perc={percentile_threshold}, cutoff={cutoff_score:.3f}, sample={sample}")
    except Exception:
        pass
    
    logger.info(f"Quality filter result: {len(candidates)} → {len(text_filtered)} candidates")
    return text_filtered
