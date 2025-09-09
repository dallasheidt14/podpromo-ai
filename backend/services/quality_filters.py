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


def filter_low_quality(candidates: List[Dict], min_score: int = 20) -> List[Dict]:
    """Filter out low-quality candidates with safety net"""
    if not candidates:
        return candidates
    
    # Sort by final_score (highest first)
    sorted_candidates = sorted(candidates, key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)
    
    filtered: List[Dict] = []
    for candidate in sorted_candidates:
        # Use final_score for quality gates, not display_score
        final_score = candidate.get("final_score", 0)
        display_score = candidate.get("display_score", 0)
        
        # Convert final_score to 0-100 range if it's in 0-1 range
        if final_score <= 1.0:
            final_score_100 = final_score * 100
        else:
            final_score_100 = final_score
        
        # Quality gates
        if final_score_100 < min_score:
            continue
            
        text = candidate.get("text", "")
        if len(text.split()) < 10:  # Reduced from 20 to be less strict
            continue
        if text and text[0].islower():
            continue
            
        filtered.append(candidate)
    
    # Safety net: if no candidates passed quality filter, keep the best ones
    if not filtered:
        # Keep top 3 candidates regardless of score
        filtered = sorted_candidates[:3]
        logger.warning(f"No candidates passed quality filter, keeping top {len(filtered)} candidates")
    elif len(filtered) < 2 and len(candidates) >= 2:
        # If only 1 candidate passed, add the second best
        filtered = sorted_candidates[:2]
        logger.warning(f"Only 1 candidate passed quality filter, keeping top 2")
    
    return filtered
