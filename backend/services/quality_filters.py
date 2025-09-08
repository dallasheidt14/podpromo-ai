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


def filter_overlapping_candidates(candidates: List[Dict], min_gap: float = 15.0) -> List[Dict]:
    """Remove overlapping candidates, keeping highest scoring ones"""
    if not candidates:
        return candidates
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    filtered: List[Dict] = []
    for candidate in sorted_candidates:
        overlaps = False
        for selected in filtered:
            if candidate["start"] < selected["end"] and candidate["end"] > selected["start"]:
                overlaps = True
                break
        if not overlaps:
            filtered.append(candidate)
    return filtered


def filter_low_quality(candidates: List[Dict], min_score: int = 40) -> List[Dict]:
    """Filter out low-quality candidates"""
    filtered: List[Dict] = []
    for candidate in candidates:
        if candidate.get("display_score", 0) < min_score:
            continue
        text = candidate.get("text", "")
        if len(text.split()) < 20:
            continue
        if text and text[0].islower():
            continue
        filtered.append(candidate)
    if len(filtered) < 3 and len(candidates) >= 3:
        return candidates[:5]
    return filtered
