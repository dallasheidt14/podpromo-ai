import logging
import math
import hashlib
import re
from typing import Dict, List

from config.settings import (
    PRERANK_ENABLED,
    TOP_K_RATIO,
    TOP_K_MIN,
    TOP_K_MAX,
    STRATIFY_ENABLED,
    SAFETY_KEEP_ENABLED,
    PRERANK_WEIGHTS,
    DURATION_TARGET_MIN,
    DURATION_TARGET_MAX,
    ENABLE_EXPLORATION,
    EXPLORATION_QUOTA,
    EXPLORATION_MIN,
)
from services.progress_writer import write_progress

logger = logging.getLogger(__name__)


def _duration_fit(duration: float) -> float:
    """Calculate duration fit score (0-1) for pre-rank scoring"""
    if duration < DURATION_TARGET_MIN:
        return duration / DURATION_TARGET_MIN
    if duration > DURATION_TARGET_MAX:
        return max(0.0, 1 - (duration - DURATION_TARGET_MAX) / DURATION_TARGET_MAX)
    return 1.0


def _hook_proxy(text: str) -> float:
    """Cheap hook detection for pre-rank scoring"""
    if not text:
        return 0.0
    text_lower = text.lower().strip()
    hook_patterns = [
        r"^(how|why|what|top\s?\d+|the secret|3 ways|you won.t believe)",
        r"^(here.s|let me|i.m going to|the thing is)",
        r"^(did you know|here.s the thing|the secret to)",
        r"^(listen|look|wait|hold on)",
    ]
    for pattern in hook_patterns:
        if re.match(pattern, text_lower):
            return 1.0
    return 0.0


def _arousal_proxy(text: str) -> float:
    """Cheap arousal detection for pre-rank scoring"""
    if not text:
        return 0.0
    arousal_score = 0.0
    arousal_score += min(1.0, text.count("!") * 0.2)
    arousal_score += min(1.0, text.count("?") * 0.15)
    arousal_score += min(1.0, text.count("wow") * 0.3)
    arousal_score += min(1.0, text.count("crazy") * 0.3)
    arousal_score += min(1.0, text.count("amazing") * 0.3)
    if "?" in text and len(text) < 160:
        arousal_score += 0.2
    return min(1.0, arousal_score)


def _info_density_proxy(words_per_sec: float) -> float:
    """Calculate info density score (sweet spot 2.5-4.5 wps)"""
    if words_per_sec <= 0:
        return 0.0
    return max(0.0, 1 - abs(words_per_sec - 3.5) / 3.5)


def _has_numbers(text: str) -> float:
    """Check if text contains numbers (lists, statistics, etc.)"""
    return 1.0 if re.search(r"\d", text or "") else 0.0


def _ad_proxy(text: str) -> bool:
    """Cheap text-based advertisement detection"""
    if not text:
        return False
    t = text.lower()
    ad_phrases = [
        "sponsored by",
        "brought to you by",
        "use code",
        "promo code",
        "check out",
        "visit our",
        "subscribe",
        "my course",
        "my book",
        "link in bio",
        "special offer",
        "limited time",
    ]
    return any(p in t for p in ad_phrases)


def pre_rank_candidates(segments: List[Dict], episode_id: str) -> List[Dict]:
    """Pre-rank candidates using cheap features only"""
    if not PRERANK_ENABLED:
        return segments
    logger.info("Starting pre-rank scoring for %d segments", len(segments))
    write_progress(episode_id, "scoring:prerank", 10, "Pre-ranking candidates...")
    for seg in segments:
        if not seg.get("is_advertisement", False):
            seg["is_advertisement"] = _ad_proxy(seg.get("text", ""))
    candidates = [seg for seg in segments if not seg.get("is_advertisement", False)]
    for seg in candidates:
        duration = seg.get("duration", 0)
        text = seg.get("text", "")
        words_per_sec = seg.get("words_per_sec", 0)
        hook_score = _hook_proxy(text)
        arousal_score = _arousal_proxy(text)
        info_density = _info_density_proxy(words_per_sec)
        duration_fit = _duration_fit(duration)
        has_numbers = _has_numbers(text)
        is_ad_penalty = 1.0 if seg.get("is_advertisement", False) else 0.0
        prerank_score = (
            PRERANK_WEIGHTS["hook"] * hook_score
            + PRERANK_WEIGHTS["arousal"] * arousal_score
            + PRERANK_WEIGHTS["info_density"] * info_density
            + PRERANK_WEIGHTS["duration_fit"] * duration_fit
            + PRERANK_WEIGHTS["has_numbers"] * has_numbers
            + PRERANK_WEIGHTS["is_ad_penalty"] * is_ad_penalty
        )
        seg["prerank_score"] = prerank_score
        seg["prerank_features"] = {
            "hook": hook_score,
            "arousal": arousal_score,
            "info_density": info_density,
            "duration_fit": duration_fit,
            "has_numbers": has_numbers,
            "is_ad_penalty": is_ad_penalty,
        }
    candidates.sort(key=lambda x: x["prerank_score"], reverse=True)
    n = len(candidates)
    k = min(TOP_K_MAX or n, max(TOP_K_MIN, math.ceil(TOP_K_RATIO * n)))
    
    top = candidates[:k]
    
    # Add exploration quota if enabled
    if ENABLE_EXPLORATION:
        M = max(EXPLORATION_MIN, int(EXPLORATION_QUOTA * k))
        rest = candidates[k:]
        rest_sorted = sorted(rest, key=lambda x: x.get("prerank_features", {}).get("info_density", 0.0), reverse=True)
        explore = []
        seen = {id(x) for x in top}
        for seg in rest_sorted:
            if id(seg) in seen:
                continue
            explore.append(seg)
            if len(explore) >= M:
                break
        
        picked = top + explore
        logger.info("prerank.explore", extra={"added": len(explore), "k": k})
        logger.info("Pre-rank complete: %d candidates -> keeping top %d + explore %d", n, k, len(explore))
        write_progress(episode_id, "scoring:prerank", 20, f"Pre-ranked {n} candidates, keeping top {k} + explore {len(explore)}")
        return picked
    else:
        logger.info("Pre-rank complete: %d candidates -> keeping top %d", n, k)
        write_progress(episode_id, "scoring:prerank", 20, f"Pre-ranked {n} candidates, keeping top {k}")
        return top


def get_safety_candidates(segments: List[Dict]) -> List[Dict]:
    """Get obvious banger candidates that should always be kept"""
    if not SAFETY_KEEP_ENABLED:
        return []
    safety_candidates: List[Dict] = []
    for seg in segments:
        if seg.get("is_advertisement", False):
            continue
        text = seg.get("text", "")
        duration = seg.get("duration", 0)
        words_per_sec = seg.get("words_per_sec", 0)
        is_safety = False
        if _hook_proxy(text) >= 0.8 and _arousal_proxy(text) >= 0.6:
            is_safety = True
        if "?" in text and _has_numbers(text) and _info_density_proxy(words_per_sec) >= 0.6:
            is_safety = True
        if _duration_fit(duration) >= 0.9 and _info_density_proxy(words_per_sec) >= 0.7:
            is_safety = True
        if "?" in text and len(text) > 0 and text.find("?") < len(text) * 0.3:
            is_safety = True
        if is_safety:
            safety_candidates.append(seg)
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(
                "Safety banger: '%s' (hook=%.2f, arousal=%.2f)",
                text_preview,
                _hook_proxy(text),
                _arousal_proxy(text),
            )
    logger.info("Safety net: found %d obvious bangers", len(safety_candidates))
    return safety_candidates


def pick_stratified(candidates: List[Dict], target_count: int) -> List[Dict]:
    """Pick candidates with stratification across time and duration"""
    if not STRATIFY_ENABLED or len(candidates) <= target_count:
        return candidates[:target_count]
    total_duration = max(seg.get("end", 0) for seg in candidates) if candidates else 1
    time_buckets = {"early": [], "mid": [], "late": []}
    for seg in candidates:
        start_time = seg.get("start", 0)
        if start_time < total_duration * 0.33:
            time_buckets["early"].append(seg)
        elif start_time < total_duration * 0.66:
            time_buckets["mid"].append(seg)
        else:
            time_buckets["late"].append(seg)
    duration_buckets = {"short": [], "med": [], "long": []}
    for seg in candidates:
        duration = seg.get("duration", 0)
        if duration < 30:
            duration_buckets["short"].append(seg)
        elif duration < 60:
            duration_buckets["med"].append(seg)
        else:
            duration_buckets["long"].append(seg)
    selected: List[Dict] = []
    min_per_bucket = 2
    for bucket_segs in time_buckets.values():
        if bucket_segs:
            bucket_segs.sort(key=lambda x: x["prerank_score"], reverse=True)
            selected.extend(bucket_segs[:min_per_bucket])
    for bucket_segs in duration_buckets.values():
        if bucket_segs:
            bucket_segs.sort(key=lambda x: x["prerank_score"], reverse=True)
            for seg in bucket_segs[:min_per_bucket]:
                if seg not in selected:
                    selected.append(seg)
    remaining = [seg for seg in candidates if seg not in selected]
    remaining.sort(key=lambda x: x["prerank_score"], reverse=True)
    needed = target_count - len(selected)
    selected.extend(remaining[:needed])
    if len(selected) < target_count:
        exploration_count = min(5, target_count - len(selected))
        exploration_candidates = [seg for seg in remaining if seg not in selected]
        if exploration_candidates:
            def _sid(seg):
                seg_id = seg.get("id")
                if seg_id:
                    return str(seg_id)
                a = seg.get("start"); b = seg.get("end")
                return f"{a:.3f}-{b:.3f}"
            def _key(seg):
                raw = _sid(seg).encode("utf-8")
                return hashlib.sha1(raw).hexdigest()
            exploration_candidates.sort(key=_key)
            selected.extend(exploration_candidates[:exploration_count])
    logger.info("Stratified selection: %d candidates from %d", len(selected), len(candidates))
    return selected[:target_count]
