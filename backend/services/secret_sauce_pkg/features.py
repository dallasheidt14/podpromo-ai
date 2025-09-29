"""
Feature computation module for viral detection system.
Contains feature extraction and computation functions.
"""

from typing import Dict, List, Tuple, Any, Iterable, Optional, Literal
from dataclasses import dataclass
import logging
from bisect import bisect_left
import re

# --- precision helpers ---
_UNCERTAINTY = ["maybe","might","could","kinda","sort of","somewhat","probably","possibly","perhaps","i think","i guess","i feel like"]
_CAUSAL = ["because","therefore","thus","which means","that means","as a result","here's why","the reason"]
_CONTRA_OPEN = re.compile(r"\b(most|many|people|everyone|they)\s+(think|say|assume)\b", re.I)
_CONTRA_FLIP = re.compile(r"\bbut\s+actually\b|\bhowever\b|\bbut\s+(it|this)\s+isn'?t\b", re.I)

def _best_audio_for_features(audio_path: str) -> str:
    """Prefer clean WAV if available to eliminate mpg123 errors"""
    import os
    base, ext = os.path.splitext(audio_path)
    wav_candidate = f"{base}.__asr16k.wav"
    if os.path.exists(wav_candidate):
        return wav_candidate
    else:
        return audio_path

def _uncertainty_score(txt: str) -> float:
    t = txt.lower()
    hits = sum(t.count(w) for w in _UNCERTAINTY)
    tokens = max(1, len(re.findall(r"[A-Za-z0-9']+", txt)))
    return hits/tokens

def _aha_density(txt: str, dur: float) -> float:
    if dur <= 0: return 0.0
    t = txt or ""
    causal = sum(1 for w in _CAUSAL if w in t.lower())
    ents = len(set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", txt)))
    nums = len(re.findall(r"\b\d+(\.\d+)?\b|%|percent|x\s*times", txt, re.I))
    return min(1.0, (causal + ents + nums)/max(1.0, dur))

def _ends_on_curiosity(text: str) -> bool:
    """Check if text ends on curiosity-raising patterns that create loops"""
    t = text.strip().lower()
    return any(kw in t[-80:] for kw in ["but", "however", "what you don't", "here's why", "the catch"]) \
        or t.endswith("?") or t.endswith("...")
# -----------------------------------------

FTStatus = Literal["finished", "sparse_finished", "unresolved"]

@dataclass
class FTTracker:
    fallback_mode: bool
    total: int = 0
    finished: int = 0
    sparse_finished: int = 0

    def mark(self, status: FTStatus) -> None:
        self.total += 1
        if status == "finished":
            self.finished += 1
        elif status == "sparse_finished":
            self.sparse_finished += 1

    @property
    def ratio_strict(self) -> float:
        return (self.finished / self.total) if self.total else 0.0

    @property
    def ratio_sparse_ok(self) -> float:
        num = self.finished + (self.sparse_finished if self.fallback_mode else 0)
        return (num / self.total) if self.total else 0.0
import numpy as np
import librosa
import re
import hashlib
from functools import lru_cache
from scipy import signal
from scipy.stats import skew, kurtosis
from bisect import bisect_left
from config_loader import get_config
import logging
import re
import hashlib
from typing import List, Dict, Any
log = logging.getLogger("services.secret_sauce_pkg.features")

# Terminal punctuation (., !, ?) possibly followed by a closing quote/bracket.
# Supports straight quotes and common curly quotes.
_TERMINAL_RE = re.compile(r'[.!?](?:["\'\)\]\u201D\u2019\u00BB\u203A]+)?\s*$')
_FILLER_SET = {"uh","um"}
_FILLER_PHRASES = ("you know","i mean","like","sort of","kind of","basically","so")

# When punctuation EOS is sparse, infer "natural" stops from timing gaps between segments.
_MIN_GAP_SEC = 0.30   # treat ≥300ms silence/segment gap as an EOS (lowered from 600ms)
_EOS_MIN_COUNT = 10   # below this, we synthesize from gaps
_EOS_MIN_DENSITY = 0.015
_COHERENCE_MAX_EXTEND = 5.0  # extend tails up to 5.0s to reach next pause/EOS
_FALLBACK_GRID = [14.0, 17.0, 20.0, 23.0, 26.0, 29.0]

# NEW tunables for seed length
_SEED_MIN_DUR = 16.0   # prefer ≥16s seeds when density is high
_SEED_TARGET_DUR = 18.0
_SEED_MAX_DUR = 28.0

# --- Helpers ---------------------------------------------------------------
def _extend_to_eos(start_s: float, end_s: float, eos_times: list[float] | None,
                   *, max_extra: float = 8.0, hard_cap: float = 30.0) -> float:
    """
    Extend end time to the first EOS marker after current end, within max_extra seconds,
    without exceeding hard_cap duration from start.
    """
    if not eos_times:
        return end_s
    max_end = min(start_s + hard_cap, end_s + max_extra)
    i = bisect_left(eos_times, end_s + 0.05)  # first EOS strictly after end
    if i < len(eos_times) and eos_times[i] <= max_end:
        return eos_times[i]
    return end_s

def _looks_finished(text: str) -> bool:
    """Lightweight heuristic: ends with ., !, or ? (optionally followed by a closing quote/bracket)."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_TERMINAL_RE.search(t))

def _trim_leadin_filler(seg: Dict[str, Any], *, max_trim: float = 1.2) -> float:
    """
    Trim common start-of-clip fillers (uh/um/you know/…) using word timings (<= max_trim).
    Returns the new start time, never advancing past end-0.2s.
    """
    start_s = float(seg["start"])
    end_s = float(seg.get("end", start_s + float(seg.get("dur", 8.0))))
    words: List[Dict[str, Any]] = seg.get("words") or []
    if not words:
        return start_s
    new_start = start_s
    consumed = 0.0
    for i, w in enumerate(words):
        wt = float(w.get("start", new_start))
        if wt - start_s > max_trim:
            break
        raw = (w.get("text") or "").lower().strip(".,!?\"'()")
        if raw in _FILLER_SET:
            wdur = float(w.get("dur", w.get("end", wt) - wt))
            new_start = max(new_start, wt + max(0.0, wdur))
            consumed = new_start - start_s
            continue
        # simple two-word phrases
        if i + 1 < len(words):
            two = ((w.get("text","") + " " + words[i+1].get("text","")).lower())
            if any(two.startswith(p) for p in _FILLER_PHRASES):
                nxt = words[i+1]
                nxt_t = float(nxt.get("start", new_start))
                nxt_d = float(nxt.get("dur", nxt.get("end", nxt_t) - nxt_t))
                if nxt_t + nxt_d - start_s <= max_trim:
                    new_start = max(new_start, nxt_t + max(0.0, nxt_d))
                    consumed = new_start - start_s
                    continue
        break
    # safety headroom
    if consumed >= 0.2 and new_start < end_s - 0.2:
        return new_start
    return start_s

def _hash_jitter(seed: str, scale: float = 2.0) -> float:
    """Deterministic jitter in [-scale, +scale] seconds based on a string seed."""
    h = hashlib.md5(seed.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    return (val * 2.0 - 1.0) * scale

def _candidate_ends(start_s: float, *, min_dur: float, max_dur: float, eos_times: list[float] | None) -> list[float]:
    """
    Collect EOS boundaries between [start+min_dur, start+max_dur].
    If sparse, add a few fallback grid points to avoid empty sets.
    """
    ends: list[float] = []
    lo = start_s + min_dur
    hi = start_s + max_dur
    if eos_times:
        i = bisect_left(eos_times, lo)
        while i < len(eos_times) and eos_times[i] <= hi:
            ends.append(eos_times[i])
            i += 1
    for g in _FALLBACK_GRID:
        t = start_s + g
        if lo <= t <= hi:
            ends.append(t)
    return sorted({round(e, 2) for e in ends})

def extend_to_coherent_end(end_s: float, eos: List[float], segs: Optional[List[Dict]] = None) -> float:
    """Extend tails up to _COHERENCE_MAX_EXTEND sec to the nearest EOS or segment boundary."""
    target = end_s
    # Prefer the next EOS if close enough
    ahead = [t for t in eos if end_s < t <= end_s + _COHERENCE_MAX_EXTEND]
    if ahead:
        return min(ahead)
    if not segs:
        return end_s
    # Else, snap to the next segment end if within window
    seg_ends = [float(s.get("end", 0.0) or 0.0) for s in segs if float(s.get("end", 0.0) or 0.0) >= end_s]
    seg_ends = [t for t in seg_ends if t - end_s <= _COHERENCE_MAX_EXTEND]
    if seg_ends:
        return min(seg_ends)
    return end_s

def _length_prior(dur: float) -> float:
    """Neutral inside [8, 45] seconds; smooth penalty outside"""
    import math
    lo, hi = 8.0, 45.0
    if dur < lo:
        # penalty ramps harder the further below 8s
        return -1.0 * (1.0 / (1.0 + math.exp(-(lo - dur))))  # ~0 to -1
    if dur > hi:
        return -1.0 * (1.0 / (1.0 + math.exp(-(dur - hi))))  # ~0 to -1
    return 0.0

def platform_gaussian(dur: float, pl_v2: float) -> float:
    """Platform fit gaussian - disabled for length-agnostic scoring"""
    return 0.0  # Disabled: PLATFORM_GAUSSIAN_WEIGHT = 0.0

def platform_tiebreaker(dur1: float, dur2: float, target: float = 22.0) -> float:
    """Tie-breaker only: prefer closer to platform target when virality is within ε=0.03"""
    dist1 = abs(dur1 - target)
    dist2 = abs(dur2 - target)
    return 1.0 if dist1 < dist2 else 0.0

def _detect_resolution_bonus(text: str) -> float:
    """Detect claim->reason patterns for resolution bonus"""
    if not text:
        return 0.0
    
    resolution_patterns = r'\b(because|so that|which means|so you can|therefore|as a result|that\'s why)\b'
    if re.search(resolution_patterns, text.lower()):
        return 0.08
    return 0.0

# Module-level effective weights (computed once)
EFFECTIVE_WEIGHTS = None
_logged_once = False

def _compute_effective_weights():
    """Compute effective weights once at module import"""
    global EFFECTIVE_WEIGHTS, _logged_once
    
    if EFFECTIVE_WEIGHTS is not None:
        return EFFECTIVE_WEIGHTS
    
    # Load config weights
    from config_loader import get_config
    config = get_config()
    CFG = config.get("weights", {})
    
    # Only use these weights (drop platform_len if gaussian is off)
    USED = {"hook", "arousal", "payoff", "info", "q_or_list", "loop"}
    W = {k: v for k, v in CFG.items() if k in USED}
    
    # Map q_or_list to ql for compatibility
    if "q_or_list" in W:
        W["ql"] = W["q_or_list"]
        W["q_list"] = W["q_or_list"] * 0.5  # Split q_or_list between ql and q_list
    
    # Normalize to sum to 1.0
    s = sum(W.values())
    if abs(s - 1.0) > 0.005:
        for k in W:
            W[k] /= s
        if not _logged_once:
            log.debug("Weights normalized from %.2f to 1.00", s)
            _logged_once = True
    
    # Add platform weight as 0.00
    W["platform"] = 0.00
    EFFECTIVE_WEIGHTS = W
    
    # Log once at startup
    log.info("Weights: {hook:%.2f, arousal:%.2f, payoff:%.2f, info:%.2f, ql:%.2f, q_list:%.2f, loop:%.2f, platform:0.00}",
             W["hook"], W["arousal"], W["payoff"], W["info"], W["ql"], W["q_list"], W["loop"])
    
    return EFFECTIVE_WEIGHTS

def compute_virality(seg: dict, start_s: float, end_s: float, pl_v2: float, flags: dict = None) -> float:
    """
    Single source of truth for virality calculation with rebalance support.
    Uses pre-computed effective weights (no runtime re-normalization).
    """
    dur = max(0.1, end_s - start_s)
    finished = bool(seg.get("finished_thought"))
    flags = flags or {}
    
    # Guard rails: default to 0.0 if missing
    hook = getattr(seg, "hook", 0.0) or 0.0
    arous = getattr(seg, "arous", 0.0) or 0.0
    payoff = getattr(seg, "payoff", 0.0) or 0.0
    info = getattr(seg, "info", 0.0) or 0.0
    ql = getattr(seg, "ql", 0.0) or 0.0
    q_list = getattr(seg, "q_list", 0.0) or 0.0
    loop = getattr(seg, "loop", 0.0) or 0.0
    
    # Conditional terms to prevent clickbait
    payoff_ok = bool(seg.get("payoff_ok"))
    q_list_term = 0.03*q_list if payoff_ok else -0.01*q_list
    loop_term = 0.02*loop if finished else 0.0
    
    # Virality rebalance (less hook dominance, more payoff)
    if flags.get("VIRALITY_TWEAKS_V1", False):
        w_hook = 0.40
        w_ar = 0.22
        w_pay = 0.32 if payoff_ok else 0.20
        w_info = 0.06
        # Log active weights once per run to avoid DEBUG spam
        if not hasattr(compute_virality, '_weights_logged'):
            log.info("VIRALITY_WEIGHTS: rebalanced v1 active (hook=%.2f, ar=%.2f, pay=%.2f, info=%.2f)", 
                     w_hook, w_ar, w_pay, w_info)
            compute_virality._weights_logged = True
    else:
        # Legacy behavior - use effective weights
        W = _compute_effective_weights()
        w_hook = W["hook"]
        w_ar = W["arousal"]
        w_pay = W["payoff"]
        w_info = W["info"]
        # Log active weights once per run to avoid DEBUG spam
        if not hasattr(compute_virality, '_weights_logged'):
            log.info("VIRALITY_WEIGHTS: legacy active (hook=%.2f, ar=%.2f, pay=%.2f, info=%.2f)", 
                     w_hook, w_ar, w_pay, w_info)
            compute_virality._weights_logged = True
    
    # Normalize weights to prevent "weights normalized" spam
    wsum = w_hook + w_ar + w_pay + w_info
    w_hook, w_ar, w_pay, w_info = (w_hook/wsum, w_ar/wsum, w_pay/wsum, w_info/wsum)
    
    # Get ql weight for both paths
    W = _compute_effective_weights()
    
    # Payoff-first virality with rebalanced weights
    virality = (
        w_hook * hook +
        w_ar * arous +
        w_pay * payoff +
        w_info * info +
        W["ql"] * ql +
        q_list_term +
        loop_term
    )
    
    # Demotion for weak medium/long clips
    if dur >= 16.0 and payoff < 0.30:
        virality -= 0.15 * (0.30 - payoff)
    
    # strong finish bonus, bigger if longer (because long + finished is rare & valuable)
    virality += 0.12 if dur >= 20.0 and finished else (0.08 if finished else 0.0)
    
    # ultra-short guard (don't let weak 7–10s win on hook alone)
    if dur <= 10.0 and (hook < 0.92 or payoff < 0.60):
        virality -= 0.50
    
    # soft length prior (only penalizes <8s or >45s)
    virality += 0.40 * _length_prior(dur)
    
    # Apply penalties from quality gates
    penalties = seg.get('penalties', {})
    penalty_sum = sum(penalties.values())
    virality = max(0.0, virality - penalty_sum)
    
    return virality

def compute_v_core(seg: dict, start_s: float, end_s: float) -> float:
    """
    Platform-neutral core virality for selection ranking.
    V_core = 0.26*hook + 0.20*arousal + 0.17*payoff + 0.11*info + 0.09*ql + 0.05*loop + 0.04*q_list
    NO platform terms here - used for ranking before platform recommendations.
    """
    dur = max(0.1, end_s - start_s)
    finished = bool(seg.get("finished_thought"))
    
    # Guard rails: default to 0.0 if missing
    hook = getattr(seg, "hook", 0.0) or 0.0
    arous = getattr(seg, "arous", 0.0) or 0.0
    payoff = getattr(seg, "payoff", 0.0) or 0.0
    info = getattr(seg, "info", 0.0) or 0.0
    ql = getattr(seg, "ql", 0.0) or 0.0
    q_list = getattr(seg, "q_list", 0.0) or 0.0
    loop = getattr(seg, "loop", 0.0) or 0.0
    
    # Conditional terms to prevent clickbait
    payoff_ok = bool(seg.get("payoff_ok"))
    q_list_term = 0.04*q_list if payoff_ok else -0.01*q_list
    loop_term = 0.05*loop if finished else 0.0
    
    # Core virality (platform-agnostic)
    v_core = (
        0.26 * hook +
        0.20 * arous +
        0.17 * payoff +
        0.11 * info +
        0.09 * ql +
        q_list_term +
        loop_term
    )
    
    # Demotion for weak medium/long clips
    if dur >= 16.0 and payoff < 0.30:
        v_core -= 0.15 * (0.30 - payoff)
    
    # Strong finish bonus
    v_core += 0.12 if dur >= 20.0 and finished else (0.08 if finished else 0.0)
    
    # Ultra-short guard
    if dur <= 10.0 and (hook < 0.92 or payoff < 0.60):
        v_core -= 0.50
    
    # Apply penalties from quality gates
    penalties = seg.get('penalties', {})
    penalty_sum = sum(penalties.values())
    v_core = max(0.0, v_core - penalty_sum)
    
    return v_core

def _utility_for_end(seg: dict, start_s: float, end_s: float, pl_v2: float) -> float:
    """
    Utility balances virality, platform fit, and finishing at EOS.
    Single source of truth: compute_virality() + non-content terms.
    """
    dur = max(0.1, end_s - start_s)
    
    # Get virality from compute_virality (single source of truth)
    virality = compute_virality(seg, start_s, end_s, pl_v2)
    
    # Add non-content terms
    # Platform fit gaussian (disabled for length-agnostic scoring)
    platform_term = 0.0 * platform_gaussian(dur, pl_v2)  # PLATFORM_GAUSSIAN_WEIGHT = 0.0
    
    # Utility nudge: platform coefficient based on finished status
    finished = bool(seg.get("finished_thought"))
    platform_coef = 0.4 * (1.0 if finished else 0.6)
    finish_bonus = 0.15 if finished else 0.0
    unfinished_pen = -0.12 if (not finished and dur >= 18.0) else 0.0
    
    virality += platform_term + platform_coef + finish_bonus + unfinished_pen
    
    # Resolution bonus (claim->reason patterns)
    text = seg.get("text", "") or seg.get("transcript", "")
    resolution_bonus = _detect_resolution_bonus(text)
    virality += resolution_bonus
    
    # Ad penalty (uncertain cases)
    from services.title_service import _looks_like_ad
    is_ad = _looks_like_ad(text)
    if is_ad:
        virality -= 0.60
    
    return virality

def _choose_target_duration(pl_v2: float, base: float, seg_id: str) -> float:
    """
    Platform-informed target with deterministic jitter so lengths don't collapse to one number:
      - pl_v2 >= 0.60: center ~24s
      - 0.40–0.60:     center ~20s
      - else:          center ~16–18s (respecting base)
    Jitter ±2s per clip id.
    """
    if pl_v2 >= 0.60:
        core = 24.0
    elif pl_v2 >= 0.40:
        core = 20.0
    else:
        core = max(16.0, min(18.0, base))
    return max(8.0, core + _hash_jitter(seg_id, scale=2.0))

def _neighbor_ends(start: float, eos_idx: list[float], words: list, max_seek: float = 42.0) -> list[float]:
    """Find natural boundaries (EOS or ≥600ms pause) after start"""
    ends = []
    for eos in eos_idx:
        if eos > start and eos - start <= max_seek:
            ends.append(eos)
    
    # Also check for pauses in words
    for i, w in enumerate(words):
        if i > 0:
            gap = w.get("start", 0) - words[i-1].get("end", 0)
            if gap >= 0.6:  # 600ms pause
                ends.append(w.get("start", 0))
    
    return sorted(ends)

def _choose_best_boundary(ends: list[float], start: float) -> float:
    """Choose the last EOS in the bucket (most complete thought)"""
    return max(ends) if ends else start + 8.0

def _words_to_text(words: list, start: float, end: float) -> str:
    """Extract text from words within [start, end] time range"""
    if not words:
        return ""
    parts = []
    for w in words:
        w_start = w.get("start", 0)
        w_end = w.get("end", 0)
        if w_start >= start and w_end <= end:
            parts.append(w.get("text", ""))
    return " ".join(parts).strip()

def _greedy_merge_short_seeds(seeds):
    """Merge consecutive ~8s seeds into 16–28s windows before variantization."""
    out, cur = [], None
    for s in seeds:
        if cur is None:
            cur = dict(s)
            continue
        # same speaker / contiguous window?
        if s["start"] <= cur["end"] + 0.25 and s.get("spk") == cur.get("spk"):
            # tentatively extend
            cur["end"] = s["end"]
            cur["dur"] = cur["end"] - cur["start"]
            # if we've hit our target band, flush
            if _SEED_MIN_DUR <= cur["dur"] <= _SEED_MAX_DUR:
                out.append(cur); cur = None
        else:
            # flush previous
            if cur:
                out.append(cur)
            cur = dict(s)
    if cur: out.append(cur)
    return out

# Duration bands for variantization
_SHORT = (10.0, 14.0)   # still allow short bangers
_MID   = (16.0, 22.0)
_LONG  = (24.0, 90.0)   # extend to 90s for long-form content

def _apply_micro_jitter(end_time: float, words: list, start: float) -> float:
    """Apply ±0.25s micro-jitter to break identical lengths, but preserve finished_thought"""
    if not words:
        return end_time
    
    # Find micro-pauses within ±0.25s
    jitter_candidates = []
    for w in words:
        w_start = w.get("start", 0)
        w_end = w.get("end", 0)
        if w_start >= end_time - 0.25 and w_end <= end_time + 0.25:
            jitter_candidates.append(w_end)
    
    if not jitter_candidates:
        return end_time
    
    # Choose the closest micro-pause that doesn't change finished_thought
    best_jitter = end_time
    for candidate in jitter_candidates:
        if abs(candidate - end_time) <= 0.25:
            # Check if this preserves finished_thought
            text = _words_to_text(words, start, candidate)
            if _looks_finished(text):
                best_jitter = candidate
                break
    
    return best_jitter

def _variantize_segment(seg, *, pl_v2: float, cap_s: float, eos_times: list[float] | None = None) -> dict:
    """
    De-biased variantizer: enumerate real lengths at targets, no platform scoring.
    Only enumerate EOS-clean ends at specific targets.
    """
    start = seg.get("start", 0.0)
    words = seg.get("words", [])
    
    # Enumerate EOS-clean finishes at targets from LENGTH_SEARCH_BUCKETS
    targets = LENGTH_SEARCH_BUCKETS
    variants = []
    
    for target in targets:
        # Find EOS-clean end at this target (skip non-existent ends)
        cand_ends = _neighbor_ends(start, eos_times or [], words, max_seek=target+3.5)
        
        # Find closest end to target that's EOS-clean
        best_end = None
        for e in cand_ends:
            if abs((e - start) - target) <= 2.0:  # Within 2s of target
                best_end = e
                break
        
        if best_end is None:
            continue  # Skip impossible ends
            
        # Extend to coherent end
        e = extend_to_coherent_end(best_end, eos_times or [], words, max_extend=3.5)
        
        # Apply micro-jitter to break identical lengths
        e = _apply_micro_jitter(e, words, start)
        
        text = _words_to_text(words, start, e)
        if not _looks_finished(text):   # <- hard requirement here
            continue
            
        variants.append({
            "start": start, "end": e, "dur": e - start,
            "text": text, "finished_thought": True,
            "target": target
        })
    
    # Log variant counts by target (DEBUG level to reduce noise)
    if variants:
        target_counts = {}
        for v in variants:
            t = v.get("target", 0)
            target_counts[t] = target_counts.get(t, 0) + 1
        log.debug("VAR_COUNTS: %s", target_counts)
    
    # Return the best variant by utility (length-agnostic)
    if not variants:
        # Fallback to original logic if no natural boundaries
        return {**seg, "start": start, "end": start + 8.0, "dur": 8.0, "finished_thought": False}
    
    best_variant = max(variants, key=lambda v: _utility_for_end(seg, v["start"], v["end"], pl_v2))
    return best_variant

def build_variants(segments, pl_v2, cap_s, eos_times=None, *args, **kwargs):
    out = []
    for seg in segments:
        v = _variantize_segment(seg, pl_v2=pl_v2, cap_s=cap_s, eos_times=eos_times)
        out.append(v)
    return out

# Length Search Configuration
LENGTH_SEARCH_BUCKETS = [8, 12, 18, 23, 30, 35, 45, 60, 75, 90]
LENGTH_MAX_HARD = 90.0
PLATFORM_NEUTRAL_SELECTION = True
FINISHED_THOUGHT_REQUIRED = True
SALVAGE_MAX_EXTEND_S = 8.0
SALVAGE_MODE = "punct_or_boundary_to_EOS"

# Unfinished ending policy configuration
UNFINISHED_HARD_DROP = False   # keep as a safety kill-switch
UNFINISHED_MALUS = 0.06        # subtract from virality (≈ 6 pts on 0–100 scale)
UNFINISHED_CONF_HARD = 0.20    # only hard-drop if no terminal punct *and* conf below this (lowered from 0.30)

# Soft ending policy configuration
END_GAP_MS_OK = 300            # accept if last silence >= 300 ms (lowered from 400ms)
END_CONF_OK = 0.80             # accept if last-word (or finish) confidence >= 0.80 (lowered from 0.85)
SOFT_ENDING_MALUS = 0.02       # tiny malus for punctuation-less "good" endings

# Gate calibration configuration
STRICT_FLOOR_DELTA = -0.04     # Lower strict floor by ~4 points
BALANCED_EXTRA_DELTA = -0.02   # If strict is empty, lower balanced by an additional 2 points
MIN_FINALS = 4                 # Raise from 3 → 4 to reduce "empty" episodes

# Finish-Thought Gate Configuration
FINISH_THOUGHT_CONFIG = {
    "eos_pause_ms": 260,
    "near_eos_tol_sec": {
        "normal": 0.30,
        "sparse": 0.50  # More forgiving in fallback mode
    },
    "max_extend_sec": {
        "safety_pass": 1.8,    # Wider on safety pass
        "shorts": 1.2,
        "reels": 1.2,
        "tiktok": 1.2,
        "youtube": 1.5,
        "default": 1.2,
        "sparse_bonus": 0.3  # Allow +0.3s more in fallback
    },
    "min_viable_after_shrink_sec": {
        "normal": 7.5,
        "safety_pass": 6.0,     # Allow deeper shrink to avoid drops
        "sparse": 6.5  # More lenient in sparse mode
    }
}

# Import new Phase 1, 2 & 3 types and utilities
from .types import Features, Scores, FEATURE_TYPES, SYNERGY_MODE, PLATFORM_LEN_V, WHITEN_PATHS, GENRE_BLEND, BOUNDARY_HYSTERESIS, PROSODY_AROUSAL, PAYOFF_GUARD, CALIBRATION_V, MIN_WORDS, MAX_WORDS, MIN_SEC, MAX_SEC, _keep
from .scoring_utils import whiten_paths, synergy_bonus, platform_length_score_v2, apply_genre_blending, find_optimal_boundaries, prosody_arousal_score, payoff_guard, apply_calibration, pick_boundary, snap_to_nearest_pause_or_punct

logger = logging.getLogger(__name__)

# Import dependencies from other modules
from .scoring import get_clip_weights
from .genres import GenreAwareScorer

def compute_features_v4(segment: Dict, audio_file: str, y_sr=None, genre: str = 'general', platform: str = 'tiktok', cfg=None) -> Dict:
    """Enhanced feature computation with genre awareness"""
    text = segment.get("text", "")
    
    # CRITICAL: Check for ads FIRST, before any feature computation
    ad_result = _ad_penalty(text, cfg=cfg)
    
    if ad_result["flag"]:
        # Return a clip that will be filtered out entirely
        return {
            "is_advertisement": True,
            "ad_reason": ad_result["reason"],
            "final_score": 0.0,  # Force to bottom
            "viral_score_100": 0,
            "should_exclude": True,
            "text": text,
            "duration": segment["end"] - segment["start"],
            "hook_score": 0.0,
            "arousal_score": 0.0,
            "emotion_score": 0.0,
            "question_score": 0.0,
            "payoff_score": 0.0,
            "info_density": 0.0,
            "loopability": 0.0,
            "_ad_flag": ad_result["flag"],
            "_ad_penalty": ad_result["penalty"],
            "_ad_reason": ad_result["reason"]
        }
    
    # Only compute full features for non-ad content
    duration = segment["end"] - segment["start"]
    
    word_count = len(text.split()) if text else 0
    words_per_sec = word_count / max(duration, 0.1)
    
    # Hook scoring with V5 implementation
    config = get_config()
    use_v5 = bool(config.get("hook_v5", {}).get("enabled", True))
    
    if use_v5:
        # Hook V5 scoring
        seg_idx = segment.get("index", 0)
        # We'll set these after features are computed
        h_raw, h_cal, h_dbg = _hook_score_v5(
            text,
            cfg=config,
            segment_index=seg_idx,
            audio_modifier=0.0,  # Will be updated after audio analysis
            arousal=0.0,  # Will be updated after arousal computation
            q_or_list=0.0,  # Will be updated after question detection
        )
        hook_score = float(h_cal)
        hook_reasons = ",".join(h_dbg.get("reasons", []))
        hook_details = h_dbg
    else:
        # Legacy V4 hook scoring
        hook_score, hook_reasons, hook_details = _hook_score_v4(text, segment.get("arousal_score", 0.0), words_per_sec, genre, 
                                                               segment.get("audio_data"), segment.get("sr"), segment.get("start", 0.0))
    # Payoff V2 with feature flag and fallback
    from config.settings import ENABLE_PAYOFF_V2
    
    # Initialize payoff variables
    payoff_score = 0.0
    payoff_type = "none"
    payoff_label = "none"
    payoff_span = None
    payoff_src = "v1"
    
    if ENABLE_PAYOFF_V2:
        score_v2, label_v2, span_v2 = _detect_payoff_v2(text, genre)
        payoff_score = score_v2
        payoff_type = label_v2  # keeps old key consumers happy
        payoff_label = label_v2  # new, human-readable
        payoff_span = span_v2
        payoff_src = "v2"
    else:
        score_v1, type_v1 = _detect_payoff(text, genre)
        payoff_score = score_v1
        payoff_type = type_v1
        payoff_label = type_v1
        payoff_span = None
        payoff_src = "v1"
    
    # NEW: Detect insight content vs. intro/filler (V2 if enabled)
    if config.get("insight_v2", {}).get("enabled", False):
        insight_score, insight_reasons = _detect_insight_content_v2(text, genre)
        # Apply confidence multiplier if available
        confidence = segment.get("confidence", None)
        if confidence is not None:
            insight_score = _apply_insight_confidence_multiplier(insight_score, confidence)
    else:
        insight_score, insight_reasons = _detect_insight_content(text, genre)
    
    niche_penalty, niche_reason = _calculate_niche_penalty(text, genre)
    
    # ENHANCED AUDIO ANALYSIS: Compute actual audio arousal with intelligent fallback
    audio_arousal = _audio_prosody_score(audio_file, segment["start"], segment["end"], text=text, genre=genre)
    
    # ENHANCED TEXT AROUSAL: Genre-aware text arousal scoring
    text_arousal = _arousal_score_text(text, genre)
    
    # ADAPTIVE COMBINATION: Adjust audio/text ratio based on genre and content type
    if genre == 'fantasy_sports':
        # Sports content benefits more from text analysis (stats, names, etc.)
        combined_arousal = 0.6 * audio_arousal + 0.4 * text_arousal
    elif genre == 'comedy':
        # Comedy benefits more from audio (timing, delivery)
        combined_arousal = 0.8 * audio_arousal + 0.2 * text_arousal
    elif genre == 'true_crime':
        # True crime benefits from both (dramatic delivery + intense content)
        combined_arousal = 0.7 * audio_arousal + 0.3 * text_arousal
    else:
        # Default balanced approach
        combined_arousal = 0.7 * audio_arousal + 0.3 * text_arousal
    
    # Base features
    feats = {
        "is_advertisement": False,  # Explicitly mark as non-ad
        "should_exclude": False,    # Explicitly mark as includable
        "hook_score": hook_score,
        "arousal_score": combined_arousal,
        "emotion_score": _emotion_score_v4(text),
        "question_score": _question_or_list(text),
        "payoff_score": payoff_score,
        "info_density": _info_density_v4(text),  # Will be updated by V2 system if enabled
        "loopability": _loopability_heuristic(text),
        "insight_score": insight_score,  # NEW: Insight content detection (may be adjusted by confidence multiplier)
        "text": text,
        "duration": duration,
        "words_per_sec": words_per_sec,
        "hook_reasons": hook_reasons,
        "payoff_type": payoff_type,
        "insight_reasons": insight_reasons,  # NEW: Insight detection reasons
        "text_arousal": text_arousal,
        "audio_arousal": audio_arousal,
        "platform_len_match": calculate_dynamic_length_score(segment, platform) if "boundary_type" in segment else _platform_length_match(duration, platform),
        "_ad_flag": ad_result["flag"],
        "_ad_penalty": ad_result["penalty"],
        "_ad_reason": ad_result["reason"],
        "ad_likelihood": ad_result["penalty"],  # Use penalty as likelihood score
        "_niche_penalty": niche_penalty,
        "_niche_reason": niche_reason,
        "type": segment.get("type", "general"),  # Preserve moment type for bonuses
        
        # ENHANCED: Multi-dimensional hook details
        "hook_components": hook_details.get("hook_components", {}),
        "hook_type": hook_details.get("hook_type", "general"),
        "hook_confidence": hook_details.get("confidence", 0.0),
        "audio_modifier": hook_details.get("audio_modifier", 0.0),
        "laughter_boost": hook_details.get("laughter_boost", 0.0),
        "time_weighted_score": hook_details.get("time_weighted_score", 0.0)
    }
    
    # Apply genre-specific enhancements if genre is specified
    if genre != 'general':
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre, genre_scorer.genres['general'])
        
        # Add genre-specific features
        genre_features = genre_profile.detect_genre_patterns(text)
        feats.update(genre_features)
        
        # Adjust features based on genre
        feats = genre_profile.adjust_features(feats)
    
    # ensure downstream names exist for synergy calculations
    feats["arousal"] = feats.get("arousal", feats.get("arousal_score", 0.0))
    feats["q_or_list"] = feats.get("q_or_list", feats.get("question_score", 0.0))
    
    # RECOMPUTE Hook V5 with real signals (arousal, question_score, audio_modifier)
    if use_v5:
        h_raw, h_cal, h_dbg = _hook_score_v5(
            text,
            cfg=config,
            segment_index=seg_idx,
            audio_modifier=feats.get("audio_modifier", 0.0),
            arousal=float(feats.get("arousal_score", 0.0)),
            q_or_list=float(feats.get("question_score", 0.0)),
        )
        feats["hook_score"] = float(h_cal)
        feats.setdefault("_debug", {})
        feats["_debug"]["hook_v5"] = h_dbg
    
    # Apply hook clamping for ad-like content and repetition
    hook = feats.get("hook_score", 0.0)
    ad_like = feats.get("ad_likelihood", 0.0)
    
    # Clamp hook for ad-like content
    if ad_like >= _cfg(cfg, "HOOK_AD_CLAMP_MIN"):
        hook = min(hook, _cfg(cfg, "HOOK_AD_CLAMP"))
    
    # Penalize repetition at the open
    opener = text[:_cfg(cfg, "REP_WINDOW_CHARS")]
    if _repetition_ratio(opener) >= _cfg(cfg, "HOOK_REP_MIN_RATIO"):
        hook *= _cfg(cfg, "HOOK_REP_PENALTY_MULT")
    
    feats["hook_score"] = hook
    
    # Add missing fields that enhanced version returns (set to zeros for compatibility)
    feats.update({
        'insight_conf': 0.0,
        'q_list_score': 0.0,
        'prosody_arousal': 0.0,
        'platform_length_score_v2': 0.0,
    })
    
    # Add payoff V2 fields for compatibility and analytics
    feats['payoff_label'] = payoff_label
    feats['payoff_span'] = payoff_span
    feats['payoff_src'] = payoff_src
    
    return feats

def compute_features(segment: Dict, audio_file: str, y_sr=None, version: str = "v4", genre: str = 'general', platform: str = 'tiktok') -> Dict:
    if version == "v4":
        return compute_features_v4(segment, audio_file, y_sr, genre, platform)
    else:
        text = segment.get("text","")
        ad_result = _ad_penalty(text)
        
        feats = {
            "hook_score": _hook_score(text),
            "arousal_score": _audio_prosody_score(audio_file, segment["start"], segment["end"], y_sr=y_sr),
            "emotion_score": _emotion_score(text),
            "question_score": _question_or_list(text),
            "payoff_score": _payoff_presence(text),
            "info_density": _info_density(text),
            "loopability": _loopability_heuristic(text),
            "_ad_flag": ad_result["flag"],
            "_ad_penalty": ad_result["penalty"],
            "_ad_reason": ad_result["reason"]
        }
        
        if ad_result["flag"]:
            feats["payoff_score"] = 0.0
            feats["info_density"] = min(feats.get("info_density", 0.0), 0.35)
        return feats

def compute_features_cached(segment_hash: str, audio_file: str, genre: str, platform: str = 'tiktok') -> Dict:
    """Cached version of compute_features_v4 for performance"""
    # This is a placeholder - in practice, you'd need to reconstruct the segment
    # from the hash or use a different caching strategy
    return compute_features_v4({"text": "", "start": 0, "end": 0}, audio_file, genre=genre, platform=platform)

def create_segment_hash(segment: Dict) -> str:
    """Create a hash from segment content for cache key"""
    content = f"{segment.get('text', '')[:100]}_{segment.get('start', 0)}_{segment.get('end', 0)}"
    return hashlib.md5(content.encode()).hexdigest()

def _hook_score(text: str) -> float:
    score, _, _ = _hook_score_v4(text)
    return score

def _emotion_score(text: str) -> float:
    return _emotion_score_v4(text)

def _payoff_presence(text: str) -> float:
    score, _ = _payoff_presence_v4(text)
    return score

def _info_density(text: str) -> float:
    return _info_density_v4(text)

def _question_or_list(text: str) -> float:
    t = text.strip().lower()
    
    greeting_questions = ["what's up", "how's it going", "you like that", "how are you"]
    if any(greeting in t[:30] for greeting in greeting_questions):
        return 0.1
    
    if "?" in t:
        question_text = t.split("?")[0][-50:]
        if len(question_text.split()) < 3:
            return 0.2
        
        engaging_patterns = [r"what if", r"why do", r"how did", r"what happens when"]
        if any(re.search(pattern, question_text) for pattern in engaging_patterns):
            return 1.0
        
        return 0.6
    
    return 0.0

def _loopability_heuristic(text: str) -> float:
    """Enhanced loopability scoring with perfect-loop detection, quotability patterns, and curiosity enders"""
    if not text: 
        return 0.0
    
    t = text.lower()
    score = 0.0
    
    # Perfect loop detection - ends where it begins
    words = t.split()
    if len(words) >= 3:
        first_phrase = " ".join(words[:3])
        last_phrase = " ".join(words[-3:])
        if first_phrase == last_phrase:
            score += 0.4
    
    # Quotability patterns
    quotable_patterns = [
        r"here's the thing",
        r"the truth is",
        r"what i learned",
        r"the key is",
        r"here's why",
        r"the secret",
        r"the trick"
    ]
    
    for pattern in quotable_patterns:
        if re.search(pattern, t):
            score += 0.2
            break
    
    # Curiosity enders - questions or incomplete thoughts
    if t.endswith('?') or t.endswith('...') or t.endswith('but'):
        score += 0.15
    
    # End-on-curiosity bonus - check final 1-2s for curiosity-raising patterns
    if _ends_on_curiosity(text):
        score += 0.15
    
    # Short, punchy statements
    if len(words) <= 8 and any(word in t for word in ['insane', 'crazy', 'wild', 'epic', 'amazing']):
        score += 0.1
    
    return float(np.clip(score, 0.0, 1.0))

def score_variant(variant: dict, audio_file: str, genre: str, platform: str) -> dict:
    """Re-score a variant with new boundaries"""
    try:
        # Re-compute features for the variant with new start/end
        features = compute_features_v4_enhanced(variant, audio_file, genre=genre, platform=platform)
        variant.update(features)
        return variant
    except Exception as e:
        logger.warning(f"Failed to score variant: {e}")
        return variant

# Cliffhanger detection patterns
CLIFF_END_RE = re.compile(
    r"(what if|because|so that|that's why|here's why|the reason|which means|"
    r"and then|but then|until|unless|so|but|and)\s*$",
    re.IGNORECASE
)

# End-boundary detection patterns
END_CUE_RE = re.compile(
    r"(i'll never forget|when i|because|but then|and then|that'?s when|"
    r"which|so that|until|unless|but|and)\s*$", re.IGNORECASE
)

# Memory/narrative continuation cues
MEMORY_CUE_RE = re.compile(r"(i'?ll never forget|the moment|that'?s when)\b", re.IGNORECASE)

def looks_like_cliffhanger(text_end: str) -> bool:
    """Detect if text ends with a cliffhanger or unfinished thought"""
    if not text_end: 
        return False
    t = text_end.strip()
    if t.endswith("..."): 
        return True
    if CLIFF_END_RE.search(t): 
        return True
    # Interrogative with no follow-up (ends with '?')
    return t.endswith("?")

def is_unfinished_tail(text: str) -> bool:
    """Detect if text ends without proper sentence completion"""
    t = (text or "").strip()
    if not t: 
        return False
    if t.endswith(("...", "…")): 
        return True
    if END_CUE_RE.search(t[-160:]): 
        return True
    # No sentence-ending punctuation
    return not any(t.endswith(p) for p in (".", "!", "?", ".", "!", "?"))

def wants_continuation(text: str) -> bool:
    """Detect if text has memory/narrative cues that want continuation"""
    tail = (text or "").strip()[-200:].lower()
    return MEMORY_CUE_RE.search(tail) is not None

def extend_to_next_boundary(segment, max_extra_s, segments_list):
    """Try to extend segment to next safe boundary"""
    if not segments_list:
        return segment
    
    cur_end = segment.get('end', 0)
    hard_cap = cur_end + max_extra_s
    
    # Find current segment index
    current_idx = -1
    for i, seg in enumerate(segments_list):
        if abs(seg.get('start', 0) - segment.get('start', 0)) < 0.1:
            current_idx = i
            break
    
    if current_idx == -1:
        return segment
    
    # Look for next safe boundary
    j = current_idx
    while j + 1 < len(segments_list) and segments_list[j + 1].get('end', 0) <= hard_cap:
        nxt = segments_list[j + 1]
        txt = nxt.get('text', '').strip()
        pause = max(0.0, nxt.get('start', 0) - segments_list[j].get('end', 0))
        
        # Boundary rules: sentence punctuation, pause, or speaker change
        if (txt.endswith((".", "!", "?")) or 
            pause >= 0.4 or 
            nxt.get('speaker') != segments_list[j].get('speaker')):
            # Found safe boundary - extend to this segment
            extended = segment.copy()
            extended['end'] = nxt.get('end', cur_end)
            extended['text'] = ' '.join([seg.get('text', '') for seg in segments_list[current_idx:j+2]])
            return extended
        j += 1
    
    return segment  # No better boundary found

def apply_cliffhanger_guard(c):
    """Apply cliffhanger guard to penalize unfinished thoughts and prefer resolving variants"""
    text = c.get('text', '').strip()
    if not text: 
        return

    if looks_like_cliffhanger(text[-120:]):  # last ~120 chars
        dur = c.get('end', 0) - c.get('start', 0)
        payoff = c.get('payoff_score', 0.0)
        pl_v2 = c.get('platform_length_score_v2', 0.0)

        # If short & low-payoff cliffhanger → penalize unless it's long enough or value is present
        if dur < 16.0 and payoff < 0.30:
            c['final_score'] = min(c['final_score'], 0.60)  # soft cap
            c.setdefault('flags', {}).setdefault('caps_applied', []).append('cliff_end_penalty')

        # If there exists a longer sibling (your variant set) with payoff improvement,
        # mark this one as disfavored so the selector picks the longer.
        c.setdefault('flags', {}).setdefault('caps_applied', []).append('prefer_longer_variant_for_resolution')

def promote_to_resolution(seed_variants, current_best):
    """Promote hook-without-payoff clips to find resolving siblings"""
    hook = current_best.get('hook_score', 0.0)
    payoff = current_best.get('payoff_score', 0.0)

    if hook >= 0.90 and payoff < 0.30:
        resolving = [v for v in seed_variants if v.get('payoff_score', 0.0) >= 0.35]
        if resolving:
            # choose shortest that hits payoff threshold and decent pl_v2
            resolving.sort(key=lambda v: (v.get('end', 0) - v.get('start', 0),
                                          -v.get('platform_length_score_v2', 0.0)))
            chosen = resolving[0]
            chosen.setdefault('flags', {}).setdefault('caps_applied', []).append('promoted_to_resolution')
            return chosen
    return current_best

def end_completeness_adjust(v):
    """Apply end-completeness bonus and unfinished penalty"""
    text = v.get("text", "").strip()
    if not text:
        return
    
    ends_clean = text.endswith((".", "!", "?"))
    payoff = v.get("payoff_score", 0.0)
    dur = v.get("end", 0) - v.get("start", 0)
    
    # Micro-bonus for complete sentences (helps 13-16s picks)
    if ends_clean and payoff >= 0.25 and dur >= 16.0:
        v["final_score"] += 0.05  # increased from 0.03 for better tie-breaking
        v.setdefault('flags', {}).setdefault('caps_applied', []).append('end_completeness_bonus')
    elif not ends_clean and dur <= 12.0:
        v["final_score"] -= 0.02  # penalty for short unfinished clips
        v.setdefault('flags', {}).setdefault('caps_applied', []).append('unfinished_penalty')

def finalize_variant_text(v, segments_list):
    """Finalize variant text by extending to proper sentence boundaries"""
    if is_unfinished_tail(v.get("text", "")):
        # Try to extend to next boundary
        extended = extend_to_next_boundary(v, max_extra_s=9.0, segments_list=segments_list)
        if extended != v and extended.get('end', 0) > v.get('end', 0):
            # Re-score the extended variant
            from .scoring import score_segment_v4
            extended_score = score_segment_v4(extended)
            extended['final_score'] = extended_score.get('final_score', 0)
            extended['payoff_score'] = extended_score.get('payoff_score', 0)
            
            # Accept if score drop small and payoff improves
            if (extended.get('final_score', 0) >= v.get('final_score', 0) - 0.02 and 
                extended.get('payoff_score', 0) >= v.get('payoff_score', 0) + 0.05):
                extended.setdefault('flags', {}).setdefault('caps_applied', []).append('end_extended_to_sentence')
                return extended
            else:
                v.setdefault('flags', {}).setdefault('caps_applied', []).append('kept_unfinished_tail_due_to_score_drop')
    
    # Check for memory continuation cues
    if (wants_continuation(v.get("text", "")) and 
        v.get("payoff_score", 0.0) < 0.35 and 
        v.get('end', 0) - v.get('start', 0) < 26.0):
        extended = extend_to_next_boundary(v, max_extra_s=9.0, segments_list=segments_list)
        if extended != v and extended.get('end', 0) > v.get('end', 0):
            from .scoring import score_segment_v4
            extended_score = score_segment_v4(extended)
            extended['final_score'] = extended_score.get('final_score', 0)
            extended['payoff_score'] = extended_score.get('payoff_score', 0)
            extended.setdefault('flags', {}).setdefault('caps_applied', []).append('memory_continuation_extension')
            return extended
    
    # Apply small penalty for unfinished tails
    if is_unfinished_tail(v.get("text", "")):
        v['final_score'] = max(0, v.get('final_score', 0) - 0.03)
        v.setdefault('flags', {}).setdefault('caps_applied', []).append('unfinished_tail_penalty')
    
    return v

def choose_variant(base: dict, variants: list, audio_file: str, genre: str, platform: str, eos_times: list = None, word_end_times: list = None, fallback_mode: bool = None, ft: FTTracker = None) -> dict:
    """Choose the best variant by re-scoring and comparing with payoff-aware selection"""
    
    # Platform-neutral constraints (true dynamic discovery)
    MIN_DUR = 7.5  # Hard floor for any short clip
    MAX_DUR = LENGTH_MAX_HARD  # 90.0s hard cap for all platforms
    ALLOWED_DURS = LENGTH_SEARCH_BUCKETS  # Use full search range
    
    # Always have a seed label
    seed = (base.get("id") or base.get("seed_id") or 
            base.get("debug_id") or f"seg_{base.get('index', '?')}")
    
    # Apply duration constraints to all variants
    def enforce_duration_constraints(variant):
        start = variant.get('start', 0)
        end = variant.get('end', 0)
        
        # 1) Ensure end >= start
        end = max(end, start)
        variant['end'] = end
        dur = end - start
        
        # 2) Guard against zero/negative duration
        if dur < (MIN_DUR - 0.05):
            # Try to extend to next EOS within platform budget
            if eos_times:
                next_eos = None
                for eos_time in eos_times:
                    if eos_time > end and eos_time <= end + 10.0:  # Within 10s
                        next_eos = eos_time
                        break
                if next_eos:
                    end = min(next_eos, start + MAX_DUR)
                    variant['end'] = end
                    dur = end - start
        
        # 3) Final floor check
        if dur < (MIN_DUR - 0.05):
            logger.warning(f"FINAL_VARIANT_FLOOR_BREACH: {seed} dur={dur:.2f}s")
            return None  # Skip this variant; do NOT return zero-length
        
        # 4) Clamp to maximum duration
        if dur > MAX_DUR:
            logger.warning(f"VARIANT_CAP_BREACH: {seed} dur={dur:.2f}s - clamping to {MAX_DUR}s")
            variant['end'] = start + MAX_DUR
            dur = MAX_DUR
        
        # 5) Snap to nearest allowed duration if close
        if ALLOWED_DURS:
            closest_allowed = min(ALLOWED_DURS, key=lambda x: abs(x - dur))
            if abs(closest_allowed - dur) <= 1.0:  # Within 1 second
                variant['end'] = start + closest_allowed
                dur = closest_allowed
        
        return variant
    
    # Apply constraints to base and all variants
    base = enforce_duration_constraints(base.copy())
    if base is None:
        logger.error(f"FINAL_VARIANT_FLOOR_BREACH: {seed} base variant failed duration check")
        return None
    
    variants = [enforce_duration_constraints(v.copy()) for v in variants]
    variants = [v for v in variants if v is not None]  # Filter out failed variants
    
    # Guard against zero-length variants after any EOS snapping
    MIN_DUR = 1.0  # sec
    def valid(v): 
        return (v.get("end", 0) - v.get("start", 0)) >= MIN_DUR
    
    variants = [v for v in variants if valid(v)]
    if not variants:
        logger.warning(f"NO_VALID_VARIANTS_AFTER_EOS: {seed} all variants too short, using base")
        return base
    
    # Re-score every variant (compute features again with new start/end!)
    scored = [score_variant(v, audio_file, genre, platform) for v in [base] + variants]
    
    # Payoff-aware variant selection
    def pick_best_variant(variants):
        def vscore(v):
            if PLATFORM_NEUTRAL_SELECTION:
                # Platform-neutral scoring using V_core
                start_s = v.get('start', 0)
                end_s = v.get('end', 0)
                return compute_v_core(v, start_s, end_s)
            else:
                # Legacy platform-aware scoring
                payoff = v.get('payoff_score', 0.0)
                pl_v2 = v.get('platform_length_score_v2', v.get('platform_len_match', 0.0))
                return pl_v2 + 0.20 * min(payoff, 0.5)
        
        # Neutral variant policy: prefer longer/EOS, only shorten when clean closure exists
        def choose_variant_neutral(base_len, variants):
            # Sort by V_core score first
            scored_variants = [(v, vscore(v)) for v in variants]
            scored_variants.sort(key=lambda x: -x[1])
            
            # Prefer variants that end at EOS
            eos_variants = [v for v, score in scored_variants if v.get('finished_thought', False)]
            if eos_variants:
                chosen = eos_variants[0]
                logger.debug(f"VARIANT_DECISION: base={base_len:.1f}, chosen={chosen.get('end', 0) - chosen.get('start', 0):.1f}, reason=EOS")
                return chosen
            
            # Prefer longer variants (round up to next bucket)
            base_dur = base_len
            longer_variants = [v for v, score in scored_variants 
                             if (v.get('end', 0) - v.get('start', 0)) >= base_dur]
            if longer_variants:
                chosen = longer_variants[0]
                logger.debug(f"VARIANT_DECISION: base={base_len:.1f}, chosen={chosen.get('end', 0) - chosen.get('start', 0):.1f}, reason=UP_BUCKET")
                return chosen
            
            # Only shorten if there's a natural sentence end and higher payoff
            shorter_variants = [v for v, score in scored_variants 
                              if (v.get('end', 0) - v.get('start', 0)) < base_dur and 
                                 v.get('finished_thought', False) and
                                 v.get('payoff_score', 0) > base.get('payoff_score', 0) + 0.1]
            if shorter_variants:
                chosen = shorter_variants[0]
                logger.debug(f"VARIANT_DECISION: base={base_len:.1f}, chosen={chosen.get('end', 0) - chosen.get('start', 0):.1f}, reason=SHORT_EOS")
                return chosen
            
            # Fallback to highest scoring
            chosen = scored_variants[0][0]
            logger.debug(f"VARIANT_DECISION: base={base_len:.1f}, chosen={chosen.get('end', 0) - chosen.get('start', 0):.1f}, reason=FALLBACK")
            return chosen
        
        if PLATFORM_NEUTRAL_SELECTION:
            base_len = base.get('end', 0) - base.get('start', 0)
            return choose_variant_neutral(base_len, variants)
        else:
            # Legacy behavior
            return sorted(variants, key=lambda v: -vscore(v))[0]
    
    # Apply cliffhanger guard to all variants
    for v in scored:
        apply_cliffhanger_guard(v)
    
    # Apply end-completeness adjustment to all variants
    for v in scored:
        end_completeness_adjust(v)
    
    # NEW: Apply Finish-Thought Gate to all variants if EOS data is available
    if eos_times and word_end_times:
        normalized_scored = []
        for v in scored:
            # Use the effective EOS data for finish-thought normalization
            normalized_v, result = finish_thought_normalize(v, eos_times, word_end_times, platform, fallback_mode)
            
            # Propagate ft_status from the result
            normalized_v["ft_status"] = result.get("status")  # 'finished'|'sparse_finished'|'unresolved'
            normalized_v["ft_meta"] = result  # keep extra fields if you have them
            
            normalized_scored.append(normalized_v)
            
            # Track finish status if FTTracker provided
            if ft is not None:
                ft.mark(result.get("status", "unresolved"))
            
            if result.get("status") != "finished":
                logger.debug(f"FINISH_THOUGHT: {v.get('id', 'unknown')} -> {result.get('status', 'unknown')}")
        scored = normalized_scored
    
    # Pick best variant using payoff-aware selection
    best = pick_best_variant(scored)
    
    # Apply promotion to resolution if needed
    best = promote_to_resolution(scored, best)
    
    # C) Sibling promotion - prefer longer finished siblings when short is unresolved
    if fallback_mode is None:
        fallback_mode = len(eos_times) < 50 or len(word_end_times) < 500 if eos_times else True
    
    # Check if current best has unfinished tail penalty
    if best.get('caps') and 'unfinished_tail_penalty' in best.get('caps', []):
        promoted = pick_finished_sibling(scored, platform, MAX_DUR)
        if promoted:
            best = promoted
            logger.info(f"SIBLING_PROMOTED: {seed} -> longer finished sibling (unfinished tail)")
    
    # Also promote in fallback mode
    if fallback_mode:
        promoted = pick_finished_sibling(scored, platform, MAX_DUR)
        if promoted and promoted != best:
            best = promoted
            logger.info(f"SIBLING_PROMOTED: {seed} -> longer finished sibling (fallback)")
    
    # Finalize variant text (extend to sentence boundaries)
    # Note: We need the original segments list for boundary extension
    # For now, we'll apply the text finalization without the segments list
    best = finalize_variant_text(best, [])
    
    # Final safety check
    final_dur = best.get('end', 0) - best.get('start', 0)
    if final_dur < MIN_DUR:
        logger.error(f"FINAL_VARIANT_FLOOR_BREACH: {best.get('id', 'unknown')} dur={final_dur:.2f}s")
        best['end'] = best.get('start', 0) + MIN_DUR
    
    return best

def grow_to_bins(segment: dict, audio_file: str, genre: str, platform: str, eos_times: list = None, word_end_times: list = None, fallback_mode: bool = None, ft: FTTracker = None) -> dict:
    """Generate length variants and choose the best one with flexible extension"""
    TARGETS = [12.0, 18.0, 24.0, 30.0]
    MIN_SEC, MAX_SEC = 8.0, 60.0
    MAX_EXTEND = 10.0  # Allow up to 10s extension per hop
    
    base_dur = segment.get('end', 0) - segment.get('start', 0)
    base_pl_v2 = segment.get('platform_length_score_v2', 0.0)
    
    # Find nearest target
    nearest_target = min(TARGETS, key=lambda x: abs(x - base_dur))
    
    variants = []
    for target in TARGETS:
        if target != nearest_target and MIN_SEC <= target <= MAX_SEC:
            # Allow ±1.0s jitter and flexible extension
            for jitter in [-1.0, 0.0, 1.0]:
                adjusted_target = max(MIN_SEC, min(MAX_SEC, target + jitter))
                if abs(adjusted_target - base_dur) <= MAX_EXTEND:
                    variant = segment.copy()
                    variant['end'] = segment['start'] + adjusted_target
                    variants.append(variant)
    
    if not variants:
        return segment
    
    # Choose best variant
    chosen = choose_variant(segment, variants, audio_file, genre, platform, eos_times, word_end_times, fallback_mode, ft)
    chosen_dur = chosen.get('end', 0) - chosen.get('start', 0)
    chosen_pl_v2 = chosen.get('platform_length_score_v2', 0.0)
    
    # VAR: Log variant selection
    logger.info("VAR: seed=%s base=%.1fs -> chosen=%.1fs drop=%.3f pl_v2 %.2f→%.2f",
        segment.get('_id', 'unknown'), base_dur, chosen_dur, 
        segment.get('final_score', 0) - chosen.get('final_score', 0),
        base_pl_v2, chosen_pl_v2
    )
    
    return chosen

# payoff_guard is imported from scoring_utils

# Ad detection patterns and helpers
AD_PHRASES = {
    "brought to you by", "sponsored by", "use code", "limited time",
    "shop now", "link in bio", "link in description", "free shipping",
    "visit", "learn more", "order today", "act now", "save", "discount"
}

DOMAIN_RE = re.compile(r'\b(?:https?://|www\.)?[a-z0-9-]+\.(?:com|net|org|io|co)\b', re.I)
SPELLOUT_RE = re.compile(
    r'\b(?:[a-z](?:\s*-\s*|\.?\s*)){3,}[a-z]\s*(?:dot|\.)\s*(?:com|net|org|io|co)\b', re.I
)

CTA_TOKENS = {"shop", "buy", "save", "order", "visit", "subscribe", "download", "sign", "sign up"}

# -----------------------
# Hook scoring constants
# -----------------------
# Micro re-trim: scan first N tokens for a stronger start and give a tiny bonus
HOOK_MICRO_RETRIM_MAX_TOKENS = 6          # how far we're allowed to "slide" the start
HOOK_MICRO_RETRIM_STEP_BONUS = 0.008      # per-token step bonus (j * step)
HOOK_MICRO_RETRIM_MAX_BONUS = 0.015       # cap so we don't distort ordering

# Hedge softening: if a hedge ("uh", "you know…") is immediately followed by strong content,
# reduce its penalty so good cold-opens aren't nuked.
HOOK_HEDGE_LOOKAHEAD_TOKENS = 5           # look ahead after hedge in first 3 tokens
HOOK_HEDGE_SOFTEN_FACTOR = 0.50           # multiply the hedge penalty by this if softened

# First clause window: give the first substantive clause a chance to register
HOOK_FIRST_CLAUSE_WINDOW = 20             # was typically ~15 in older configs

# Words considered weak/filler at start; keep short and conservative
HOOK_START_FILLERS = {
    "uh", "um", "er", "ah", "like", "you", "know", "so", "well", "okay", "ok",
}

# Tokens that hint at a strong, actionable opener (helps the micro re-trim)
HOOK_STRONG_STARTERS = {
    "what", "why", "how", "here's", "listen", "the", "truth", "nobody", "you",
    "let", "here", "this", "stop", "remember",
}

# Safe defaults for when config is not passed
_DEFAULTS = dict(
    ALLOW_ADS=0,
    AD_SIG_MIN=0.50,
    HOOK_AD_CLAMP_MIN=0.40,
    HOOK_AD_CLAMP=0.30,
    REP_WINDOW_CHARS=200,
    HOOK_REP_MIN_RATIO=0.30,
    HOOK_REP_PENALTY_MULT=0.70,
    Q_ONLY_MIN_Q=0.50,
    Q_ONLY_MAX_PAYOFF=0.15,
    LOG_AD_SIGNALS=1,
)

def _cfg(cfg, key):
    """Get config value with safe defaults"""
    return getattr(cfg, key, _DEFAULTS[key]) if cfg else _DEFAULTS[key]

def _ad_likelihood(text: str, cfg=None) -> float:
    """Comprehensive ad likelihood scoring"""
    t = text.lower()
    score = 0.0
    reasons = []
    
    # URLs / domains
    if DOMAIN_RE.search(t):
        score += 0.45
        reasons.append("domain")
    if SPELLOUT_RE.search(t):  # catches "wayfair dot com", "w-a-y-f-a-i-r dot com"
        score += 0.45
        reasons.append("spellout")
    
    # Sponsor phrases / CTAs
    if any(p in t for p in AD_PHRASES):
        score += 0.25
        reasons.append("ad_phrases")
    
    cta_hits = sum(tok in t for tok in CTA_TOKENS)
    if cta_hits >= 2:
        score += 0.15
        reasons.append("cta")
    
    # Brand repetition (proper-nounish token repeated 2–3x in short span)
    properish = re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', text)
    if properish:
        from collections import Counter
        rep = max(Counter(properish).values())
        if rep >= 3:
            score += 0.20
            reasons.append("repetition")
        elif rep == 2:
            score += 0.10
            reasons.append("repetition")
    
    # Log ad signals if enabled
    if reasons and _cfg(cfg, "LOG_AD_SIGNALS"):
        logging.getLogger(__name__).debug(f"AD_SCAN: like={score:.2f} reasons={reasons}")
    
    return min(1.0, score)

def _repetition_ratio(snippet: str) -> float:
    """Calculate repetition ratio in text snippet"""
    toks = re.findall(r"[a-zA-Z']+", snippet.lower())
    if not toks: return 0.0
    uniq = len(set(toks))
    return 1.0 - (uniq / max(1, len(toks)))

def _ad_penalty(text: str, cfg=None) -> dict:
    """Enhanced ad detection with comprehensive patterns"""
    ad_like = _ad_likelihood(text, cfg=cfg)
    is_ad = ad_like >= _cfg(cfg, "AD_SIG_MIN")
    
    return {
        "flag": is_ad,
        "penalty": ad_like,
        "reason": "ad_likelihood" if is_ad else "no_promotion"
    }

def _platform_length_match(duration: float, platform: str = 'tiktok') -> float:
    """Calculate how well the duration matches platform preferences"""
    platform_ranges = {
        'tiktok': {'optimal': (15, 30), 'acceptable': (5, 45), 'minimal': (3, 60)},
        'instagram': {'optimal': (15, 30), 'acceptable': (5, 45), 'minimal': (3, 60)},
        'instagram_reels': {'optimal': (15, 30), 'acceptable': (5, 45), 'minimal': (3, 60)},
        'youtube_shorts': {'optimal': (20, 45), 'acceptable': (5, 60), 'minimal': (3, 90)},
        'linkedin': {'optimal': (30, 60), 'acceptable': (10, 90), 'minimal': (5, 120)}
    }
    
    ranges = platform_ranges.get(platform, platform_ranges['tiktok'])
    
    if ranges['optimal'][0] <= duration <= ranges['optimal'][1]:
        return 1.0  # Perfect match
    elif ranges['acceptable'][0] <= duration <= ranges['acceptable'][1]:
        # Linear falloff from optimal edge
        if duration < ranges['optimal'][0]:
            return 0.5 + 0.5 * (duration - ranges['acceptable'][0]) / (ranges['optimal'][0] - ranges['acceptable'][0])
        else:
            return 0.5 + 0.5 * (ranges['acceptable'][1] - duration) / (ranges['acceptable'][1] - ranges['optimal'][1])
    elif ranges['minimal'][0] <= duration <= ranges['minimal'][1]:
        # Very short clips get partial credit
        if duration < ranges['acceptable'][0]:
            return 0.2 + 0.3 * (duration - ranges['minimal'][0]) / (ranges['acceptable'][0] - ranges['minimal'][0])
        else:
            return 0.2 + 0.3 * (ranges['minimal'][1] - duration) / (ranges['minimal'][1] - ranges['acceptable'][1])
    else:
        return 0.0  # Outside all ranges

def calculate_dynamic_length_score(segment: Dict, platform: str) -> float:
    """
    Calculate length score for dynamic segments, considering natural boundaries.
    """
    # Check if Platform Length V2 is enabled
    config = get_config()
    plat_cfg = config.get("platform_length_v2", {})
    
    if not plat_cfg.get("enabled", True):
        # V1 path - legacy implementation
        duration = segment.get("end", 0) - segment.get("start", 0)
        base_score = _platform_length_match(duration, platform)
        
        # Bonus for natural boundaries
        boundary_type = segment.get("boundary_type", "")
        confidence = segment.get("confidence", 0.0)
        
        if boundary_type in ["sentence_end", "insight_marker"] and confidence > 0.8:
            base_score += 0.1  # Bonus for clean boundaries
        
        return min(1.0, base_score)
    
    # V2 path - enhanced implementation
    duration = (segment.get("end", 0.0) - segment.get("start", 0.0)) or 0.0
    
    # Calculate WPS with fallbacks
    wps = None
    if segment.get("word_count"):
        wps = float(segment["word_count"]) / max(duration, 1e-6)
    elif segment.get("text") and duration > 0:
        # Fallback: compute word count from text
        word_count = len(segment["text"].split())
        wps = float(word_count) / max(duration, 1e-6)
    
    # Extract text tail for outro detection (last 1-2 seconds)
    text_tail = segment.get("tail_text", "")
    if not text_tail and segment.get("text") and duration > 0:
        # Simple fallback: take last few words if no tail_text provided
        words = segment["text"].split()
        if len(words) > 3:
            text_tail = " ".join(words[-3:])  # Last 3 words as approximation
    
    # Get other segment data with defaults
    loopability = segment.get("loopability", 0.0)
    boundary_type = segment.get("boundary_type", "")
    boundary_conf = float(segment.get("confidence", 0.0) or 0.0)
    
    # Use V2 scoring
    score = _platform_length_score_v2(
        duration=duration,
        platform=platform,
        loopability=loopability,
        wps=wps,
        boundary_type=boundary_type,
        boundary_conf=boundary_conf,
        text_tail=text_tail,
    )
    
    # Add debug information
    if "_debug" not in segment:
        segment["_debug"] = {}
    
    debug = segment["_debug"]
    debug["length_v2_duration"] = duration
    debug["length_v2_wps"] = wps
    debug["length_v2_loopability"] = loopability
    debug["length_v2_boundary_type"] = boundary_type
    debug["length_v2_boundary_conf"] = boundary_conf
    debug["length_v2_text_tail_present"] = bool(text_tail)
    debug["length_v2_final_score"] = score
    
    return score

def _audio_prosody_score(audio_path: str, start: float, end: float, y_sr=None, text: str = "", genre: str = 'general') -> float:
    """Enhanced audio analysis for arousal/energy detection with intelligent fallback"""
    try:
        if y_sr is None:
            # Prefer clean WAV if available to eliminate mpg123 errors
            audio_for_features = _best_audio_for_features(audio_path)
            y, sr = librosa.load(audio_for_features, sr=None, offset=max(0, start-0.2), duration=(end-start+0.4))
        else:
            y, sr = y_sr
            s = max(int((start-0.2)*sr), 0)
            e = min(int((end+0.4)*sr), len(y))
            y = y[s:e]
        
        if len(y) == 0:
            # Fallback to text-based estimation
            return _text_based_audio_estimation(text, genre)
        
        # Compute audio features
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate arousal score based on audio features
        rms_mean = np.mean(rms)
        spectral_mean = np.mean(spectral_centroids)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Normalize and combine features
        arousal_score = (rms_mean * 0.4 + spectral_mean * 0.3 + zcr_mean * 0.3)
        arousal_score = float(np.clip(arousal_score, 0.0, 1.0))
        
        return arousal_score
        
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}, falling back to text-based estimation")
        return _text_based_audio_estimation(text, genre)

def _arousal_score_text(text: str, genre: str = 'general') -> float:
    """Enhanced text arousal scoring with genre awareness and intensity levels"""
    if not text:
        return 0.0
    
    t = text.lower()
    score = 0.0
    
    # Enhanced exclamation detection with intensity
    exclam_count = text.count('!')
    if exclam_count > 0:
        # More exclamations = higher intensity
        if exclam_count >= 3:
            score += 0.4  # High intensity
        elif exclam_count == 2:
            score += 0.25  # Medium intensity
        else:
            score += 0.15  # Low intensity
    
    # Enhanced caps detection (include short impactful words)
    caps_words = sum(1 for word in text.split() if word.isupper() and len(word) >= 2)
    if caps_words > 0:
        score += min(caps_words * 0.12, 0.25)  # Slightly higher weight
    
    # Enhanced emotion words with intensity levels
    high_intensity_words = ["insane", "shocking", "unbelievable", "mind-blowing", "incredible", "crazy", "wild", "epic", "amazing"]
    medium_intensity_words = ["awesome", "great", "fantastic", "wonderful", "exciting", "thrilling", "intense", "powerful"]
    low_intensity_words = ["good", "nice", "cool", "interesting", "fun", "enjoyable"]
    
    high_hits = sum(1 for word in high_intensity_words if word in t)
    medium_hits = sum(1 for word in medium_intensity_words if word in t)
    low_hits = sum(1 for word in low_intensity_words if word in t)
    
    # Weighted scoring by intensity
    score += min(high_hits * 0.15, 0.4)      # High intensity words
    score += min(medium_hits * 0.08, 0.2)    # Medium intensity words
    score += min(low_hits * 0.03, 0.1)       # Low intensity words
    
    # Genre-specific arousal patterns
    if genre == 'fantasy_sports':
        sports_intensity_words = ["fire", "draft", "start", "bench", "target", "sleeper", "bust", "league-winner"]
        sports_hits = sum(1 for word in sports_intensity_words if word in t)
        score += min(sports_hits * 0.1, 0.2)
    elif genre == 'comedy':
        comedy_intensity_words = ["hilarious", "funny", "lol", "haha", "rofl", "joke", "punchline"]
        comedy_hits = sum(1 for word in comedy_intensity_words if word in t)
        score += min(comedy_hits * 0.12, 0.25)
    elif genre == 'true_crime':
        crime_intensity_words = ["murder", "killer", "victim", "evidence", "mystery", "suspicious", "alibi"]
        crime_hits = sum(1 for word in crime_intensity_words if word in t)
        score += min(crime_hits * 0.1, 0.2)
    
    # Question marks add engagement (arousal)
    question_count = text.count('?')
    if question_count > 0:
        score += min(question_count * 0.05, 0.15)
    
    # Urgency indicators
    urgency_words = ["now", "immediately", "urgent", "critical", "important", "must", "need", "quickly"]
    urgency_hits = sum(1 for word in urgency_words if word in t)
    score += min(urgency_hits * 0.08, 0.2)
    
    return float(np.clip(score, 0.0, 1.0))

def _detect_payoff(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect payoff strength: resolution, answer, or value delivery using genre-specific patterns"""
    if not text or len(text.strip()) < 8:
        return 0.0, "too_short"
    
    t = text.lower()
    score = 0.0
    reasons = []
    
    # Guard: require uncertainty to drop from tease→resolution
    # Split 40/30/30 across the text as a proxy for time.
    words = text.split()
    n = len(words)
    if n >= 12:
        a = " ".join(words[:int(n*0.40)])
        b = " ".join(words[int(n*0.40):int(n*0.70)])
        uncertainty_before = _uncertainty_score(a)
        uncertainty_after = _uncertainty_score(b)
        d = uncertainty_before - uncertainty_after
        THRESH = 0.02  # was 0.0; require actual uncertainty reduction
        if d <= THRESH:
            # disable payoff unless other strong markers redeem it
            score -= 0.2  # pushes final clip toward penalties if no resolution
            reasons.append(f"unfinished_tail_penalty(unc_before={uncertainty_before:.3f},unc_after={uncertainty_after:.3f},delta={d:.3f})")
    
    # Resolution patterns
    resolution_patterns = [
        r"(so|therefore|thus|as a result|consequently)",
        r"(the answer is|the solution is|here's how)",
        r"(that's why|which explains|this means)",
        r"(in conclusion|to sum up|the bottom line)"
    ]
    
    for pattern in resolution_patterns:
        if re.search(pattern, t):
            score += 0.2
            reasons.append("resolution")
            break
    
    # Value delivery patterns
    value_patterns = [
        r"(here's what|the key|the secret|the trick)",
        r"(you should|you need to|you must|you have to)",
        r"(this will|this can|this helps|this makes)",
        r"(the benefit|the advantage|the upside)"
    ]
    
    for pattern in value_patterns:
        if re.search(pattern, t):
            score += 0.15
            reasons.append("value_delivery")
            break
    
    # Genre-specific payoff patterns
    if genre == 'fantasy_sports':
        sports_patterns = [
            r"(start|bench|target|avoid|sleeper|bust)",
            r"(this week|next week|playoffs|season)",
            r"(league winner|championship|title)"
        ]
        for pattern in sports_patterns:
            if re.search(pattern, t):
                score += 0.1
                reasons.append("sports_payoff")
                break
    
    # Penalty for questions without answers
    if "?" in t and not any(word in t for word in ["answer", "solution", "here's", "the key"]):
        score -= 0.1
        reasons.append("question_without_answer")
    
    final_score = float(np.clip(score, 0.0, 1.0))
    reason_str = ";".join(reasons) if reasons else "no_payoff"
    return final_score, reason_str

# --- Payoff V2: rhetorical/QA/numeric resolution with tail bias ---
_PAYOFF_LEADS = [
    r"\bthe (?:bottom|top) line\b",
    r"\btl;?dr\b",
    r"\bhere(?:'|')?s (?:the|what)\b",
    r"\bso (?:the|what this) (?:means|is)\b",
    r"\bnet[-\s]?net\b",
    r"\bkey takeaway\b",
    r"\bin (?:short|summary)\b",
    r"\bthe answer is\b",
    r"\bthe play (?:here|now) is\b",
    r"\bmy advice\b", r"\bpro tip\b",
    r"\bdo this\b", r"\byou should\b", r"\bthe trick is\b",
    r"\bhere(?:'|')?s how\b", r"\bstep(?:s)? to\b",
    r"\bstop doing\b", r"\bstart doing\b",
    r"\btherefore\b", r"\bbecause\b",
]

# CTA-ish endings we should *not* reward as payoff
_PAYOFF_NEG = [
    r"\bsubscribe\b", r"\bfollow\b", r"\bsmash (?:that )?like\b",
    r"\bcheck (?:the )?link\b", r"\bjoin (?:my|our)\b",
]

_NUMERIC_HINT = r"(?:\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+%|\$\d+(?:\.\d+)?)"

# Precompile regexes at module import for performance
_PAYOFF_LEADS_RE = [re.compile(p, re.I) for p in _PAYOFF_LEADS]
_PAYOFF_NEG_RE = [re.compile(p, re.I) for p in _PAYOFF_NEG]
_NUMERIC_HINT_RE = re.compile(_NUMERIC_HINT)

def _detect_payoff_v2(text: str, genre: str = "general") -> tuple[float, str, Optional[tuple[int, int]]]:
    """
    Returns: (score[0..1], label: str, span: (start,end) or None)
    Heuristics:
      - reward: explicit payoff leads, numeric commitment, QA resolution
      - tail bias: payoff in final 35% of the segment is stronger
      - penalties: open-ended teasers, pure CTAs
    """
    import logging
    from config.settings import PAYOFF_TAIL_BIAS, PAYOFF_V2_DEBUG
    
    if not text or len(text) < 24:
        return 0.0, "none", None

    # Cap text length for analysis (first/last 1,800 chars)
    if len(text) > 1800:
        text = text[:900] + " ... " + text[-900:]

    t = " ".join(text.split())
    n = len(t)
    tail_start = int(PAYOFF_TAIL_BIAS * n)  # weight payoff that appears near the end

    # Sentence split (cheap)
    sents = re.split(r"(?<=[\.\!\?])\s+", t)
    sent_spans = []
    off = 0
    for s in sents:
        s2 = s.strip()
        if not s2:
            continue
        i0 = t.find(s2, off)
        if i0 < 0:
            continue
        i1 = i0 + len(s2)
        sent_spans.append((s2, i0, i1))
        off = i1

    score = 0.0
    label = "none"
    span = None
    lead_hit_count = 0
    numeric_hit_count = 0
    qa_detected = False
    tail_boost_applied = False

    # 1) payoff leads
    for pat in _PAYOFF_LEADS_RE:
        cnt = 0
        for m in pat.finditer(t):
            w = 0.25
            if m.start() >= tail_start:
                w += 0.15
                tail_boost_applied = True
            score += w
            lead_hit_count += 1
            label = "lead"
            span = (m.start(), min(n, m.end()))
            cnt += 1
            if cnt >= 5:  # limit matches per pattern
                break

    # 2) numeric commitments
    cnt = 0
    for m in _NUMERIC_HINT_RE.finditer(t):
        w = 0.12
        # slight extra weight if in tail
        if m.start() >= tail_start:
            w += 0.08
            tail_boost_applied = True
        score += w
        numeric_hit_count += 1
        if label == "none":
            label = "numeric"
            span = (m.start(), min(n, m.end()))
        cnt += 1
        if cnt >= 5:  # limit matches
            break

    # 3) QA resolution: question followed by declarative answer
    qpos = t.find("?")
    if qpos != -1 and qpos < n - 5:
        answer_tail = t[qpos+1:].strip()
        # non-trivial answer & not another question
        if len(answer_tail) > 15 and "?" not in answer_tail:
            w = 0.22
            # tail bias if most of the answer sits in the tail
            if qpos >= int(0.4 * n):
                w += 0.10
                tail_boost_applied = True
            score += w
            qa_detected = True
            if label == "none":
                label = "qa"
                # approximate span: from qpos to end or sentence end
                span = (qpos+1, min(n, qpos + 1 + len(answer_tail)))

    # 4) last-sentence reinforcement
    if sent_spans:
        last_text, i0, i1 = sent_spans[-1]
        # if last sentence looks conclusive
        if re.search(r"\b(so|hence|therefore|that means|in short)\b", last_text, re.I):
            score += 0.15
            if label == "none":
                label = "conclusion"
                span = (i0, i1)

    # 5) penalize pure CTAs masquerading as payoff
    for pat in _PAYOFF_NEG_RE:
        if pat.search(t):
            score -= 0.20

    # light cap/boosts
    score = max(0.0, min(1.0, score))
    # normalize gently toward 0..0.75 typical range
    if score > 0.75:
        score = 0.75 + 0.25 * (score - 0.75)
    
    # Debug logging
    if PAYOFF_V2_DEBUG:
        logger = logging.getLogger(__name__)
        logger.debug("payoff.v2.matches", extra={
            "lead_hits": lead_hit_count,
            "numeric_hits": numeric_hit_count,
            "qa": qa_detected,
            "tail_bias": tail_boost_applied,
            "score": score
        })
    
    return float(score), label, span

def _detect_insight_content(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect if content contains actual insights vs. intro/filler material"""
    if not text or len(text.strip()) < 10:
        return 0.0, "too_short"
    
    t = text.lower()
    insight_score = 0.0
    reasons = []
    
    # Fantasy sports insight patterns
    if genre in ['fantasy_sports', 'sports']:
        insight_patterns = [
            r"(observation|insight|noticed|realized|discovered)",
            r"(main|key|important|significant) (takeaway|point|finding)",
            r"(casual|serious|experienced) (drafters|players|managers)",
            r"(way better|much better|improved|evolved)",
            r"(under my belt|experience|seen|witnessed)",
            r"(home league|draft|waiver|roster)",
            r"(sleeper|bust|value|target|avoid)",
            r"(this week|next week|season|playoffs)"
        ]
        
        for pattern in insight_patterns:
            if re.search(pattern, t):
                insight_score += 0.2
                reasons.append("fantasy_insight")
        
        # Boost for specific insights
        if re.search(r"(casual drafters are way better)", t):
            insight_score += 0.3
            reasons.append("specific_insight_boost")
    
    # General insight patterns
    general_insight_patterns = [
        r"(here's what|the thing is|what i found|what i learned)",
        r"(the key|the secret|the trick|the strategy)",
        r"(most people|everyone|nobody) (thinks|believes|knows)",
        r"(contrary to|despite|although|even though)",
        r"(the truth is|reality is|actually|in fact)"
    ]
    
    for pattern in general_insight_patterns:
        if re.search(pattern, t):
            insight_score += 0.15
            reasons.append("general_insight")
    
    # Penalty for filler content
    filler_patterns = [
        r"^(yo|hey|hi|hello|what's up)",
        r"^(it's|this is) (monday|tuesday|wednesday)",
        r"^(hope you|hope everyone)",
        r"^(let's get|let's start|let's begin)"
    ]
    
    for pattern in filler_patterns:
        if re.match(pattern, t):
            insight_score -= 0.3
            reasons.append("filler_penalty")
            break
    
    final_score = float(np.clip(insight_score, 0.0, 1.0))
    reason_str = ";".join(reasons) if reasons else "no_insights"
    return final_score, reason_str

def _calculate_niche_penalty(text: str, genre: str = 'general') -> tuple[float, str]:
    t = text.lower()
    penalty = 0.0
    reasons = []
    
    # Skip penalties entirely for sports genres
    if genre in ['sports', 'fantasy_sports']:
        return 0.0, "sports_genre_no_penalty"
    
    # Apply penalties for other genres
    context_patterns = [r"\b(like that|that's|this is)\b"]
    for pattern in context_patterns:
        if re.search(pattern, t):
            penalty += 0.10
            reasons.append("context_dependent")
            break
    
    final_penalty = float(np.clip(penalty, 0.0, 0.5))
    reason_str = ";".join(reasons) if reasons else "no_niche_penalty"
    return final_penalty, reason_str


# Real implementations from the original sophisticated scoring system
import math

# Regex patterns for Hook V5
_WORD = re.compile(r"[A-Za-z']+|\d+%?")
_PUNCT_CLAUSE = re.compile(r"(?<=[.!?])\s+")
_HAS_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b")
_HAS_COMP = re.compile(r"\b(?:vs\.?|versus|more than|less than)\b|[<>]")
_HAS_HOWWHY = re.compile(r"\b(?:how|why|what)\b")
_SOFT_FLIP = re.compile(r"\bbut (?:actually|in reality)\b")

# Question/List scoring patterns
_QMARK = re.compile(r"\?\s*$")
_INTERROG = re.compile(r"\b(what|why|how|when|where|which|who)\b", re.I)
_COMPARE = re.compile(r"\b(vs\.?|versus|better than|worse than|compare(?:d)? to)\b", re.I)
_CHOICE = re.compile(r"\b(which|pick|choose|would you rather|either|or)\b", re.I)
_RHET_IND = re.compile(r"\b(you know|right|isn't it|don't you think|agree)\b", re.I)
_CLIFF_Q = re.compile(r"\b(what if|imagine|suppose|what would happen|what do you think)\b", re.I)
_GENUINE = re.compile(r"\b(genuinely|honestly|really|truly|actually)\b", re.I)
_LIST_MARKERS = re.compile(r"\b(first|second|third|fourth|fifth|last|finally|next|then|also|additionally|moreover|furthermore)\b", re.I)
_LIST_ITEMS = re.compile(r"\d+\.\s+[^.!?]+[.!?]?")
_BAIT_PATTERNS = re.compile(r"\b(you won't believe|shocking|amazing|incredible|unbelievable|mind-blowing)\b", re.I)
_VACUOUS_Q = re.compile(r"\b(what do you think|what's your opinion|agree\?|right\?|you know\?)\b", re.I)

def _tokens_ql(s: str):
    """Tokenize for question/list scoring"""
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def _sentences(text: str):
    """Split text into sentences"""
    return re.split(r"[.!?]+\s+", text.strip())

def _sigmoid_ql(x: float, a: float) -> float:
    """Sigmoid function for question/list scoring"""
    return 1.0 / (1.0 + np.exp(-a * x))

def _saturating_sum(scores: List[float], cap: float = 1.0) -> float:
    """Saturating sum for combining multiple scores"""
    prod = 1.0
    for s in scores:
        s = max(0.0, min(1.0, float(s)))
        prod *= (1.0 - s)
    return min(cap, 1.0 - prod)

def _proximity_bonus(index_in_video: int, k: float) -> float:
    """Proximity bonus for early segments"""
    try:
        i = max(0, int(index_in_video))
    except Exception:
        i = 0
    return math.exp(- i / max(1e-6, float(k)))

def _normalize_quotes_lower(text: str) -> str:
    """Normalize quotes and convert to lowercase"""
    t = (text or "").strip().lower()
    return t.translate({
        0x2019: 0x27,  # ' -> '
        0x2018: 0x27,  # ' -> '
        0x201C: 0x22,  # " -> "
        0x201D: 0x22,  # " -> "
    })

def _first_clause(text: str, max_words: int) -> str:
    """Extract first clause up to max_words"""
    sent = _PUNCT_CLAUSE.split(text, maxsplit=1)[0]
    toks = _WORD.findall(sent)
    return " ".join(toks[:max_words])

def _get_hook_cues_from_config(cfg: Dict[str, Any]) -> Dict[str, List[re.Pattern]]:
    """Extract hook cues from configuration"""
    raw = (cfg.get("HOOK_CUES")
           or cfg.get("lexicons", {}).get("HOOK_CUES")
           or {})
    # If someone left HOOK_CUES as a flat list, wrap it.
    if isinstance(raw, list):
        raw = {"generic": raw}
    cues: Dict[str, List[re.Pattern]] = {}
    for fam, arr in raw.items():
        pats = []
        for s in arr:
            try:
                pats.append(re.compile(s, re.I))
            except Exception:
                pass
        if pats:
            cues[fam] = pats
    return cues

def _family_score(text: str, cues: Dict[str, List[re.Pattern]], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Score text against hook cue families"""
    fam_scores: Dict[str, float] = {}
    partials: List[float] = []
    for fam, pats in cues.items():
        w = float(weights.get(fam, 1.0))
        m = 0.0
        for p in pats:
            if p.search(text):
                m = 1.0
                break
        fam_scores[fam] = min(1.0, m * w)
        if fam_scores[fam] > 0:
            partials.append(min(1.0, fam_scores[fam]))
    combined = _saturating_sum(partials, cap=1.0)
    return combined, fam_scores

def _evidence_guard(t: str, need_words: int, clause_words: int) -> Tuple[bool, Dict[str, bool]]:
    """Check for evidence in early text or first clause"""
    toks = _WORD.findall(t)
    early = " ".join(toks[:max(0, need_words)])
    has_A = bool(_HAS_HOWWHY.search(early) or _HAS_NUM.search(early) or _HAS_COMP.search(early))
    if has_A:
        return True, {"early": True, "clause": False, "flip": False}
    clause = _first_clause(t, max_words=clause_words)
    has_B = bool(_HAS_HOWWHY.search(clause) or _HAS_NUM.search(clause) or _HAS_COMP.search(clause))
    flip = bool(_SOFT_FLIP.search(clause))
    return bool(has_B or flip), {"early": False, "clause": has_B, "flip": flip}

def _anti_intro_outro_penalties(t: str, hv5: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    """Apply penalties for intro/outro content"""
    reasons = []
    pin = 0.0
    pout = 0.0
    intro = [s.strip().lower() for s in hv5.get("intro_tokens", [])]
    outro = [s.strip().lower() for s in hv5.get("outro_tokens", [])]
    for tok in intro:
        if tok and t.startswith(tok):
            pin = float(hv5.get("anti_intro_penalty", 0.05)); reasons.append("anti_intro")
            break
    for tok in outro:
        if tok and tok in t:
            pout = float(hv5.get("anti_outro_penalty", 0.06)); reasons.append("anti_outro")
            break
    return pin, pout, reasons

def _audio_micro_for_hook(audio_mod: float, cap: float) -> float:
    """Apply audio modifier with cap"""
    try:
        return max(0.0, min(cap, float(audio_mod)))
    except Exception:
        return 0.0

def _sigmoid(z: float, a: float) -> float:
    """Sigmoid function for calibration"""
    return 1.0 / (1.0 + math.exp(-a * z))

def _calibrate_simple(raw: float, mu: float = 0.40, sigma: float = 0.18, a: float = 1.6) -> float:
    """Simple calibration using sigmoid"""
    z = 0.0 if sigma <= 0 else (raw - mu) / sigma
    return _sigmoid(z, a)

def _hook_score_v4(text: str, arousal: float = 0.0, words_per_sec: float = 0.0, genre: str = 'general', 
                   audio_data=None, sr=None, start_time: float = 0.0) -> Tuple[float, str, Dict]:
    """Real Hook V4 implementation"""
    # Enhanced hook scoring with multiple indicators
    hook_indicators = [
        'you', 'imagine', 'what if', 'did you know', 'here\'s why', 'the secret', 
        'shocking', 'amazing', 'incredible', 'unbelievable', 'mind-blowing',
        'here\'s what', 'the truth is', 'the key', 'the trick', 'the secret',
        'what happens when', 'why do', 'how did', 'here\'s how'
    ]
    hook_score = 0.0
    reasons = []
    
    text_lower = text.lower()
    
    # Pattern interrupt: "most people think ... but actually ..."
    if _CONTRA_OPEN.search(text_lower) and _CONTRA_FLIP.search(text_lower):
        hook_score += 0.15
        reasons.append("contradiction_flip")
    
    # Check for hook indicators
    for indicator in hook_indicators:
        if indicator in text_lower:
            hook_score += 0.15
            reasons.append(f"hook_indicator_{indicator.replace(' ', '_')}")
    
    # Boost for questions
    if '?' in text:
        hook_score += 0.25
        reasons.append("question_mark")
    
    # Boost for short, punchy text
    word_count = len(text.split())
    if word_count < 8:
        hook_score += 0.2
        reasons.append("very_short_punchy")
    elif word_count < 15:
        hook_score += 0.1
        reasons.append("short_punchy")
    
    # Boost for exclamations
    exclam_count = text.count('!')
    if exclam_count > 0:
        hook_score += min(exclam_count * 0.1, 0.3)
        reasons.append(f"exclamation_{exclam_count}")
    
    # Boost for numbers/statistics
    if re.search(r'\d+', text):
        hook_score += 0.1
        reasons.append("contains_numbers")
    
    # Boost for comparison words
    comparison_words = ['vs', 'versus', 'more than', 'less than', 'better than', 'worse than']
    if any(word in text_lower for word in comparison_words):
        hook_score += 0.1
        reasons.append("comparison")
    
    hook_score = min(hook_score, 1.0)
    hook_details = {
        "hook_components": {"word_count": word_count, "exclam_count": exclam_count},
        "hook_type": "general",
        "confidence": hook_score,
        "audio_modifier": 0.0,
        "laughter_boost": 0.0,
        "time_weighted_score": hook_score
    }
    
    return hook_score, ",".join(reasons), hook_details

def _hook_score_v5(text: str, cfg: Dict = None, segment_index: int = 0, audio_modifier: float = 0.0,
                   arousal: float = 0.0, q_or_list: float = 0.0) -> Tuple[float, float, Dict]:
    """Real Hook V5 implementation - the sophisticated scoring system"""
    hv5 = cfg.get("hook_v5", {}) if cfg else {}
    a_sig = float(hv5.get("sigmoid_a", 1.6))
    need_words = int(hv5.get("require_after_words", 12))
    clause_words = int(hv5.get("first_clause_max_words", 20))
    print(f"HOOK: clause_window={clause_words}")
    k = float(hv5.get("time_decay_k", 5))
    early_bonus_scale = float(hv5.get("early_pos_bonus", 0.25))
    audio_cap = float(hv5.get("audio_cap", 0.05))
    fam_w = hv5.get("family_weights", {}) or {}

    t = _normalize_quotes_lower(text)
    cues = _get_hook_cues_from_config(cfg)

    fam_combined, fam_scores = _family_score(t, cues, fam_w)
    evidence_ok, evidence_bits = _evidence_guard(t, need_words, clause_words)

    base = fam_combined
    reasons: List[str] = []
    if fam_combined <= 0.0:
        reasons.append("no_family_match")
    if not evidence_ok and fam_combined > 0.0:
        base *= 0.80
        reasons.append("no_evidence_early")

    pin, pout, pr = _anti_intro_outro_penalties(t, hv5)
    base = max(0.0, base - pin - pout)
    reasons.extend(pr)

    prox = _proximity_bonus(segment_index, k)
    base += early_bonus_scale * prox

    base += _audio_micro_for_hook(audio_modifier, audio_cap)

    syn = hv5.get("synergy", {}) or {}
    syn_bonus = 0.0
    if arousal >= float(syn.get("arousal_gate", 0.60)):
        syn_bonus += float(syn.get("bonus_each", 0.015))
    if q_or_list >= float(syn.get("q_or_list_gate", 0.60)):
        syn_bonus += float(syn.get("bonus_each", 0.015))
    syn_bonus = min(syn_bonus, float(syn.get("cap_total", 0.04)))
    base = max(0.0, base) + syn_bonus
    if syn_bonus > 0: 
        reasons.append(f"synergy+{syn_bonus:.2f}")
        print(f"HOOK: synergy bonus q_or_list={q_or_list:.3f} arousal={arousal:.3f} +{syn_bonus:.3f}")

    raw = min(1.0, max(0.0, base))

    mu = 0.40
    sigma = 0.18
    cal = _calibrate_simple(raw, mu=mu, sigma=sigma, a=a_sig)

    debug = {
        "hook_v5_raw": round(raw, 6),
        "hook_v5_cal": round(cal, 6),
        "fam_scores": fam_scores,
        "fam_combined": round(fam_combined, 6),
        "evidence_ok": evidence_ok,
        "evidence_bits": evidence_bits,
        "proximity": round(prox, 6),
        "audio_mod": round(_audio_micro_for_hook(audio_modifier, audio_cap), 6),
        "pins": {"intro": pin, "outro": pout},
        "reasons": reasons,
        "mu": mu, "sigma": sigma, "a": a_sig,
    }
    return raw, cal, debug

def _emotion_score_v4(text: str) -> float:
    """Real emotion score v4 implementation"""
    # Get emotion words from config
    config = get_config()
    emo_words = config.get("lexicons", {}).get("EMO_WORDS", [])
    
    t = text.lower()
    high_intensity = ["incredible", "insane", "mind-blowing", "shocking"]
    high_hits = sum(1 for w in high_intensity if w in t)
    regular_hits = sum(1 for w in emo_words if w in t and w not in high_intensity)
    total_score = (high_hits * 2 + regular_hits) / 5.0
    return float(min(total_score, 1.0))

def _payoff_presence_v4(text: str) -> Tuple[float, str]:
    """Real payoff presence v4 implementation"""
    # Get payoff markers from config
    config = get_config()
    payoff_markers = config.get("lexicons", {}).get("PAYOFF_MARKERS", [])
    
    if not text or len(text.strip()) < 8:
        return 0.0, "too_short"
    
    t = text.lower()
    score = 0.0
    reasons = []

    # General payoff markers
    general_payoff_patterns = [
        r"here('?s)? (how|why|the deal|what you need)",  # "here's how"
        r"the (solution|answer|truth|key)", 
        r"because", r"so that", r"which means",
        r"in other words", r"this means",
        r"the (lesson|takeaway|insight|bottom line)",
        r"(therefore|thus|that's why)",
        r"turns out", r"it turns out"
    ]
    
    for pattern in general_payoff_patterns:
        if re.search(pattern, t):
            score += 0.25
            reasons.append("general_payoff")
            break

    # Check for payoff markers from config
    for marker in payoff_markers:
        if marker.lower() in t:
            score += 0.15
            reasons.append("config_payoff_marker")
            break

    # Value delivery patterns
    value_patterns = [
        r"(here's what|the key|the secret|the trick)",
        r"(you should|you need to|you must|you have to)",
        r"(this will|this can|this helps|this makes)",
        r"(the benefit|the advantage|the upside)"
    ]
    
    for pattern in value_patterns:
        if re.search(pattern, t):
            score += 0.15
            reasons.append("value_delivery")
            break

    # Penalty for questions without answers
    if "?" in t and not any(word in t for word in ["answer", "solution", "here's", "the key"]):
        score -= 0.1
        reasons.append("question_without_answer")

    final_score = float(np.clip(score, 0.0, 1.0))
    reason_str = ";".join(reasons) if reasons else "no_payoff"
    return final_score, reason_str

def _info_density_v4(text: str) -> float:
    """Real info density v4 implementation"""
    if not text or len(text.strip()) < 8:
        return 0.1
    
    t = text.lower()
    words = t.split()
    base = min(0.3, len(words) * 0.012)
    
    # Get filler words from config
    config = get_config()
    filler_words = config.get("lexicons", {}).get("FILLERS", ["you know", "like", "um", "uh", "right", "so"])
    filler_count = sum(1 for filler in filler_words if filler in t)
    filler_penalty = min(filler_count * 0.08, 0.4)
    
    numbers_and_stats = len(re.findall(r'\b\d+\b|[\$\d,]+|\d+%|\d+\.\d+', text))
    proper_nouns = sum(1 for w in text.split() if w[0].isupper() and len(w) > 2)
    
    specificity_boost = min((numbers_and_stats * 0.15 + proper_nouns * 0.12), 0.6)
    
    final = base - filler_penalty + specificity_boost
    return float(max(0.1, min(1.0, final)))

def compute_features_v4_enhanced(segment: Dict, audio_file: str, y_sr=None, genre: str = 'general', platform: str = 'tiktok', segments: list = None) -> Dict:
    """
    Enhanced feature computation with Phase 1, 2 & 3 improvements:
    - Typed containers with validation
    - Unified synergy scoring
    - Platform length v2
    - Path whitening (Phase 2)
    - Genre confidence blending (Phase 2)
    - Boundary hysteresis (Phase 2)
    - Prosody-aware arousal (Phase 3)
    - Payoff evidence guard (Phase 3)
    - Calibration system (Phase 3)
    """
    # Use existing compute_features_v4 as base
    features_dict = compute_features_v4(segment, audio_file, y_sr, genre, platform)
    
    if not FEATURE_TYPES:
        return features_dict
    
    # Convert to typed Features container
    features = Features.from_dict(features_dict)
    features.validate(strict=False)  # Clamp in production
    
    # Apply path whitening if enabled
    raw_paths = {
        'hook': features.hook,
        'arousal': features.arousal,
        'emotion': features.emotion,
        'payoff': features.payoff,
        'info_density': features.info_density,
        'q_list': features.q_list,
        'loopability': features.loopability,
        'platform_length': features.platform_length
    }
    
    whitened_paths = None
    if WHITEN_PATHS:
        whitened_paths = whiten_paths(raw_paths)
        # Update features with whitened values
        features.hook = whitened_paths['hook']
        features.arousal = whitened_paths['arousal']
        features.emotion = whitened_paths['emotion']
        features.payoff = whitened_paths['payoff']
        features.info_density = whitened_paths['info_density']
        features.q_list = whitened_paths['q_list']
        features.loopability = whitened_paths['loopability']
        features.platform_length = whitened_paths['platform_length']
        
        # Aha density boost (insightful + quantified statements)
        try:
            dur = float(segment.get("end",0.0) - segment.get("start",0.0))
            aha = _aha_density(segment.get("text",""), dur)
            white_info = float(whitened_paths['info_density'])
            whitened_paths['info_density'] = min(1.0, 0.5*white_info + 0.5*aha)
            features.info_density = whitened_paths['info_density']
        except Exception:
            pass
    
    # Apply genre confidence blending if enabled
    genre_debug = {}
    if GENRE_BLEND and segments:
        try:
            from .genres import GenreAwareScorer
            genre_scorer = GenreAwareScorer()
            
            # Get base weights
            base_weights = get_clip_weights()
            
            # Apply genre blending
            blended_weights, genre_debug = apply_genre_blending(base_weights, genre_scorer, segments)
            
            # Apply blended weights to features (simplified - in practice you'd integrate this with scoring)
            # This is a placeholder for the full integration
            features.meta['genre_blending'] = genre_debug
            features.meta['blended_weights'] = blended_weights
            
        except Exception as e:
            logger.warning(f"Genre blending failed: {e}")
            genre_debug = {"error": str(e), "blending_applied": False}
    
    # Apply Phase 3 enhancements
    phase3_debug = {}
    
    # Prosody-aware arousal
    if PROSODY_AROUSAL and audio_file:
        try:
            start = segment.get('start', 0)
            end = segment.get('end', start + 30)
            text = segment.get('text', '')
            
            # Get prosody-enhanced arousal score
            prosody_arousal = prosody_arousal_score(text, audio_file, start, end, genre)
            features.arousal = prosody_arousal
            phase3_debug['prosody_arousal'] = {
                'enabled': True,
                'original_arousal': features_dict.get('arousal_score', 0.0),
                'prosody_arousal': prosody_arousal,
                'improvement': prosody_arousal - features_dict.get('arousal_score', 0.0)
            }
        except Exception as e:
            logger.warning(f"Prosody arousal failed: {e}")
            phase3_debug['prosody_arousal'] = {'enabled': False, 'error': str(e)}
    else:
        phase3_debug['prosody_arousal'] = {'enabled': False, 'reason': 'disabled_or_no_audio'}
    
    # Payoff evidence guard
    if PAYOFF_GUARD:
        try:
            text = segment.get('text', '')
            hook_text = text[:100]  # First 100 chars as hook
            body_text = text[100:]  # Rest as body
            
            # Apply payoff guard
            original_payoff = features.payoff
            guarded_payoff = payoff_guard(hook_text, body_text, original_payoff, genre)
            features.payoff = guarded_payoff
            
            phase3_debug['payoff_guard'] = {
                'enabled': True,
                'original_payoff': original_payoff,
                'guarded_payoff': guarded_payoff,
                'capped': guarded_payoff < original_payoff,
                'genre': genre
            }
        except Exception as e:
            logger.warning(f"Payoff guard failed: {e}")
            phase3_debug['payoff_guard'] = {'enabled': False, 'error': str(e)}
    else:
        phase3_debug['payoff_guard'] = {'enabled': False, 'reason': 'disabled'}
    
    # Apply unified synergy scoring
    if SYNERGY_MODE == "unified":
        synergy = synergy_bonus(raw_paths)
    else:
        # Use existing synergy logic
        synergy = features_dict.get('synergy_multiplier', 1.0) - 1.0
    
    # Apply calibration
    if CALIBRATION_V:
        try:
            # Calibrate final score
            original_final = features_dict.get('final_score', 0.0)
            calibrated_final = apply_calibration(original_final, CALIBRATION_V)
            
            phase3_debug['calibration'] = {
                'enabled': True,
                'version': CALIBRATION_V,
                'original_final': original_final,
                'calibrated_final': calibrated_final,
                'adjustment': calibrated_final - original_final
            }
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            phase3_debug['calibration'] = {'enabled': False, 'error': str(e)}
    else:
        phase3_debug['calibration'] = {'enabled': False, 'reason': 'disabled'}
    
    # Quantize for stability
    features.quantize()
    
    # Convert back to dict with enhanced debug info
    result = features.to_dict()
    
    # ---- compute or default optional signals ----
    # Helper functions for safe computation
    def _maybe_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return None

    def _maybe_get(d, k, default=0.0):
        v = d.get(k)
        return default if v is None else v
    
    # question/list patterns
    # Ad detection first
    ad_result = _ad_penalty(text)
    ad_penalty = ad_result.get("penalty", 0.0)
    should_exclude = ad_result.get("flag", False)
    
    # Apply payoff guard to widen detection for narrative speech
    base_payoff = result.get('payoff_score', 0.0)
    # Split text into hook (first 100 chars) and body (rest)
    hook_text = text[:100] if text else ""
    body_text = text[100:] if text else ""
    enhanced_payoff = payoff_guard(hook_text, body_text, base_payoff, genre)
    result['payoff_score'] = enhanced_payoff
    # Add payoff V2 fields for compatibility and analytics (get from base function result)
    result['payoff_label'] = features_dict.get('payoff_label', 'none')
    result['payoff_span'] = features_dict.get('payoff_span', None)
    result['payoff_src'] = features_dict.get('payoff_src', 'v1')
    
    # Calculate duration for enhanced features
    duration_s = segment.get("end", 0) - segment.get("start", 0)
    
    q_list_score = 0.0
    if "_question_list_raw_v2" in globals():
        q_list_score = float(max(0.0, min(1.0, _question_list_raw_v2(text, duration_s, genre))))
    
    # platform-length v2 (fit + density) - defensive compute + fallback
    pl_v2_val = None
    try:
        if "platform_length_score_v2" in globals():
            pl_v2_val = float(globals()["platform_length_score_v2"](
                seconds=duration_s,
                info_density=result.get('info_density', 0.0),
                platform=platform
            ))
    except Exception as e:
        logger and logger.warning(f"pl_v2 compute failed: {e}")
    
    platform_length_score_v2 = max(0.0, min(1.0, pl_v2_val)) if pl_v2_val is not None else result.get('platform_len_match', 0.0)
    
    # prosody-aware arousal (if audio was present and prosody computed)
    prosody_arousal = None
    if "_audio_prosody_score" in globals() and audio_file:
        try:
            prosody_arousal = float(max(0.0, min(1.0, _audio_prosody_score(audio_file, start_ts, end_ts))))
        except Exception:
            prosody_arousal = None
    
    # insight confidence (if detector exposes it)
    insight_conf = locals().get("insight_conf", None)  # or pull from your detect_insight() result if available
    
    # Calculate final score using the scoring system
    from .scoring import score_segment_v4
    scored_result = score_segment_v4(result, genre=genre)
    viral_score_100 = scored_result.get('viral_score_100', 0)
    final_score = viral_score_100 / 100.0  # Convert from 0-100 to 0-1
    display_score = viral_score_100  # Keep in 0-100 range for quality filtering
    
    # Apply anti-bait cap to prevent micro-clips from scoring too high
    from .scoring_utils import apply_anti_bait_cap
    word_count = len(text.split()) if text else 0
    features_for_cap = {
        'text': text,
        'words': word_count,
        'hook_score': result.get('hook_score', 0),
        'payoff_score': enhanced_payoff,
        'ad_penalty': ad_penalty,
        'final_score': final_score,
        '_has_answer': False  # Will be set by question-answer stitching if applicable
    }
    final_score = apply_anti_bait_cap(features_for_cap)
    
    # Quantize final scores for stability
    from .types import quantize
    final_score = quantize(final_score)
    # Keep display_score in 0-100 range for quality filtering
    display_score = int(round(final_score * 100))  # Convert back to 0-100 range
    
    result.update({
        'final_score': final_score,
        'display_score': display_score,
        'raw_score': final_score,
        'clip_score_100': display_score,
        'synergy_multiplier': 1.0 + synergy,
        'synergy_bonus': synergy,
        'scoring_version': 'v4.7.2-unified-syn-whiten-blend-prosody-guard-cal',
        
        # newly surfaced (optional but now returned)
        'insight_conf': _maybe_get(locals(), 'insight_conf', 0.0),
        'q_list_score': _maybe_get(locals(), 'q_list_score', 0.0),
        'prosody_arousal': _maybe_get(locals(), 'prosody_arousal', 0.0),
        'platform_length_score_v2': _maybe_get(locals(), 'platform_length_score_v2', 0.0),
        
    # Ad detection enforcement
    'ad_penalty': _maybe_get(locals(), 'ad_penalty', 0.0),
    'should_exclude': _maybe_get(locals(), 'should_exclude', False),
    
    'debug': {
            'raw_paths': raw_paths,
            'whitened_paths': whitened_paths,
            'synergy_mode': SYNERGY_MODE,
            'feature_types_enabled': FEATURE_TYPES,
            'platform_len_v': PLATFORM_LEN_V,
            'whiten_paths_enabled': WHITEN_PATHS,
            'genre_blend_enabled': GENRE_BLEND,
            'boundary_hysteresis_enabled': BOUNDARY_HYSTERESIS,
            'prosody_arousal_enabled': PROSODY_AROUSAL,
            'payoff_guard_enabled': PAYOFF_GUARD,
            'calibration_enabled': bool(CALIBRATION_V),
            'calibration_version': CALIBRATION_V,
            'genre_debug': genre_debug,
            'phase3_debug': phase3_debug,
            'scoring_version': 'v4.8-unified-2025-09',
            'weights_version': '2025-09-01',
            'flags': {
                'USE_PL_V2': True,
                'USE_Q_LIST': True
            }
        }
    })
    
    return result

def find_viral_clips_enhanced(segments: List[Dict], audio_file: str, genre: str = 'general', platform: str = 'tiktok', fallback_mode: bool = None, effective_eos_times: List[float] = None, effective_word_end_times: List[float] = None, eos_source: str = None) -> Dict:
    """
    Enhanced viral clip finding with Phase 1, 2 & 3 improvements:
    - Path whitening
    - Genre confidence blending
    - Boundary hysteresis
    - Prosody-aware arousal
    - Payoff evidence guard
    - Calibration system
    """
    logger.info(f"Enhanced pipeline received {len(segments)} segments")
    
    # Use effective EOS data if provided, otherwise build from segments
    if effective_eos_times is not None and effective_word_end_times is not None:
        eos_times = effective_eos_times
        word_end_times = effective_word_end_times
        logger.info(f"EOS_REUSED: count={len(eos_times)}, src={eos_source} (inherited), fallback={fallback_mode}")
    else:
        # Build EOS index for Finish-Thought Gate (fallback)
        eos_times, word_end_times, eos_source = build_eos_index(segments)
        logger.info(f"EOS index: {len(eos_times)} EOS markers, {len(word_end_times)} word boundaries, source={eos_source}")
        
        # Use provided fallback_mode or determine from EOS density
        if fallback_mode is None:
            eos_density = len(eos_times) / max(len(word_end_times), 1)
            fallback_mode = eos_density < 0.020
    
    # Create FTTracker for authoritative finish-thought tracking
    ft = FTTracker(fallback_mode=fallback_mode)
    
    if not segments:
        return {
            'clips': [],
            'genre': genre,
            'platform': platform,
            'scoring_version': 'v4.7.1-unified-syn-whiten-blend',
            'debug': {
                'phase2_enabled': True,
                'whiten_paths': WHITEN_PATHS,
                'genre_blend': GENRE_BLEND,
                'boundary_hysteresis': BOUNDARY_HYSTERESIS
            }
        }
    
    # Apply boundary hysteresis if enabled
    if BOUNDARY_HYSTERESIS:
        optimal_segments = find_optimal_boundaries(segments, audio_file)
        if optimal_segments:
            segments = optimal_segments
    
    # Process each segment with enhanced features
    enhanced_segments = []
    skip_reasons = {}
    logger.info(f"Processing {len(segments)} segments with enhanced pipeline")
    
    for i, segment in enumerate(segments):
        skip_reason = None
        
        # Check for skip reasons before processing
        text = segment.get('text', '').strip()
        if not text:
            skip_reason = "empty_text"
        elif len(text.split()) < 3:
            skip_reason = "too_short_words"
        elif segment.get('end', 0) - segment.get('start', 0) < 5:
            skip_reason = "too_short_secs"
        elif segment.get('is_advertisement', False):
            skip_reason = "ad_flag"
        elif segment.get('should_exclude', False):
            skip_reason = "exclude_flag"
        
        if skip_reason:
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            logger.debug(f"Segment {i} skipped: {skip_reason}")
            continue
            
        try:
            enhanced_features = compute_features_v4_enhanced(
                segment, audio_file, genre=genre, platform=platform, segments=segments
            )
            
            # Check for post-processing skip reasons
            if enhanced_features.get('should_exclude', False):
                skip_reason = "post_processing_exclude"
            elif enhanced_features.get('final_score', 0) < 0.1:
                skip_reason = "min_score_gate"
            elif enhanced_features.get('payoff_score', 0) < 0.1 and enhanced_features.get('hook_score', 0) < 0.1:
                skip_reason = "min_payoff_gate"
            
            if skip_reason:
                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                logger.debug(f"Segment {i} skipped after processing: {skip_reason}")
                continue
            
            # Preserve original segment data with enhanced features
            enhanced_segment = {
                **segment,  # Keep original segment data (text, start, end, etc.)
                **enhanced_features  # Add enhanced features
            }
            enhanced_segments.append(enhanced_segment)
            
            # Log progress every 10 segments
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(segments)} segments")
                
        except Exception as e:
            logger.warning(f"Failed to process segment {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(enhanced_segments)}/{len(segments)} segments")
    
    # DYN: Duration histogram after dynamic segmentation
    durations = [s['end'] - s['start'] for s in enhanced_segments]
    hist, _ = np.histogram(durations, bins=[0, 12.5, 18.5, 24.5, 30.5, 60])
    logger.info("DYN: %d segs | dur histogram=%s", len(enhanced_segments), hist.tolist())
    
    # Generate length variants for top segments
    for i, segment in enumerate(enhanced_segments[:10]):  # Only for top 10 to avoid performance issues
        if segment.get('final_score', 0) > 0.3:  # Only for decent segments
            segment['_id'] = f"seg_{i}"
            
            # Try answer-stitching for questions before growing to bins
            from .scoring_utils import looks_like_question, try_stitch_answer
            if looks_like_question(segment.get('text', '')):
                next_segment = enhanced_segments[i + 1] if i + 1 < len(enhanced_segments) else None
                stitched = try_stitch_answer(segment, next_segment)
                if stitched:
                    # Re-score the stitched segment
                    stitched = score_variant(stitched, audio_file, genre, platform)
                    if stitched.get('final_score', 0) > segment.get('final_score', 0):
                        enhanced_segments[i] = stitched
                        continue
            
            enhanced_segments[i] = grow_to_bins(segment, audio_file, genre, platform, eos_times, word_end_times, fallback_mode, ft)
    
    # Sort by final score, then platform fit, then duration (prefer longer)
    def sort_key(seg):
        return (
            round(seg.get('final_score', 0), 3),
            round(seg.get('platform_length_score_v2', 0.0), 3),
            round((seg.get('end', 0) - seg.get('start', 0)), 2)
        )
    enhanced_segments.sort(key=sort_key, reverse=True)
    
    # Log score distribution
    if enhanced_segments:
        scores = [seg.get('final_score', 0) for seg in enhanced_segments]
        logger.info(f"Score distribution: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
        
        # Log top 3 segments
        for i, seg in enumerate(enhanced_segments[:3]):
            score = seg.get('final_score', 0)
            display_score = seg.get('display_score', 0)
            text_length = len(seg.get('text', '').split())
            logger.info(f"Top segment {i+1}: score={score:.3f}, display={display_score}, words={text_length}")
    
    # Take top clips
    top_clips = enhanced_segments[:10]  # Top 10 clips
    
    # Diversity guard: ensure at least one long platform-fit clip
    if len(top_clips) >= 4 and max(c.get('end', 0) - c.get('start', 0) for c in top_clips) < 16.0:
        # Find best long candidate from pre-NMS pool
        long_best = next((c for c in enhanced_segments 
                         if (c.get('end', 0) - c.get('start', 0)) >= 18.0 
                         and c.get('platform_length_score_v2', 0) >= 0.80), None)
        if long_best and long_best not in top_clips:
            top_clips[-1] = long_best  # Replace last slot
            top_clips.sort(key=sort_key, reverse=True)  # Re-sort
    
    # Calculate health metrics
    import statistics
    
    durations = [seg.get('end', 0) - seg.get('start', 0) for seg in enhanced_segments]
    scores = [seg.get('final_score', 0) for seg in enhanced_segments]
    
    # Calculate score spread and percentiles
    score_spread = max(scores) - min(scores) if scores else 0
    p70_score = np.percentile(scores, 70) if scores else 0
    p60_score = np.percentile(scores, 60) if scores else 0
    
    health_metrics = {
        'segments': len(enhanced_segments),
        'sec_p50': statistics.median(durations) if durations else 0,
        'sec_p90': statistics.quantiles(durations, n=10)[8] if len(durations) >= 10 else max(durations) if durations else 0,
        'yield_rate': len(top_clips) / max(1, len(enhanced_segments)),
        'score_analysis': {
            'min': min(scores) if scores else 0,
            'max': max(scores) if scores else 0,
            'avg': statistics.mean(scores) if scores else 0,
            'spread': score_spread,
            'p70': p70_score,
            'p60': p60_score,
            'low_contrast': score_spread < 0.20
        },
        'filters': {
            'ads_removed': 0,  # Will be calculated by filtering functions
            'intros_removed': 0,  # Will be calculated by filtering functions
            'caps_applied': len(segments) - len(enhanced_segments)  # Segments filtered by caps
        },
        'drop_reasons': skip_reasons
    }
    
    # Log authoritative FT summary
    logger.info("FT_SUMMARY_AUTH: total=%d finished=%d sparse=%d ratio_strict=%.2f ratio_sparse_ok=%.2f",
                ft.total, ft.finished, ft.sparse_finished, ft.ratio_strict, ft.ratio_sparse_ok)
    
    return {
        'clips': top_clips,
        'genre': genre,
        'platform': platform,
        'scoring_version': 'v4.7.2-unified-syn-whiten-blend-prosody-guard-cal',
        'ft': {
            'fallback_mode': ft.fallback_mode,
            'total': ft.total,
            'finished': ft.finished,
            'sparse_finished': ft.sparse_finished,
            'ratio_strict': ft.ratio_strict,
            'ratio_sparse_ok': ft.ratio_sparse_ok
        },
        'debug': {
            'phase2_enabled': True,
            'phase3_enabled': True,
            'whiten_paths': WHITEN_PATHS,
            'genre_blend': GENRE_BLEND,
            'boundary_hysteresis': BOUNDARY_HYSTERESIS,
            'prosody_arousal': PROSODY_AROUSAL,
            'payoff_guard': PAYOFF_GUARD,
            'calibration': bool(CALIBRATION_V),
            'calibration_version': CALIBRATION_V,
            'synergy_mode': SYNERGY_MODE,
            'platform_len_v': PLATFORM_LEN_V,
            'total_segments': len(segments),
            'processed_segments': len(enhanced_segments),
            'top_clips_count': len(top_clips),
            'health': health_metrics
        }
    }

# Stub implementations for unused functions
def debug_segment_scoring(*args, **kwargs):
    """Stub implementation"""
    return {}

def compute_features_v4_batch(*args, **kwargs):
    """Stub implementation"""
    return []

def compute_audio_hook_modifier(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def detect_laughter_exclamations(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def calculate_hook_components(*args, **kwargs):
    """Stub implementation"""
    return {}

def calculate_time_weighted_hook_score(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def score_patterns_in_text(*args, **kwargs):
    """Stub implementation"""
    return 0.0

# Removed duplicate stub implementations - using real implementations above

def _text_based_audio_estimation(text: str, genre: str = 'general') -> float:
    """Intelligent text-based audio arousal estimation when audio analysis fails"""
    if not text:
        return 0.3  # Default moderate arousal
    
    t = text.lower()
    score = 0.3  # Base moderate arousal
    
    # High-energy text indicators
    high_energy_indicators = [
        ('!', 0.1), ('amazing', 0.15), ('incredible', 0.15), ('crazy', 0.15),
        ('insane', 0.2), ('?!', 0.2), ('wow', 0.1), ('unbelievable', 0.15),
        ('shocking', 0.2), ('wild', 0.1), ('epic', 0.15), ('mind-blowing', 0.2)
    ]
    
    for indicator, boost in high_energy_indicators:
        if indicator in t:
            score = min(score + boost, 0.9)
            break  # Only apply the first match to avoid over-scoring
    
    # Genre-specific audio estimation
    if genre == 'comedy':
        comedy_indicators = ['hilarious', 'funny', 'lol', 'haha', 'rofl', 'joke']
        if any(indicator in t for indicator in comedy_indicators):
            score = min(score + 0.1, 0.9)
    elif genre == 'fantasy_sports':
        sports_indicators = ['fire', 'draft', 'start', 'bench', 'target', 'sleeper', 'bust']
        if any(indicator in t for indicator in sports_indicators):
            score = min(score + 0.1, 0.9)
    elif genre == 'true_crime':
        crime_indicators = ['murder', 'killer', 'victim', 'evidence', 'mystery', 'suspicious']
        if any(indicator in t for indicator in crime_indicators):
            score = min(score + 0.1, 0.9)
    
    return float(np.clip(score, 0.0, 1.0))

def _calibrate_info_density_stats(*args, **kwargs):
    """Stub implementation"""
    return {}

def _ql_calibrate_stats(*args, **kwargs):
    """Stub implementation"""
    return {}

def _calibrate_emotion_stats(*args, **kwargs):
    """Stub implementation"""
    return {}

def build_emotion_audio_sidecar(*args, **kwargs):
    """Stub implementation"""
    return {}

def compute_audio_energy(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def info_density_score_v2(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def _question_list_raw_v2(text: str, duration_s: float | None = None, genre: str | None = None) -> float:
    """Enhanced question/list scoring with multiple signals"""
    # Get configuration
    config = get_config()
    ql_cfg = config.get("question_list_v2", {})
    
    # Default configuration
    default_cfg = {
        "sigmoid_a": 1.5,
        "ideal_items_range": [3, 7],
        "pen_bait_cap": 0.25,
        "pen_vacuous_q_cap": 0.20,
        "textual_guard_min_tokens": 6,
        "genre_tweaks": {
            "education": {"list_bonus": 0.03, "question_bonus": 0.00},
            "entertainment": {"list_bonus": 0.00, "question_bonus": 0.03},
            "linkedin": {"list_bonus": 0.02, "question_bonus": 0.00}
        }
    }
    cfg = {**default_cfg, **ql_cfg}
    
    if not text:
        return 0.0
    
    toks = _tokens_ql(text)
    if len(toks) < cfg["textual_guard_min_tokens"]:
        return 0.35  # neutral for very short strings

    sents = _sentences(text)
    first = sents[0] if sents else text
    last = sents[-1] if sents else text

    # -------- Questions subscore --------
    qmark = 1.0 if _QMARK.search(last or "") else 0.0
    wh = 1.0 if (_INTERROG.search(last or "") or _INTERROG.search(first or "")) else 0.0
    compare = 1.0 if _COMPARE.search(text) else 0.0
    choice = 1.0 if _CHOICE.search(text) else 0.0
    rhetind = 1.0 if _RHET_IND.search(text) else 0.0
    cliffq = 1.0 if (_CLIFF_Q.search(last or "") and qmark) else 0.0
    genuine = 1.0 if _GENUINE.search(text) else 0.0

    # Internal weights (kept in code to avoid config sprawl)
    q_direct = 0.6 * qmark + 0.4 * wh
    q_compare = 0.3 * compare
    q_choice = 0.25 * choice
    q_rhet = 0.15 * rhetind
    q_cliff = 0.15 * cliffq
    q_prompt = min(0.10, 0.10 * genuine)  # tiny boost for honest prompts

    Q_raw = _saturating_sum([q_direct, q_compare, q_choice, q_rhet, q_cliff]) + q_prompt

    # -------- Lists subscore --------
    list_markers = _LIST_MARKERS.search(text)
    list_items = _LIST_ITEMS.findall(text)
    ideal_range = cfg["ideal_items_range"]
    
    list_count = len(list_items)
    if list_count == 0:
        list_density = 0.0
    elif ideal_range[0] <= list_count <= ideal_range[1]:
        list_density = 1.0
    else:
        # Penalty for too few or too many items
        if list_count < ideal_range[0]:
            list_density = 0.3 + 0.7 * (list_count / ideal_range[0])
        else:
            list_density = max(0.3, 1.0 - 0.1 * (list_count - ideal_range[1]))

    list_marker_bonus = 0.2 if list_markers else 0.0
    L_raw = list_density + list_marker_bonus

    # -------- Genre tweaks --------
    genre_tweaks = cfg["genre_tweaks"].get(genre or "", {})
    q_bonus = genre_tweaks.get("question_bonus", 0.0)
    l_bonus = genre_tweaks.get("list_bonus", 0.0)

    Q_raw += q_bonus
    L_raw += l_bonus

    # -------- Anti-bait penalties --------
    bait_penalty = 0.0
    if _BAIT_PATTERNS.search(text):
        bait_penalty = min(cfg["pen_bait_cap"], 0.1 * len(_BAIT_PATTERNS.findall(text)))
    
    vacuous_penalty = 0.0
    if _VACUOUS_Q.search(text):
        vacuous_penalty = min(cfg["pen_vacuous_q_cap"], 0.05 * len(_VACUOUS_Q.findall(text)))

    # -------- Final combination --------
    raw_score = _saturating_sum([Q_raw, L_raw]) - bait_penalty - vacuous_penalty
    return max(0.0, min(1.0, raw_score))

def question_list_score_v2(segment: dict, MU: float, SIGMA: float) -> float:
    """Enhanced question/list scoring with calibration"""
    text = segment.get("text") or segment.get("transcript") or ""
    genre = (segment.get("genre") or "").lower() or None
    raw = _question_list_raw_v2(text, genre=genre)
    
    # Get configuration
    config = get_config()
    ql_cfg = config.get("question_list_v2", {})
    a = ql_cfg.get("sigmoid_a", 1.5)
    
    z = (raw - MU) / (SIGMA if SIGMA > 1e-6 else 1.0)
    score = float(_sigmoid_ql(z, a))
    
    # Add debug information
    if "_debug" not in segment:
        segment["_debug"] = {}
    
    debug = segment["_debug"]
    debug["ql_raw"] = raw
    debug["ql_mu"] = MU
    debug["ql_sigma"] = SIGMA
    debug["ql_final"] = score
    
    return max(0.0, min(1.0, score))

def attach_question_list_scores_v2(segments: list[dict]) -> None:
    """Batch processing for question/list V2 with calibration"""
    if not segments:
        return
    
    # Calculate calibration parameters
    raw_scores = []
    for seg in segments:
        text = seg.get("text") or seg.get("transcript") or ""
        if text:
            raw = _question_list_raw_v2(text)
            raw_scores.append(raw)
    
    if not raw_scores:
        return
    
    # Simple calibration
    MU = sum(raw_scores) / len(raw_scores)
    SIGMA = (sum((x - MU) ** 2 for x in raw_scores) / len(raw_scores)) ** 0.5
    
    # Apply scores
    for seg in segments:
        seg["question_score"] = question_list_score_v2(seg, MU, SIGMA)

def emotion_score_v2(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def _emotion_raw_v2(text: str, arousal: float = 0.0) -> float:
    """
    Enhanced emotion detection with arousal coupling.
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Enhanced emotion keywords with affective lexicon
    emotion_keywords = [
        'excited', 'thrilled', 'amazed', 'surprised', 'shocked', 'stunned',
        'happy', 'joyful', 'delighted', 'pleased', 'satisfied', 'content',
        'sad', 'disappointed', 'frustrated', 'angry', 'mad', 'upset',
        'worried', 'anxious', 'nervous', 'scared', 'afraid', 'terrified',
        'proud', 'confident', 'grateful', 'thankful', 'appreciative',
        'love', 'hate', 'adore', 'despise', 'enjoy', 'dislike',
        'fascinated', 'intrigued', 'curious', 'interested', 'bored',
        'inspired', 'motivated', 'determined', 'focused', 'confused',
        'surprised', 'shocked', 'stunned', 'amazed', 'impressed',
        'disgusted', 'revolted', 'sickened', 'horrified', 'appalled',
        'jealous', 'envious', 'resentful', 'bitter', 'cynical',
        'hopeful', 'optimistic', 'pessimistic', 'doubtful', 'skeptical',
        'embarrassed', 'ashamed', 'guilty', 'regretful', 'remorseful',
        'relieved', 'relaxed', 'calm', 'peaceful', 'serene',
        'overwhelmed', 'stressed', 'pressured', 'burdened', 'exhausted',
        'energized', 'pumped', 'hyped', 'enthusiastic', 'passionate',
        'furious', 'livid', 'enraged', 'irate', 'fuming',
        'ecstatic', 'euphoric', 'blissful', 'elated', 'overjoyed',
        'devastated', 'crushed', 'heartbroken', 'miserable', 'wretched',
        'nostalgic', 'sentimental', 'melancholy', 'wistful', 'yearning',
        'grateful', 'thankful', 'appreciative', 'blessed', 'fortunate',
        'proud', 'accomplished', 'achieved', 'successful', 'victorious',
        'defeated', 'beaten', 'crushed', 'demoralized', 'discouraged',
        'hopeful', 'optimistic', 'confident', 'assured', 'certain',
        'doubtful', 'uncertain', 'hesitant', 'cautious', 'wary',
        'surprised', 'shocked', 'stunned', 'amazed', 'impressed',
        'disappointed', 'let down', 'disillusioned', 'disenchanted',
        'excited', 'thrilled', 'pumped', 'hyped', 'enthusiastic',
        'bored', 'uninterested', 'apathetic', 'indifferent', 'blasé',
        'fascinated', 'intrigued', 'captivated', 'engrossed', 'absorbed',
        'confused', 'bewildered', 'perplexed', 'puzzled', 'baffled',
        'inspired', 'motivated', 'encouraged', 'uplifted', 'empowered',
        'discouraged', 'demoralized', 'disheartened', 'deflated', 'crushed',
        'determined', 'resolved', 'committed', 'dedicated', 'focused',
        'distracted', 'unfocused', 'scattered', 'disorganized', 'chaotic',
        'peaceful', 'calm', 'serene', 'tranquil', 'placid',
        'agitated', 'restless', 'uneasy', 'uncomfortable', 'disturbed',
        'content', 'satisfied', 'fulfilled', 'complete', 'whole',
        'empty', 'hollow', 'void', 'incomplete', 'unfulfilled',
        'grateful', 'thankful', 'appreciative', 'blessed', 'fortunate',
        'resentful', 'bitter', 'cynical', 'jaded', 'disillusioned',
        'hopeful', 'optimistic', 'confident', 'assured', 'certain',
        'pessimistic', 'negative', 'gloomy', 'dismal', 'bleak',
    ]
    
    # Count emotion keywords
    emotion_count = sum(1 for keyword in emotion_keywords if keyword in text_lower)
    
    # Calculate base emotion score
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    
    base_emotion = min(emotion_count / total_words * 20, 1.0)
    
    # Enhanced arousal coupling: if high arousal and contains affective tokens, boost emotion score
    if arousal >= 0.65 and emotion_count >= 2:
        base_emotion = max(base_emotion, 0.25)
    
    return min(base_emotion, 1.0)

# Regex for outro detection
_OUTRO_RE = re.compile(
    r"(thanks for watching|subscribe|follow|like and subscribe|link in bio|see you next time)",
    re.I
)

def _gauss01(x: float, mu: float, sigma: float) -> float:
    """Gaussian function normalized to 0-1 range"""
    if sigma <= 1e-6:
        return 1.0 if abs(x - mu) < 1e-6 else 0.0
    z = (x - mu) / sigma
    return float(math.exp(-0.5 * z * z))

def _tri_band01(x: float, lo: float, hi: float) -> float:
    """Triangular function: 0 outside [lo,hi], 1 at midpoint, linear slopes"""
    if lo >= hi:
        return 0.0
    mid = 0.5 * (lo + hi)
    if x <= lo or x >= hi:
        return 0.0
    return (x - lo) / (mid - lo) if x < mid else (hi - x) / (hi - mid)

def _platform_length_score_v2(
    duration: float,
    platform: str,
    *,
    loopability: float = 0.0,
    wps: float | None = None,
    boundary_type: str = "",
    boundary_conf: float = 0.0,
    text_tail: str = "",
) -> float:
    """Enhanced platform length scoring with smooth curves and adaptive features"""
    if PLATFORM_LEN_V >= 2:
        # Use new Phase 1 implementation
        # Estimate info_density from text_tail if available
        info_density = 0.0
        if text_tail:
            # Simple heuristic for info density
            words = len(text_tail.split())
            if words > 0:
                info_density = min(1.0, words / 20.0)  # Rough estimate
        
        return platform_length_score_v2(duration, info_density, platform)
    
    # Fallback to original implementation
    # Get platform configuration with fallbacks
    config = get_config()
    plat_cfg = config.get("platform_length_v2", {})
    platforms = plat_cfg.get("platforms", {})
    
    # Default platform config
    default_cfg = {"mu": 22.0, "sigma": 7.0, "cap": 60.0, "wps": [2.8, 4.5]}
    cfg = platforms.get(platform, default_cfg)
    
    mu, sigma, cap = cfg["mu"], cfg["sigma"], cfg["cap"]
    wps_range = cfg["wps"]

    # Loop-aware shift: shorter & tighter target when highly loopable
    if loopability >= 0.60:
        mu -= 2.0
        sigma = max(5.0, sigma - 1.5)

    # Guardrails
    if duration <= 0.0:
        return 0.0
    if duration > cap:
        return 0.0

    # Smooth base score using Gaussian
    base = _gauss01(duration, mu, sigma)

    # Near-cap anxiety penalty (don't risk getting cut by platform)
    if duration >= cap - 1.0:
        base *= 0.85

    # Density harmony: gentle blend with platform's ideal WPS band
    if wps is not None and wps > 0:
        lo, hi = wps_range
        dens = _tri_band01(wps, lo, hi)  # 0..1
        base = 0.85 * base + 0.15 * dens

    # Boundary quality bonuses/penalties
    if boundary_type == "sentence_end" and boundary_conf >= 0.90:
        base = min(1.0, base + 0.10)
    elif boundary_type in ("sentence_end", "insight_marker") and boundary_conf >= 0.75:
        base = min(1.0, base + 0.05)
    elif boundary_type == "mid_word":
        base *= 0.85

    # Anti-outro penalty
    if text_tail and _OUTRO_RE.search(text_tail):
        base = max(0.0, base - 0.15)

    return float(max(0.0, min(1.0, base)))

def _info_density_raw_v2(*args, **kwargs):
    """Stub implementation"""
    return 0.0

def _detect_insight_content_v2(text: str, genre: str = 'general') -> tuple[float, str]:
    """Detect insight content V2 with evidence-based scoring and saturating combiner"""
    if not text or len(text.strip()) < 10:
        return 0.0, "too_short"
    
    t = text.lower()
    reasons = []
    
    # Evidence patterns (same as ViralMomentDetector V2)
    CONTRAST = re.compile(r"(most (people|folks)|everyone|nobody).{0,40}\b(actually|but|instead)\b", re.I)
    CAUSAL = re.compile(r"\b(because|therefore|so|which means)\b", re.I)
    HAS_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b", re.I)
    COMPAR = re.compile(r"\b(vs\.?|versus|more than|less than|bigger than|smaller than)\b", re.I)
    IMPER = re.compile(r"\b(try|avoid|do|don['']t|stop|start|focus|use|measure|swap|choose|should|need|must)\b", re.I)
    HEDGE = re.compile(r"\b(maybe|probably|i think|i guess|kinda|sort of)\b", re.I)
    
    # Evidence components (0-1 each)
    evidence_parts = []
    
    # Contrast detection
    if CONTRAST.search(t):
        evidence_parts.append(0.8)
        reasons.append("contrast")
    
    # Number/metric detection
    if HAS_NUM.search(t):
        evidence_parts.append(0.7)
        reasons.append("number")
    
    # Comparison detection
    if COMPAR.search(t):
        evidence_parts.append(0.6)
        reasons.append("comparison")
    
    # Causal reasoning
    if CAUSAL.search(t):
        evidence_parts.append(0.5)
        reasons.append("causal")
    
    # Imperative/actionable content
    if IMPER.search(t):
        evidence_parts.append(0.6)
        reasons.append("imperative")
    
    # Genre-specific patterns (reduced weights for V2)
    if genre in ['fantasy_sports', 'sports']:
        insight_patterns = [
            r"(observation|insight|noticed|realized|discovered)",
            r"(main|key|important|significant) (takeaway|point|finding)",
            r"(casual|serious|experienced) (drafters|players|managers)",
            r"(way better|much better|improved|evolved)",
            r"(under my belt|experience|seen|witnessed)",
            r"(home league|draft|waiver|roster)",
            r"(sleeper|bust|value|target|avoid)",
            r"(this week|next week|season|playoffs)"
        ]
        
        for pattern in insight_patterns:
            if re.search(pattern, t):
                evidence_parts.append(0.4)
                reasons.append("fantasy_insight")
                break  # Only count once per genre
        
        # Specific insight boost
        if re.search(r"(casual drafters are way better)", t):
            evidence_parts.append(0.6)
            reasons.append("specific_insight_boost")
    
    # General insight patterns (reduced weights)
    general_insight_patterns = [
        r"(here's what|the thing is|what i found|what i learned)",
        r"(the key|the secret|the trick|the strategy)",
        r"(most people|everyone|nobody) (thinks|believes|knows)",
        r"(contrary to|despite|although|even though)",
        r"(the truth is|reality is|actually|in fact)"
    ]
    
    for pattern in general_insight_patterns:
        if re.search(pattern, t):
            evidence_parts.append(0.3)
            reasons.append("general_insight")
            break  # Only count once
    
    # Hedge penalty (reduces confidence)
    hedge_penalty = 0.0
    if HEDGE.search(t):
        hedge_penalty = 0.2
        reasons.append("hedge_penalty")
    
    # Filler penalty (same as V1)
    filler_penalty = 0.0
    filler_patterns = [
        r"^(yo|hey|hi|hello|what's up)",
        r"^(it's|this is) (monday|tuesday|wednesday)",
        r"^(hope you|hope everyone)",
        r"^(let's get|let's start|let's begin)"
    ]
    
    for pattern in filler_patterns:
        if re.match(pattern, t):
            filler_penalty = 0.4
            reasons.append("filler_penalty")
            break
    
    # Saturating combiner: 1 - ∏(1 - xᵢ)
    if evidence_parts:
        sat_score = 1.0
        for part in evidence_parts:
            sat_score *= (1.0 - part)
        sat_score = 1.0 - sat_score
    else:
        sat_score = 0.0
    
    # Apply penalties
    final_score = sat_score - hedge_penalty - filler_penalty
    final_score = float(np.clip(final_score, 0.0, 1.0))
    
    reason_str = ";".join(reasons) if reasons else "no_insights"
    return final_score, reason_str

def _apply_insight_confidence_multiplier(insight_score: float, confidence: float = None) -> float:
    """Apply confidence-based multiplier to insight score if V2 is enabled"""
    config = get_config()
    if not config.get("insight_v2", {}).get("enabled", False) or confidence is None:
        return insight_score
    
    conf_config = config.get("insight_v2", {}).get("confidence_multiplier", {})
    min_mult = conf_config.get("min_mult", 0.9)
    max_mult = conf_config.get("max_mult", 1.2)
    conf_range = conf_config.get("confidence_range", [0.5, 0.9])
    
    # Map confidence to multiplier: confidence 0.5→×0.95, 0.9→×1.20
    conf_min, conf_max = conf_range
    if conf_min >= conf_max:
        return insight_score
    
    # Linear interpolation
    multiplier = min_mult + (max_mult - min_mult) * ((confidence - conf_min) / (conf_max - conf_min))
    multiplier = max(min_mult, min(max_mult, multiplier))
    
    # Apply multiplier and cap at 1.0
    adjusted_score = insight_score * multiplier
    return min(1.0, adjusted_score)


def detect_podcast_genre(*args, **kwargs):
    """Stub implementation for podcast genre detection"""
    return 'general'

def find_natural_boundaries(text: str) -> List[Dict]:
    """
    Find natural content boundaries in text for dynamic segmentation.
    Returns list of boundary points with their types and confidence.
    """
    boundaries = []
    words = text.split()
    
    # Look for natural break points
    for i, word in enumerate(words):
        # Strong content boundaries - check if word ends with punctuation
        if any(word.endswith(punct) for punct in [".", "!", "?", ":", ";"]):
            boundaries.append({
                "position": i + 1,  # Start after the punctuation
                "type": "sentence_end",
                "confidence": 0.9
            })
        
        # Topic transitions (expanded and more sensitive)
        elif any(phrase in " ".join(words[max(0, i-2):i+3]) for phrase in [
            "but", "however", "meanwhile", "on the other hand", "speaking of",
            "that reminds me", "by the way", "oh wait", "actually", "now", "so",
            "well", "okay", "right", "alright", "anyway", "moving on", "next",
            "another thing", "also", "plus", "additionally", "furthermore"
        ]):
            boundaries.append({
                "position": i,
                "type": "topic_shift",
                "confidence": 0.6  # Lowered from 0.7 to catch more transitions
            })
        
        # Story/insight markers (expanded and more sensitive)
        elif any(phrase in " ".join(words[max(0, i-1):i+2]) for phrase in [
            "here's the thing", "the key is", "the key insight", "what i learned", "my take",
            "the bottom line", "in summary", "to wrap up", "main observation",
            "key takeaway", "the thing is", "what i found", "here's what", "the insight",
            "here's why", "let me tell you", "you know what", "this is why", "the reason is",
            "the problem is", "the issue is", "the challenge is", "the solution is", "the answer is",
            "the truth is", "the reality is", "the fact is", "the secret is", "the trick is",
            "the way to", "the best way", "the only way", "the right way", "the wrong way",
            "i think", "i believe", "i feel", "i know", "i understand", "i realize",
            "the point is", "the idea is", "the concept is", "the principle is"
        ]):
            boundaries.append({
                "position": i,
                "type": "insight_marker",
                "confidence": 0.7  # Lowered from 0.8 to catch more insights
            })
        
        # Question/answer patterns
        elif word == "?" and i < len(words) - 5:
            # Look for answer patterns after question
            next_words = " ".join(words[i+1:i+6])
            if any(pattern in next_words for pattern in [
                "well", "so", "the answer", "here's", "let me tell you"
            ]):
                boundaries.append({
                    "position": i + 1,
                    "type": "qa_boundary",
                    "confidence": 0.8
                })
        
        # Comma boundaries (weaker but useful)
        elif word == "," and i > 5 and i < len(words) - 5:
            # Check if it's a natural pause
            context = " ".join(words[i-3:i+4])
            if any(phrase in context for phrase in [
                "first", "second", "third", "also", "additionally", "furthermore"
            ]):
                boundaries.append({
                    "position": i + 1,
                    "type": "comma_boundary",
                    "confidence": 0.5
                })
    
    # Remove duplicate positions and sort
    unique_boundaries = []
    seen_positions = set()
    
    for boundary in sorted(boundaries, key=lambda x: x["position"]):
        if boundary["position"] not in seen_positions and 0 < boundary["position"] < len(words):
            unique_boundaries.append(boundary)
            seen_positions.add(boundary["position"])
    
    return unique_boundaries

def create_dynamic_segments(segments: List[Dict], platform: str = 'tiktok', eos_times: list[float] | None = None) -> List[Dict]:
    """
    Create dynamic segments based on natural content boundaries and platform optimization.
    """
    # Local helper for null-safe length
    def _safe_len(x):
        return len(x) if isinstance(x, (list, tuple, str)) else 0
    
    dynamic_segments = []
    
    # Platform-specific target bins (allow extension to others via grow_to_bins)
    TARGETS = [12.0, 18.0, 24.0, 30.0]
    MIN_SEC, MAX_SEC = 8.0, 60.0
    
    # Calculate EOS density for seed merging decision
    eos_density = len(eos_times) / (segments[-1].get("end", 0) - segments[0].get("start", 0)) if segments and eos_times else 0.0
    
    # Compute dense flag once after EOS_UNIFIED
    src = "episode"  # or "gap_fallback" if from fallback
    dense = (eos_density >= 0.04 and src != 'gap_fallback') or (src == 'gap_fallback' and eos_density >= 0.05)
    
    # Only log EOS details if we were actually given eos_times
    if eos_times is not None:
        count = len(eos_times)
        fallback_flag = (src == "gap_fallback")
        log.info("EOS: %d markers, density=%.3f, source=%s, fallback=%s",
                 count, eos_density, src, fallback_flag)

    platform_lengths = {
        'tiktok': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': TARGETS},
        'instagram': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': TARGETS},
        'instagram_reels': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': TARGETS},
        'youtube': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': TARGETS + [45.0, 60.0]},
        'youtube_shorts': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': TARGETS + [45.0, 60.0]},
        'twitter': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': [8.0, 12.0, 18.0, 24.0]},
        'linkedin': {'min': MIN_SEC, 'max': MAX_SEC, 'targets': TARGETS + [45.0]}
    }
    
    target_length = platform_lengths.get(platform, platform_lengths['tiktok'])
    
    # Combine all segments into one continuous text for better boundary detection
    combined_text = " ".join([seg.get("text", "") for seg in segments])
    total_start = segments[0].get("start", 0) if segments else 0
    total_end = segments[-1].get("end", 0) if segments else 0
    total_duration = total_end - total_start
    
    # Find natural boundaries in the combined text
    boundaries = find_natural_boundaries(combined_text)
    
    if not boundaries or len(boundaries) < 2:
        # No natural boundaries found, use original segments
        return segments
    
    # Create segments based on boundaries
    words = combined_text.split() if combined_text else []
    if not words:
        log.warning("DYNSEG: no words found; coercing to [] (ep=%s)", segments[0].get('episode_id', 'unknown') if segments else 'unknown')
        return segments
    current_start = total_start
    
    # Add start boundary
    all_boundaries = [{"position": 0, "type": "start", "confidence": 1.0}] + boundaries
    
    for i, boundary in enumerate(all_boundaries):
        if boundary["confidence"] < 0.3:  # Much lower threshold to include more boundaries
            continue  # Skip low-confidence boundaries
        
        # Calculate end position
        if i + 1 < len(all_boundaries):
            next_boundary = all_boundaries[i + 1]
            end_position = next_boundary["position"]
        else:
            end_position = len(words)
        
        # Extract segment text
        segment_words = words[boundary["position"]:end_position]
        segment_text = " ".join(segment_words)
        
        if len(segment_words) < 3:  # Further reduced to allow even shorter segments
            continue
        
        # Calculate timing based on word count and total duration
        total_words = len(words)
        segment_ratio = len(segment_words) / total_words
        segment_duration = total_duration * segment_ratio
        
        # Ensure minimum duration for platform requirements
        if segment_duration < target_length["min"]:
            segment_duration = target_length["min"]
        
        # CRITICAL FIX: Find actual transcript segments that overlap with this text range
        # Calculate approximate time range for this text segment
        text_ratio = boundary["position"] / total_words
        end_ratio = end_position / total_words
        approx_start_time = total_start + (text_ratio * total_duration)
        approx_end_time = total_start + (end_ratio * total_duration)
        
        # Find original segments that overlap with this time range
        overlapping_segments = []
        for orig_seg in segments:
            orig_start = orig_seg.get("start", 0)
            orig_end = orig_seg.get("end", 0)
            # Check if this original segment overlaps with our calculated time range
            if orig_start < approx_end_time and orig_end > approx_start_time:
                overlapping_segments.append(orig_seg)
        
        # Use the actual text from overlapping segments
        if overlapping_segments:
            # Reconstruct text from actual transcript segments
            segment_text = " ".join([seg.get("text", "") for seg in overlapping_segments])
            original_start = min(seg.get("start", 0) for seg in overlapping_segments)
            original_end = max(seg.get("end", 0) for seg in overlapping_segments)
            segment_duration = original_end - original_start
            current_start = original_start
        else:
            # Fallback to calculated timing
            segment_duration = total_duration * segment_ratio
        
        # Check if segment meets platform requirements
        if target_length["min"] <= segment_duration <= target_length["max"]:
            segment = {
                "text": segment_text,
                "start": current_start,
                "end": current_start + segment_duration,
                "boundary_type": boundary["type"],
                "confidence": boundary["confidence"]
            }
            # Apply EOS extension for coherence
            pl_v2 = segment.get("platform_length_score_v2", 0.5)
            segment = _variantize_segment(segment, pl_v2=pl_v2, cap_s=30.0, eos_times=eos_times)
            # Apply caps filtering
            if _keep(segment):
                dynamic_segments.append(segment)
        elif segment_duration < target_length["min"]:
            # If segment is too short, just extend it to minimum length instead of merging
            extended_duration = target_length["min"]
            segment = {
                "text": segment_text,
                "start": current_start,
                "end": current_start + extended_duration,
                "boundary_type": "extended",
                "confidence": boundary["confidence"]
            }
            # Apply EOS extension for coherence
            pl_v2 = segment.get("platform_length_score_v2", 0.5)
            segment = _variantize_segment(segment, pl_v2=pl_v2, cap_s=30.0, eos_times=eos_times)
            # Apply caps filtering
            if _keep(segment):
                dynamic_segments.append(segment)
        
        current_start += segment_duration
    
    # Apply seed merging BEFORE variantization using dense flag
    if dynamic_segments:
        # Log pre-merge stats
        pre_durs = [round(seg.get("end", 0) - seg.get("start", 0), 1) for seg in dynamic_segments]
        pre_hist = {}
        for d in pre_durs:
            pre_hist[d] = pre_hist.get(d, 0) + 1
        log.debug("SEED_STATS_PRE: n=%d, dur_hist=%s", len(dynamic_segments), pre_hist)
    
    if dense:
        # Merge short seeds into 16-28s windows
        merged_seeds = _greedy_merge_short_seeds(dynamic_segments)
        dynamic_segments = merged_seeds
        log.info("SEED_MERGE: merged=%d -> median_len=%.1fs", len(merged_seeds), sorted(pre_durs)[len(pre_durs)//2] if pre_durs else 0)
    else:
        # Even for non-dense, merge 8s seeds to get sentence-level chunks
        if pre_durs and max(pre_durs) <= 10.0:  # All seeds are 8-10s
            merged_seeds = _greedy_merge_short_seeds(dynamic_segments)
            dynamic_segments = merged_seeds
            log.info("SEED_MERGE: merged=%d -> median_len=%.1fs", len(merged_seeds), sorted(pre_durs)[len(pre_durs)//2] if pre_durs else 0)
    
    # Log post-merge stats and assert dense requirement
    if dynamic_segments:
        post_durs = [round(seg.get("end", 0) - seg.get("start", 0), 1) for seg in dynamic_segments]
        post_hist = {}
        for d in post_durs:
            post_hist[d] = post_hist.get(d, 0) + 1
        log.debug("SEED_STATS_POST: n=%d, dur_hist=%s", len(dynamic_segments), post_hist)
        
        # Assert dense requirement
        if dense:
            median_dur = sorted(post_durs)[len(post_durs)//2]
            if median_dur < 16.0:
                log.warning("WARN: dense=True but median seed length %.1fs < 16.0s", median_dur)
            else:
                log.info("SEEDS: pre={%s} post={%s} median=%.1f", 
                         {k: v for k, v in pre_hist.items() if v > 0},
                         {k: v for k, v in post_hist.items() if v > 0},
                         median_dur)
    
    # If no dynamic segments were created, return original segments
    if not dynamic_segments:
        return segments
    
    # Post-process: ensure all segments meet minimum length requirements
    final_segments = []
    for seg in dynamic_segments:
        duration = seg["end"] - seg["start"]
        if duration < target_length["min"]:
            # Extend short segments to minimum length
            seg["end"] = seg["start"] + target_length["min"]
            seg["boundary_type"] = "extended"
        final_segments.append(seg)
    
    # Apply final caps filtering to ensure all segments meet requirements
    final_segments = [seg for seg in final_segments if _keep(seg)]
    
    return final_segments

def _explain_viral_potential_v4(features: Dict, scoring: Dict, genre: str = 'general') -> str:
    """Generate human-readable explanation of viral potential with genre context"""
    score = scoring["viral_score_100"]
    path = scoring["winning_path"]
    
    # Genre-specific context
    genre_context = {
        'fantasy_sports': 'fantasy sports analysis',
        'sports': 'sports commentary',
        'comedy': 'comedy content',
        'business': 'business advice',
        'education': 'educational content',
        'true_crime': 'true crime content',
        'health_wellness': 'health and wellness advice',
        'news_politics': 'news and political content',
        'general': 'content'
    }
    
    genre_name = genre_context.get(genre, 'content')
    
    if score >= 80:
        level = "EXCEPTIONAL"
        description = f"This {genre_name} clip has exceptional viral potential"
    elif score >= 70:
        level = "HIGH"
        description = f"This {genre_name} clip has high viral potential"
    elif score >= 60:
        level = "GOOD"
        description = f"This {genre_name} clip has good viral potential"
    elif score >= 50:
        level = "MODERATE"
        description = f"This {genre_name} clip has moderate viral potential"
    elif score >= 40:
        level = "LOW"
        description = f"This {genre_name} clip has low viral potential"
    else:
        level = "VERY LOW"
        description = f"This {genre_name} clip has very low viral potential"
    
    # Path explanation
    path_explanations = {
        'hook': 'strong opening that grabs attention',
        'payoff': 'valuable conclusion or insight',
        'energy': 'high energy and emotional engagement',
        'structured': 'well-organized and informative',
        'actionable': 'provides specific, actionable advice',
        'hot_take': 'controversial or bold opinion',
        'setup_punchline': 'classic comedy structure',
        'mystery': 'intriguing mystery elements',
        'resolution': 'satisfying resolution',
        'authority': 'expert authority and credibility',
        'specificity': 'concrete details and specifics',
        'curiosity': 'curiosity gap and knowledge',
        'clarity': 'clear explanations',
        'credibility': 'trustworthy health advice',
        'transformation': 'life transformation potential',
        'urgency': 'breaking news urgency',
        'controversy': 'controversial political content'
    }
    
    path_desc = path_explanations.get(path, 'balanced scoring across multiple dimensions')
    
    # Feature highlights
    highlights = []
    if features.get('hook_score', 0) > 0.7:
        highlights.append("strong hook")
    if features.get('payoff_score', 0) > 0.7:
        highlights.append("clear payoff")
    if features.get('arousal_score', 0) > 0.7:
        highlights.append("high energy")
    if features.get('emotion_score', 0) > 0.7:
        highlights.append("emotional engagement")
    
    # Genre-specific highlights
    if genre != 'general':
        from .genres import GenreAwareScorer
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre)
        if genre_profile:
            if 'viral_trigger_boost' in features and features['viral_trigger_boost'] > 0:
                highlights.append(f"genre-specific viral triggers")
            if 'confidence_score' in features and features['confidence_score'] > 0.5:
                highlights.append("high confidence indicators")
            if 'urgency_score' in features and features['urgency_score'] > 0.5:
                highlights.append("time-sensitive content")
    
    if highlights:
        feature_text = f"Key strengths: {', '.join(highlights)}"
    else:
        feature_text = "Balanced performance across features"
    
    # Synergy explanation
    synergy = scoring.get('synergy_multiplier', 1.0)
    if synergy > 1.1:
        synergy_text = "Excellent synergy between features"
    elif synergy > 1.0:
        synergy_text = "Good synergy between features"
    elif synergy < 0.9:
        synergy_text = "Features could work better together"
    else:
        synergy_text = "Standard feature interaction"
    
    return f"{description} ({level}: {score}/100). The clip excels in {path_desc}. {feature_text}. {synergy_text}."

def _grade_breakdown(feats: dict) -> dict:
    """Generate detailed grade breakdown for scoring explanation"""
    breakdown = {}
    
    # Core scoring components
    breakdown['hook'] = {
        'score': feats.get('hook_score', 0.0),
        'grade': _score_to_grade(feats.get('hook_score', 0.0)),
        'description': 'Attention-grabbing opening'
    }
    
    breakdown['arousal'] = {
        'score': feats.get('arousal_score', 0.0),
        'grade': _score_to_grade(feats.get('arousal_score', 0.0)),
        'description': 'Energy and excitement level'
    }
    
    breakdown['emotion'] = {
        'score': feats.get('emotion_score', 0.0),
        'grade': _score_to_grade(feats.get('emotion_score', 0.0)),
        'description': 'Emotional engagement'
    }
    
    breakdown['payoff'] = {
        'score': feats.get('payoff_score', 0.0),
        'grade': _score_to_grade(feats.get('payoff_score', 0.0)),
        'description': 'Clear value or insight'
    }
    
    breakdown['info'] = {
        'score': feats.get('info_density', 0.0),
        'grade': _score_to_grade(feats.get('info_density', 0.0)),
        'description': 'Information density'
    }
    
    breakdown['q_or_list'] = {
        'score': feats.get('question_score', 0.0),
        'grade': _score_to_grade(feats.get('question_score', 0.0)),
        'description': 'Questions or list format'
    }
    
    breakdown['loop'] = {
        'score': feats.get('loopability', 0.0),
        'grade': _score_to_grade(feats.get('loopability', 0.0)),
        'description': 'Replayability factor'
    }
    
    breakdown['length'] = {
        'score': feats.get('platform_length_score', 0.0),
        'grade': _score_to_grade(feats.get('platform_length_score', 0.0)),
        'description': 'Optimal length for platform'
    }
    
    return breakdown

def _score_to_grade(score: float) -> str:
    """Convert score to letter grade"""
    if score >= 0.9:
        return 'A+'
    elif score >= 0.8:
        return 'A'
    elif score >= 0.7:
        return 'B+'
    elif score >= 0.6:
        return 'B'
    elif score >= 0.5:
        return 'C+'
    elif score >= 0.4:
        return 'C'
    elif score >= 0.3:
        return 'D'
    else:
        return 'F'

def _heuristic_title(text: str, feats: dict, cfg: dict, rank: int | None = None, avoid: set[str] | None = None) -> str:
    """Generate heuristic title based on content and features - delegates to new generator"""
    try:
        from ..title_service import generate_titles, normalize_platform
        
        # Extract platform from features if available
        platform = "default"
        if feats and isinstance(feats, dict):
            platform = feats.get("platform", "default")
        
        plat = normalize_platform(platform)
        variants = generate_titles(text or "", platform=plat, n=6, avoid_titles=(avoid or set()))
        title = variants[0]["title"] if variants else "Most Leaders Solve the Wrong Problem"
        
        # Add rank if provided (maintain backward compatibility)
        if rank is not None:
            title = f"#{rank}: {title}"
        
        return title
    except Exception:
        # never break the pipeline because of titles
        return "Most Leaders Solve the Wrong Problem"

# Platform and tone mapping features
PLATFORM_GENRE_MULTIPLIERS = {
    'tiktok': {
        'comedy': 1.2,  # Perfect match
        'fantasy_sports': 0.8,  # Harder to succeed
        'education': 0.9,
        'true_crime': 1.1,
        'business': 0.95,
        'news_politics': 1.0,
        'health_wellness': 1.05
    },
    'instagram': {
        'comedy': 1.15,  # Great match
        'fantasy_sports': 0.9,  # Better than TikTok
        'education': 1.0,
        'true_crime': 1.05,
        'business': 1.0,
        'news_politics': 0.95,
        'health_wellness': 1.1
    },
    'instagram_reels': {
        'comedy': 1.15,  # Great match
        'fantasy_sports': 0.9,  # Better than TikTok
        'education': 1.0,
        'true_crime': 1.05,
        'business': 1.0,
        'news_politics': 0.95,
        'health_wellness': 1.1
    },
    'youtube_shorts': {
        'education': 1.15,  # Great match
        'comedy': 1.1,
        'fantasy_sports': 0.9,
        'true_crime': 1.0,
        'business': 1.05,
        'news_politics': 1.0,
        'health_wellness': 1.1
    },
    'linkedin': {
        'business': 1.2,  # Perfect match
        'education': 1.1,
        'news_politics': 1.0,
        'health_wellness': 0.95,
        'comedy': 0.7,  # Not ideal for LinkedIn
        'fantasy_sports': 0.8,
        'true_crime': 0.85
    }
}

TONE_TO_GENRE_MAP = {
    'tutorial_business': 'business',
    'comedy': 'comedy',
    'motivation': 'health_wellness',
    'educational': 'education',
    'sports_analysis': 'fantasy_sports',
    'news_commentary': 'news_politics',
    'personal_story': 'true_crime',  # Personal stories often have narrative arcs
    'how_to': 'education',
    'product_review': 'business',
    'fitness_tips': 'health_wellness',
    'cooking': 'education',
    'travel': 'education',
    'gaming': 'comedy',  # Gaming content often has entertainment value
    'music_reaction': 'comedy',
    'movie_review': 'comedy',
    'book_summary': 'education',
    'investment_advice': 'business',
    'relationship_advice': 'health_wellness',
    'parenting_tips': 'health_wellness'
}

# Frontend Platform to Backend Platform Mapping
PLATFORM_MAP = {
    'tiktok_reels': 'tiktok',
    'instagram_reels': 'instagram_reels',
    'shorts': 'youtube_shorts',
    'linkedin_sq': 'linkedin'
}

def resolve_platform(frontend_platform: str) -> str:
    """Map frontend platform names to backend platform names"""
    return PLATFORM_MAP.get(frontend_platform, frontend_platform)

def resolve_genre_from_tone(tone: str, auto_detected: str) -> str:
    """Map frontend tone to backend genre with fallback to auto-detected"""
    if tone and tone in TONE_TO_GENRE_MAP:
        mapped_genre = TONE_TO_GENRE_MAP[tone]
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print(f"🎯 Tone '{tone}' mapped to genre: {mapped_genre}")
        return mapped_genre
    
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        print(f"🎯 No tone mapping found for '{tone}', using auto-detected: {auto_detected}")
    return auto_detected

def interpret_synergy(synergy_mult: float, features: Dict) -> Dict:
    """Provide actionable synergy interpretation with improved feedback"""
    if synergy_mult < 0.7:
        return {
            "label": "Imbalanced",
            "advice": "Hook is strong but lacks energy/payoff",
            "color": "#ffc107",
            "severity": "warning"
        }
    elif synergy_mult < 0.85:
        return {
            "label": "⚖️ Mixed Performance", 
            "advice": "Some elements work, others need improvement",
            "color": "#6c757d",
            "severity": "info"
        }
    elif synergy_mult < 1.0:
        return {
            "label": "Good Balance",
            "advice": "All elements working together",
            "color": "#28a745",
            "severity": "success"
        }
    else:
        return {
            "label": "🔥 Excellent Synergy",
            "advice": "Perfect balance of hook, energy, and payoff",
            "color": "#007bff",
            "severity": "excellent"
        }

def get_genre_detection_debug(segments: List[Dict], detected_genre: str, applied_genre: str, tone: str = None) -> Dict:
    """Generate debug information for genre detection and application"""
    if not segments:
        return {
            "auto_detected_genre": "none",
            "applied_genre": "none",
            "genre_confidence": "none",
            "top_genre_patterns": [],
            "tone_used": tone,
            "mapping_applied": False
        }
    
    # Analyze top genre patterns
    sample_text = " ".join([seg.get('text', '') for seg in segments[:3]])
    from .genres import GenreAwareScorer
    genre_scorer = GenreAwareScorer()
    
    # Get confidence scores for all genres
    genre_scores = {}
    for genre_name in genre_scorer.genres.keys():
        confidence = genre_scorer.detect_genre_with_confidence(segments, genre_name)
        genre_scores[genre_name] = confidence
    
    # Sort by confidence
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    top_patterns = [{"genre": g, "confidence": c} for g, c in sorted_genres[:3]]
    
    return {
        "auto_detected_genre": detected_genre,
        "applied_genre": applied_genre,
        "genre_confidence": "high" if detected_genre != 'general' else "low",
        "top_genre_patterns": top_patterns,
        "tone_used": tone,
        "mapping_applied": tone is not None and tone in TONE_TO_GENRE_MAP,
        "sample_text": sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
    }

def find_viral_clips_with_tone(segments: List[Dict], audio_file: str, tone: str = None, auto_detect: bool = True) -> Dict:
    """
    Enhanced viral clip finding with tone-to-genre mapping and comprehensive debug info.
    
    Args:
        segments: List of podcast segments
        audio_file: Path to audio file
        tone: Frontend tone selection (optional)
        auto_detect: Whether to auto-detect genre if no tone provided
    
    Returns:
        Dict with candidates, debug info, and genre details
    """
    # Auto-detect genre from content
    from .genres import detect_podcast_genre
    detected_genre = detect_podcast_genre(segments)
    
    # Override with tone if provided
    if tone:
        applied_genre = resolve_genre_from_tone(tone, detected_genre)
    else:
        applied_genre = detected_genre
    
    # Process with selected genre
    result = find_viral_clips(segments, audio_file, genre=applied_genre)
    
    # Add comprehensive debug information
    debug_info = get_genre_detection_debug(segments, detected_genre, applied_genre, tone)
    
    # Enhance result with debug info
    enhanced_result = {
        'genre': applied_genre,
        'clips': result.get('clips', []),
        'debug': debug_info,
        'tone_mapping': {
            'tone_provided': tone,
            'auto_detected': detected_genre,
            'final_genre': applied_genre,
            'mapping_used': tone is not None and tone in TONE_TO_GENRE_MAP
        }
    }
    
    # Add synergy interpretation for each clip
    for clip in enhanced_result['clips']:
        if 'synergy_multiplier' in clip:
            clip['synergy_interpretation'] = interpret_synergy(
                clip['synergy_multiplier'], 
                clip.get('features', {})
            )
    
    return enhanced_result

def find_viral_clips_with_genre(segments: List[Dict], audio_file: str, user_genre: str = None) -> Dict:
    """
    Enhanced viral clip finding with genre auto-detection and user override.
    
    Args:
        segments: List of podcast segments
        audio_file: Path to audio file
        user_genre: User-selected genre (optional, will auto-detect if None)
    
    Returns:
        Dict with genre info and top viral clips
    """
    # Auto-detect genre if user didn't specify
    if user_genre is None:
        from .genres import detect_podcast_genre
        detected_genre = detect_podcast_genre(segments)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print(f"Auto-detected genre: {detected_genre}")
            print("You can override this by selecting a specific genre")
        genre = detected_genre
    else:
        genre = user_genre
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print(f"Using user-selected genre: {genre}")
    
    # Find viral clips with genre awareness
    result = find_viral_clips(segments, audio_file, genre=genre)
    
    # Add genre detection confidence
    if user_genre is None:
        result['auto_detected'] = True
        result['detection_confidence'] = 'high' if genre != 'general' else 'low'
    else:
        result['auto_detected'] = False
        result['detection_confidence'] = 'user_override'
    
    return result

def find_candidates(segments: List[Dict], audio_file: str, platform: str = 'tiktok', tone: str = None, auto_detect: bool = True) -> Dict:
    """
    Main API function for finding viral candidates with tone-to-genre mapping.
    
    Args:
        segments: List of podcast segments
        audio_file: Path to audio file
        platform: Target platform (tiktok, youtube_shorts, etc.)
        tone: Frontend tone selection
        auto_detect: Whether to auto-detect genre
    
    Returns:
        Dict with candidates and comprehensive metadata
    """
    import datetime
    
    # Use the enhanced function with tone mapping
    result = find_viral_clips_with_tone(segments, audio_file, tone, auto_detect)
    
    # Add platform-specific information
    result['platform'] = platform
    result['processing_timestamp'] = str(datetime.datetime.now())
    
    # Add platform-genre compatibility scores
    if platform in PLATFORM_GENRE_MULTIPLIERS:
        genre = result['genre']
        compatibility = PLATFORM_GENRE_MULTIPLIERS[platform].get(genre, 1.0)
        result['platform_compatibility'] = {
            'score': compatibility,
            'interpretation': 'excellent' if compatibility >= 1.1 else 'good' if compatibility >= 1.0 else 'challenging' if compatibility >= 0.9 else 'difficult'
        }
    
    # Sanity logs for pl_v2 debugging (temporarily keep)
    if logger and segment_idx < 5:  # first few only
        logger.info(
            "pl_v2 dbg: dur=%.1f info=%.2f platform=%s -> v2=%s (v1=%.2f)",
            duration_s, result.get('info_density', 0.0), platform,
            str(platform_length_score_v2), float(result.get('platform_len_match', 0.0))
        )
    
    return result

def filter_ads_from_features(all_features: List[Dict]) -> List[Dict]:
    """
    Filter out advertisements completely from the feature list.
    Returns only non-ad content for scoring.
    """
    non_ad_features = [f for f in all_features if not f.get("is_advertisement", False)]
    
    if len(non_ad_features) < 5:
        return {"error": "Episode is mostly advertisements, no viable clips found"}
    
    return non_ad_features

def filter_intro_content_from_features(all_features: List[Dict]) -> List[Dict]:
    """
    Filter out intro/greeting content completely from the feature list.
    Returns only substantive content for scoring.
    """
    # Filter out intro content based on insight score and hook reasons
    non_intro_features = []
    
    for f in all_features:
        # Skip if it's marked as intro content
        hook_reasons = f.get("hook_reasons", "")
        insight_score = f.get("insight_score", 0.0)
        text = f.get("text", "").lower()
        
        # Check if this is a mixed segment (contains both intro and good content)
        has_intro_start = any(pattern in text[:100] for pattern in [
            "yo, what's up", "hey", "hi", "hello", "what's up", 
            "it's monday", "it's tuesday", "it's wednesday", "it's thursday", "it's friday",
            "i'm jeff", "my name is", "hope you", "hope everyone"
        ])
        
        has_good_content = insight_score > 0.3 or any(pattern in text for pattern in [
            "observation", "insight", "casual drafters", "way better", "main observation",
            "key takeaway", "the thing is", "what i found", "what i learned"
        ])
        
        # If it's a mixed segment with good content, keep it but with reduced score
        if has_intro_start and has_good_content:
            # Reduce the score to account for intro content
            f["mixed_intro_penalty"] = 0.2
            f["hook_score"] = max(0.1, f.get("hook_score", 0.0) - 0.2)
            non_intro_features.append(f)
            continue
        
        # Skip pure intro content
        if "intro_greeting_penalty" in hook_reasons and not has_good_content:
            continue
            
        # Skip content with very low insight scores (likely filler) - but be less aggressive
        if insight_score < 0.05 and not has_good_content:
            continue
            
        non_intro_features.append(f)
    
    if len(non_intro_features) < 3:
        # If we filtered out too much, be less aggressive - only filter obvious intro content
        non_intro_features = [f for f in all_features if "intro_greeting_penalty" not in f.get("hook_reasons", "")]
    
    if len(non_intro_features) < 2:
        return {"error": "Episode has too much intro content, no viable clips found"}
    
    return non_intro_features

def split_mixed_segments(segments: List[Dict]) -> List[Dict]:
    """
    Split segments that contain both intro content and good content.
    This helps separate the valuable content from the filler.
    """
    split_segments = []
    
    for seg in segments:
        text = seg.get("text", "").lower()
        
        # Check if this segment contains both intro and good content
        has_intro = any(pattern in text[:100] for pattern in [
            "yo, what's up", "hey", "hi", "hello", "what's up",
            "it's monday", "it's tuesday", "it's wednesday", "it's thursday", "it's friday",
            "i'm jeff", "my name is", "hope you", "hope everyone"
        ])
        
        has_good_content = any(pattern in text for pattern in [
            "observation", "insight", "casual drafters", "way better", "main observation",
            "key takeaway", "the thing is", "what i found", "what i learned"
        ])
        
        if has_intro and has_good_content:
            # Try to find where the good content starts
            words = text.split()
            good_content_start = 0
            
            # Look for transition markers
            for i, word in enumerate(words):
                if any(pattern in " ".join(words[i:i+3]) for pattern in [
                    "observation", "insight", "casual drafters", "way better",
                    "main observation", "key takeaway", "the thing is"
                ]):
                    good_content_start = i
                    break
            
            if good_content_start > 0:
                # Split the segment
                intro_text = " ".join(words[:good_content_start])
                good_text = " ".join(words[good_content_start:])
                
                # Calculate timing split
                total_duration = seg.get("end", 0) - seg.get("start", 0)
                intro_ratio = good_content_start / len(words)
                intro_duration = total_duration * intro_ratio
                
                # Create intro segment (will be filtered out)
                intro_seg = seg.copy()
                intro_seg["text"] = intro_text
                intro_seg["end"] = seg.get("start", 0) + intro_duration
                intro_seg["is_intro"] = True
                
                # Create good content segment
                good_seg = seg.copy()
                good_seg["text"] = good_text
                good_seg["start"] = seg.get("start", 0) + intro_duration
                good_seg["is_intro"] = False
                
                split_segments.extend([intro_seg, good_seg])
            else:
                # Can't split cleanly, keep as is
                split_segments.append(seg)
        else:
            split_segments.append(seg)
    
    return split_segments

def find_viral_clips(segments: List[Dict], audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """
    Main pipeline function that pre-filters ads and finds viral clips with genre awareness.
    """
    # Auto-detect genre if not specified
    if genre == 'general':
        # Use first few segments to detect genre
        sample_text = " ".join([seg.get('text', '') for seg in segments[:3]])
        from .genres import GenreAwareScorer
        genre_scorer = GenreAwareScorer()
        detected_genre = genre_scorer.auto_detect_genre(sample_text)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print(f"Auto-detected genre: {detected_genre}")
            print("You can override this by specifying a different genre")
        genre = detected_genre
    
    # Split mixed segments to separate intro from good content
    split_segments = split_mixed_segments(segments)
    
    # Create dynamic segments based on natural boundaries and platform optimization
    dynamic_segments = create_dynamic_segments(split_segments, platform)
    
    # Compute features for all segments with genre awareness
    all_features = [compute_features_v4(seg, audio_file, genre=genre, platform=platform) for seg in dynamic_segments]
    
    # FILTER OUT ADS COMPLETELY
    non_ad_features = filter_ads_from_features(all_features)
    
    if isinstance(non_ad_features, dict) and "error" in non_ad_features:
        return non_ad_features
    
    # FILTER OUT INTRO CONTENT COMPLETELY
    non_intro_features = filter_intro_content_from_features(non_ad_features)
    
    if isinstance(non_intro_features, dict) and "error" in non_intro_features:
        return non_intro_features
    
    # Score only the non-ad, non-intro content with genre awareness
    from .scoring import score_segment_v4
    scored_clips = [score_segment_v4(f, genre=genre) for f in non_intro_features]
    
    # Sort by viral score and return top 5
    return {
        'genre': genre,
        'clips': sorted(scored_clips, key=lambda x: x["viral_score_100"], reverse=True)[:5]
    }

# Export all functions
__all__ = [
    # Main feature computation functions
    "compute_features_v4",
    "compute_features_v4_batch", 
    "compute_features",
    "compute_features_cached",
    
    # Individual feature functions
    "_hook_score",
    "_hook_score_v4",
    "_hook_score_v5",
    "_emotion_score",
    "_emotion_score_v4",
    "_payoff_presence",
    "_payoff_presence_v4",
    "_detect_payoff",
    "_info_density",
    "_info_density_v4",
    "_info_density_raw_v2",
    "_question_or_list",
    "_loopability_heuristic",
    "_arousal_score_text",
    "_audio_prosody_score",
    "_detect_insight_content",
    "_detect_insight_content_v2",
    "_apply_insight_confidence_multiplier",
    "_calculate_niche_penalty",
    "_ad_penalty",
    "_platform_length_match",
    "calculate_dynamic_length_score",
    
    # Dynamic segmentation functions
    "find_natural_boundaries",
    "create_dynamic_segments",
    
    # Pipeline functions
    "filter_ads_from_features",
    "filter_intro_content_from_features", 
    "split_mixed_segments",
    "find_viral_clips",
    
    # Explanation and analysis functions
    "_explain_viral_potential_v4",
    "_grade_breakdown",
    "_score_to_grade",
    "_heuristic_title",
    
    # Platform and tone mapping
    "PLATFORM_GENRE_MULTIPLIERS",
    "TONE_TO_GENRE_MAP",
    "PLATFORM_MAP",
    "resolve_platform",
    "resolve_genre_from_tone",
    "interpret_synergy",
    "get_genre_detection_debug",
    
    # Advanced API functions
    "find_viral_clips_with_tone",
    "find_viral_clips_with_genre",
    "find_candidates",
    
    # Question/List scoring V2
    "_question_list_raw_v2",
    "question_list_score_v2",
    "attach_question_list_scores_v2",
    
    # Phase 1 Enhanced Features
    "compute_features_v4_enhanced",
    
    # Phase 2 Enhanced Features
    "find_viral_clips_enhanced",
    
    # Phase 3 Enhanced Features
    "prosody_arousal_score",
    "payoff_guard",
    "apply_calibration",
    
    # Utility functions
    "create_segment_hash",
    "debug_segment_scoring",
    
    # Hook V5 functions
    "compute_audio_hook_modifier",
    "detect_laughter_exclamations",
    "calculate_hook_components",
    "calculate_time_weighted_hook_score",
    "score_patterns_in_text",
    "_saturating_sum",
    "_proximity_bonus",
    "_normalize_quotes_lower",
    "_first_clause",
    "_get_hook_cues_from_config",
    "_family_score",
    "_evidence_guard",
    "_anti_intro_outro_penalties",
    "_audio_micro_for_hook",
    "_sigmoid",
    "detect_podcast_genre",
]

# ============================================================================
# Finish-Thought Gate: EOS Index and Word-Boundary Functions
# ============================================================================

def likely_finished_text(text: str) -> bool:
    """
    Robust end-of-thought heuristic. Never returns a non-bool.
    """
    s = (text or "").strip().lower()
    if not s:
        return False
    if s.endswith((".", "!", "?")):
        return True
    words = s.split()
    if not words:
        return False
    # Avoid endings that are almost surely mid-thought
    bad_last = {"and", "or", "but", "because", "so", "that", "which"}
    if words[-1] in bad_last:
        return False
    # Mild guard: short fragments usually aren't a complete thought
    return len(words) >= 6

def cfg_for_finish_thought(fallback_mode: bool, platform: str) -> dict:
    """Get finish-thought configuration based on fallback mode and platform."""
    cfg = FINISH_THOUGHT_CONFIG
    
    # Get base extend time for platform
    max_extend_config = cfg["max_extend_sec"]
    if isinstance(max_extend_config, dict):
        base_extend = max_extend_config.get(platform, max_extend_config.get("default", 1.2))
    else:
        base_extend = max_extend_config
    
    if fallback_mode:
        return {
            "near_eos_tol_sec": cfg["near_eos_tol_sec"]["sparse"],
            "max_extend_sec": base_extend + cfg["max_extend_sec"].get("sparse_bonus", 0.3),
            "min_viable_after_shrink_sec": cfg["min_viable_after_shrink_sec"]["sparse"]
        }
    else:
        return {
            "near_eos_tol_sec": cfg["near_eos_tol_sec"]["normal"],
            "max_extend_sec": base_extend,
            "min_viable_after_shrink_sec": cfg["min_viable_after_shrink_sec"]["normal"]
        }

def build_eos_index(segments: List[Dict], episode_words: List[Dict] = None, episode_raw_text: str = None) -> Tuple[List[float], List[float], str]:
    """
    Build End-of-Sentence (EOS) index from transcript segments with word-level timings.
    Unifies multiple EOS sources: word timings (best), punctuation+VAD (fallback), segment boundaries (last resort).
    
    Args:
        segments: List of transcript segments with start/end times and optional words
        episode_words: Episode-level word timestamps (preferred)
        episode_raw_text: Episode raw text with punctuation intact (fallback)
        
    Returns:
        Tuple of (eos_times, word_end_times, eos_source) - EOS markers, word boundaries, and source type
    """
    eos_times = set()
    word_end_times = []
    eos_source = "none"
    
    # 1) Try episode-level word timings (best)
    episode_eos = []
    if episode_words and len(episode_words) >= 100:
        # Use existing sophisticated EOS detection
        episode_eos = _build_eos_from_words(episode_words)
        
        # Add enhanced punctuation-based detection
        from services.util import detect_sentence_endings_from_words, unify_eos_markers
        enhanced_eos = detect_sentence_endings_from_words(episode_words)
        
        if episode_eos or enhanced_eos:
            # Unify both approaches, preferring enhanced punctuation detection
            unified_eos = unify_eos_markers(list(episode_eos), enhanced_eos)
            eos_times.update(unified_eos)
            eos_source = "episode+enhanced"
            logger.info(f"EOS from episode words: {len(episode_eos)} markers, enhanced: {len(enhanced_eos)} markers, unified: {len(unified_eos)} markers")
    
    # 2) Try segment-level words (next best)
    if not eos_times:
        segment_word_count = 0
        for segment in segments:
            words = segment.get('words', [])
            if words:
                segment_eos = _build_eos_from_words(words)
                eos_times.update(segment_eos)
                segment_word_count += len(words)
        
        if eos_times:
            eos_source = "segment"
            logger.info(f"EOS from segment words: {len(eos_times)} markers from {segment_word_count} words")
    
    # 3) Fallback to punctuation+VAD (if no word-based EOS)
    if not eos_times and episode_raw_text:
        fallback_eos = _build_eos_from_punctuation_vad(episode_raw_text, segments)
        if fallback_eos:
            eos_times.update(fallback_eos)
            eos_source = "fallback"
            logger.info(f"EOS from punctuation+VAD: {len(fallback_eos)} markers")
    
    # 4) Last resort: segment boundaries
    if not eos_times:
        seg_boundary_eos = [s.get('end', 0) for s in segments if s.get('end', 0) > 0]
        if seg_boundary_eos:
            eos_times.update(seg_boundary_eos)
            eos_source = "segment_boundary"
            logger.info(f"EOS from segment boundaries: {len(seg_boundary_eos)} markers")
    
    # Build word_end_times from available words (do this early for gap fallback)
    all_words = episode_words or []
    if not all_words:
        for segment in segments:
            words = segment.get('words', [])
            if words:
                all_words.extend(words)
    
    # 5) Gap-based fallback: synthesize EOS from timing gaps when punctuation is sparse
    if len(eos_times) < _EOS_MIN_COUNT or (all_words and len(eos_times) / max(len(all_words), 1) < _EOS_MIN_DENSITY):
        gap_eos = []
        # Walk segment boundaries and add EOS where gaps are large enough
        for i in range(len(segments) - 1):
            cur_end = float(segments[i].get("end", 0.0) or 0.0)
            nxt_start = float(segments[i+1].get("start", cur_end) or cur_end)
            if nxt_start - cur_end >= _MIN_GAP_SEC:
                gap_eos.append(cur_end)
        # Merge and dedupe with any existing markers
        if gap_eos:
            # Enforce min_words_between_eos=8-10 to avoid over-splitting
            filtered_gap_eos = []
            for eos_time in gap_eos:
                # Count words between this EOS and the next one
                words_between = 0
                for word in all_words:
                    if word.get('start', 0) > eos_time:
                        words_between += 1
                        if words_between >= 8:  # min_words_between_eos
                            break
                
                if words_between >= 8:
                    filtered_gap_eos.append(eos_time)
            
            eos_times.update(filtered_gap_eos)
            eos_source = "gap_fallback"
            logger.warning(f"EOS_FALLBACK_GAPS: added {len(filtered_gap_eos)} gap-derived markers (now {len(eos_times)} total)")
    
    for word in all_words:
        word_end_times.append(word.get('end', 0))
    
    # Log final EOS density
    eos_density = len(eos_times) / max(len(word_end_times), 1) if word_end_times else 0
    logger.info(f"EOS unified: {len(eos_times)} markers, {len(word_end_times)} words, density={eos_density:.3f}, source={eos_source}")
    
    return sorted(list(eos_times)), sorted(word_end_times), eos_source

def _build_eos_from_words(words: List[Dict]) -> List[float]:
    """Build EOS markers from word-level timings with adaptive gap detection."""
    eos_times = []
    
    if not words or len(words) < 2:
        return eos_times
    
    # Calculate gap statistics for adaptive EOS detection
    gaps = []
    for i, word in enumerate(words[:-1]):
        word_end = word.get('end', 0)
        next_word = words[i + 1]
        next_start = next_word.get('start', 0)
        gap_ms = (next_start - word_end) * 1000
        if gap_ms > 0:
            gaps.append(gap_ms)
    
    # Calculate adaptive gap threshold
    GAP_EOS_MS_BASE = 280
    if gaps:
        import statistics
        median_gap = statistics.median(gaps)
        iqr_gaps = [g for g in gaps if abs(g - median_gap) <= statistics.stdev(gaps) if len(gaps) > 1]
        iqr_gap = statistics.median(iqr_gaps) if iqr_gaps else median_gap
        GAP_EOS_MS = max(GAP_EOS_MS_BASE, median_gap + iqr_gap)
    else:
        GAP_EOS_MS = GAP_EOS_MS_BASE
    
    # Discourse closers for conversational EOS
    discourse_closers = {"right?", "you know?", "that's it.", "that's right.", "all set.", "we're done."}
    
    for i, word in enumerate(words[:-1]):
        word_text = word.get('text', '').strip()
        word_end = word.get('end', 0)
        next_word = words[i + 1]
        next_start = next_word.get('start', 0)
        
        # 1) Punctuation EOS
        if word_text.endswith(('.', '?', '!', '…')):
            eos_times.append(word_end)
        
        # 2) Gap-based EOS (adaptive threshold)
        pause_ms = (next_start - word_end) * 1000
        if pause_ms >= GAP_EOS_MS:
            eos_times.append(word_end)
        
        # 3) Discourse closer EOS
        if word_text.lower() in discourse_closers:
            eos_times.append(word_end)
    
    return eos_times

def _build_eos_from_punctuation_vad(raw_text: str, segments: List[Dict]) -> List[float]:
    """Build EOS markers from punctuation and VAD segment boundaries."""
    eos_times = []
    
    if not raw_text or not segments:
        return eos_times
    
    # Find punctuation-based EOS by mapping text positions to time
    text_pos = 0
    for segment in segments:
        seg_text = segment.get('text', '')
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)
        
        # Find sentence endings in this segment
        for i, char in enumerate(seg_text):
            if char in '.!?':
                # Map character position to time within segment
                char_ratio = i / max(len(seg_text), 1)
                char_time = seg_start + (seg_end - seg_start) * char_ratio
                eos_times.append(char_time)
        
        text_pos += len(seg_text) + 1  # +1 for space
    
    # Add VAD segment boundaries as EOS
    for segment in segments:
        seg_end = segment.get('end', 0)
        if seg_end > 0:
            eos_times.append(seg_end)
    
    return eos_times

def build_eos_fallback(segments: List[Dict]) -> Tuple[List[float], List[float]]:
    """Fallback EOS building using punctuation and VAD (legacy function)."""
    # This is now handled by the unified build_eos_index function
    return [], []


def next_eos_after(time: float, eos_times: List[float]) -> Optional[float]:
    """Find next EOS after given time using binary search."""
    i = bisect_left(eos_times, time)
    return eos_times[i] if i < len(eos_times) else None

def prev_eos_before(time: float, eos_times: List[float]) -> Optional[float]:
    """Find previous EOS before given time using binary search."""
    i = bisect_left(eos_times, time) - 1
    return eos_times[i] if i >= 0 else None

def near_eos(time: float, eos_times: List[float], tolerance: float = 0.25) -> bool:
    """Check if time is within tolerance of any EOS."""
    for eos in eos_times:
        if abs(time - eos) <= tolerance:
            return True
    return False

def snap_to_last_word_end(time: float, word_end_times: List[float]) -> float:
    """Snap time to the last word end before or at the given time."""
    i = bisect_left(word_end_times, time)
    if i > 0:
        return word_end_times[i - 1]
    elif i < len(word_end_times):
        return word_end_times[i]
    else:
        return time

def ends_on_eos(end_s: float, eos_list: List[float], tol_ms: float = 120) -> bool:
    """Check if clip ends within tolerance of an EOS marker."""
    if not eos_list:
        return False
    delta = min(abs(end_s - t) for t in eos_list)
    return (delta * 1000.0) <= tol_ms

def finish_thought_normalize(candidate: Dict, eos_times: List[float], word_end_times: List[float], platform: str = "default", fallback_mode: bool = False) -> Tuple[Dict, Dict]:
    """
    Normalize to a clean finish near EOS (or extend safely to one).
    Returns: (clip_dict, {"status": FTStatus, "eos_hit": bool})
    """
    text = (candidate.get("text") or "").strip()
    start = float(candidate["start"])
    end = float(candidate["end"])
    
    # Create a copy to avoid modifying original
    c = candidate.copy()
    
    # Tolerances
    near_tol = 0.30 if not fallback_mode else 0.50
    max_extend = 1.25 if platform in ("yt", "yt_shorts") else 0.80
    
    # Find nearest EOS at/after current end
    eos = [t for t in (eos_times or []) if t >= start - 0.01]
    hit_status: FTStatus = "unresolved"
    eos_hit = False
    
    if eos:
        # small extend if needed
        next_eos = min((t for t in eos if t >= end), default=None)
        if next_eos is not None:
            delta = next_eos - end
            if delta <= near_tol:
                end = next_eos
                eos_hit = True
            elif 0 < delta <= max_extend:
                end = next_eos
                eos_hit = True
    
    if eos_hit:
        c["end"] = float(end)
        c["ft_status"] = "sparse_finished" if fallback_mode else "finished"
        c["finished_thought"] = 1
        c["finish_reason"] = "eos"
        hit_status = c["ft_status"]
    else:
        # Fallback to text-based detection
        if fallback_mode and likely_finished_text(text):
            c["ft_status"] = "sparse_finished"
            c["finished_thought"] = 1
            c["finish_reason"] = "text"
            hit_status = "sparse_finished"
        else:
            c["ft_status"] = "unresolved"
            c["finished_thought"] = 0
            c["finish_reason"] = "unresolved"
            hit_status = "unresolved"
    
    c["duration"] = c["end"] - c["start"]
    return c, {"status": hit_status, "eos_hit": eos_hit}

def prefer_finished_sibling(candidate: Dict, siblings: List[Dict]) -> Dict:
    """
    Prefer a longer finished sibling if current candidate is unresolved.
    
    Args:
        candidate: Current candidate that may be unresolved
        siblings: List of sibling candidates from same seed
        
    Returns:
        Best candidate (current or preferred sibling)
    """
    if candidate.get('finished_thought', 0) == 1:
        return candidate
    
    # Check if this looks like a question/conjunction that needs resolution
    text = candidate.get('text', '').strip()
    needs_resolution = (
        text.endswith('?') or 
        any(text.endswith(conj) for conj in ['and', 'but', 'so', 'because', 'that\'s why']) or
        candidate.get('resolution_delta', 0) > 0.12
    )
    
    if not needs_resolution:
        return candidate
    
    # Find longer siblings (18-21s) that are finished
    finished_long_siblings = [
        s for s in siblings 
        if (s.get('finished_thought', 0) == 1 and 
            18.0 <= s.get('duration', 0) <= 21.0 and
            s.get('end', 0) - s.get('start', 0) >= 18.0)
    ]
    
    if finished_long_siblings:
        # Pick best by (pl_v2 desc, duration desc, score desc)
        best = max(finished_long_siblings, key=lambda x: (
            x.get('platform_length_score_v2', 0),
            x.get('duration', 0),
            x.get('final_score', 0)
        ))
        logger.info(f"PREFERRED_SIBLING: {candidate.get('id', 'unknown')} -> {best.get('id', 'unknown')} (finished long)")
        return best
    
    return candidate

def pick_finished_sibling(variants: List[Dict], platform: str, platform_budget_end: float) -> Dict:
    """
    Pick the longest finished sibling that fits within platform budget.
    Used in fallback mode to prefer longer clips that finish properly.
    """
    if not variants:
        return None
    
    # 1) Keep only variants that likely finish (punctuation/phrase)
    finished = [v for v in variants if v.get("finished_thought") == 1]
    if not finished:
        return None
    
    # 2) Prefer the longest that stays within budget and >= MIN_FLOOR
    MIN_FLOOR = 7.5
    finished = [v for v in finished 
                if (v.get("end", 0) - v.get("start", 0)) >= MIN_FLOOR - 0.05
                and v.get("end", 0) <= platform_budget_end + 1e-3]
    
    if not finished:
        return None
    
    # Return the longest finished variant
    return max(finished, key=lambda v: v.get("end", 0) - v.get("start", 0))

def apply_family_aware_preference(candidates: List[Dict]) -> List[Dict]:
    """
    Apply family-aware preference to ensure longer finished siblings are preferred.
    
    Args:
        candidates: List of all candidates with family_id set
        
    Returns:
        List of candidates with family preference applied
    """
    from collections import defaultdict
    
    # Group by family_id
    by_family = defaultdict(list)
    for c in candidates:
        family_id = c.get("family_id", c.get("id", "unknown"))
        by_family[family_id].append(c)
    
    # Apply preference within each family
    preferred = []
    for family_id, group in by_family.items():
        if len(group) == 1:
            preferred.append(group[0])
        else:
            # Sort by: finished_thought first, then longer length, then score
            group.sort(key=lambda x: (
                int(not x.get("finished_thought", 0)),  # 0 if finished (prefer), 1 otherwise
                -x.get("duration", 0),                   # longer first
                -x.get("fs", 0)                          # then score
            ))
            # Keep only the best from each family
            preferred.append(group[0])
    
    return preferred


def extend_to_coherent_end(seg, eos_times, *, max_extra=5.0, min_pause=0.45, LOG=None):
    """
    Idempotently extend seg['t2'] to a near EOS/pause within +max_extra.
    - seg: dict with 't1','t2', optional seg['meta'] dict
    - eos_times: sorted list[float] of EOS markers (sec)
    Returns: (updated_seg, extended_bool)
    """
    t1 = float(seg.get("t1", 0.0))
    t2 = float(seg.get("t2", 0.0))
    meta = seg.setdefault("meta", {})
    if meta.get("coherent_extended"):  # already extended
        return seg, False

    if not eos_times:
        return seg, False

    i = bisect_left(eos_times, t2)
    # candidate EOS within window
    nxt = eos_times[i] if i < len(eos_times) else None
    if nxt is None or (nxt - t2) > max_extra:
        return seg, False

    # optional: require a minimum pause (if you store pause confidences, you can check them)
    new_t2 = float(nxt)
    if new_t2 <= t2:
        return seg, False

    seg["t2"] = new_t2
    meta["coherent_extended"] = True
    if LOG:
        LOG.info("EXTEND_COHERENT: +%.2fs to t2=%.2f (from %.2f)", new_t2 - t2, new_t2, t2)
    return seg, True
