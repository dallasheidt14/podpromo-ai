"""
ClipScore Service - Glue layer between secret_sauce and the rest of the system.
This service orchestrates the clip scoring pipeline using the proprietary algorithms.
"""

import numpy as np
import logging
import re
import random
import math
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from math import sqrt
from distutils.util import strtobool
from models import AudioFeatures, TranscriptSegment, MomentScore
from config.settings import (
    UPLOAD_DIR, OUTPUT_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    PRERANK_ENABLED, TOP_K_RATIO, TOP_K_MIN, TOP_K_MAX, STRATIFY_ENABLED, 
    SAFETY_KEEP_ENABLED, COMPARE_SCORING_MODES, PRERANK_WEIGHTS,
    DURATION_TARGET_MIN, DURATION_TARGET_MAX, CLIP_LEN_MIN, CLIP_LEN_MAX,
    FEATURE_SEMANTIC_END_BONUS, SEED_START_PAD_SEC
)
from config import settings

# Payoff cues for semantic end bonus
_PAYOFF_CUES = (
    "that's why", "that's why", "so that's", "so that's", "and that's", "and that's",
    "which means", "this shows", "proves that", "the point is", "bottom line",
    "in the end", "finally", "as a result", "so in short", "long story short"
)

# Defensive null-safety helpers for enhanced selector
def _nz(x, default=0.0) -> float:
    try:
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def _ensure_clip_fields(c: dict) -> dict:
    c.setdefault("features", {})
    c["start"]    = _nz(c.get("start", c.get("start_time", 0.0)))
    c["end"]      = _nz(c.get("end",   c.get("end_time",   c["start"])))
    c["duration"] = max(0.0, c["end"] - c["start"])
    c["text"]     = (c.get("text") or "").strip()
    c["hook_score"]  = _nz(c.get("hook_score", 0.0))
    c["info_score"]  = _nz(c.get("info_score", 0.0))
    pf = c.get("features", {}).get("payoff_score")
    c["features"]["payoff_score"] = _nz(pf, 0.0)
    c["pl_v2"] = _nz(c.get("pl_v2", 0.0))
    return c

def _blend_final_score(c: dict) -> float:
    s  = _nz(c.get("final_score", 0.0))
    h  = _nz(c.get("hook_score",  0.0))
    pf = _nz(c.get("features", {}).get("payoff_score", 0.0))
    if s == 0.0:
        s = 0.5*h + 0.5*pf
    c["final_score"] = float(s)
    return c["final_score"]

# Word schema adapters (normalize different transcript word formats)
def _w_start(w) -> float:
    if "start" in w: return float(w["start"])
    if "ts"    in w: return float(w["ts"])
    if "t0"    in w: return float(w["t0"]) / 1000.0
    if "s"     in w: return float(w["s"])
    raise KeyError("word start time missing")

def _w_end(w) -> float:
    if "end" in w: return float(w["end"])
    if "te"  in w: return float(w["te"])
    if "t1"  in w: return float(w["t1"]) / 1000.0
    if "e"   in w: return float(w["e"])
    if "dur" in w: return _w_start(w) + float(w["dur"])
    raise KeyError("word end time missing")

def _w_text(w) -> str:
    return (w.get("w") or w.get("text") or w.get("token") or "").strip()

# Phase 3: Precision patches
FILLER_RE = re.compile(r"^(uh|um|well|so|you know)[\s,]+", re.I)

def soft_start_pad_and_trim(seg, words, pad=0.45, max_trim=0.6):
    """Apply gentle filler trim + warm-in padding after boundary refinement"""
    start = max(seg['start'] - pad, 0.0)
    lead_text = _slice_words_text(words, start, start + 1.2)
    if FILLER_RE.match(lead_text.lower()):
        start = min(start + max_trim, seg['end'] - 0.5)
    seg['start'] = start
    return seg

def jaccard(a, b):
    """Calculate Jaccard similarity between two texts"""
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

def dedupe_keep_best(ranked):
    """Keep best scoring clip among near-duplicate texts"""
    kept, out = [], []
    for c in ranked:
        if all(jaccard(c['text'], k['text']) < 0.6 for k in kept):
            kept.append(c)
            out.append(c)
    return out

def last_sentence_of(text: str) -> str:
    """Extract the last sentence from text for payoff sentence"""
    import re
    SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
    parts = [p.strip() for p in SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts[-1] if parts else text.strip()

def payoff_q_a_bump(text):
    """Small payoff bump for Q→A pairs (question followed by declarative answer)"""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return 0.1 if any(s.endswith('?') for s in sents[:-1]) and sents[-1].endswith(('.', '!', '?')) else 0.0

def _semantic_end_bonus(text_tail: str) -> float:
    """Calculate semantic end bonus based on payoff cues and punctuation"""
    tl = text_tail.lower().strip()
    bonus = 0.0
    
    # Payoff cue bonus
    if any(cue in tl for cue in _PAYOFF_CUES):
        bonus += 0.25
    
    # Terminal punctuation bonus
    if tl.endswith(('.', '!', '?')):
        bonus += 0.12
    
    # Mild penalty if clearly mid-entity (proper noun/number at end)
    if re.search(r'\b([A-Z][a-z]+|[0-9]+)$', text_tail):
        bonus -= 0.15
    
    return bonus

# --- Feature toggles (safe defaults) ---
FEATURES = {
    "VIRALITY_TWEAKS_V1": True,
    "EXTEND_MAX_5S": True,
    "DEDUP_TIGHTEN_V1": True,
    "LONG_UNFINISHED_GATE": True,
    "SLICE_TRANSCRIPT_TO_AUDIO": True,
}

# --- Dynamic length support ---
# Target bands for fair scoring & normalization
BANDS = [(8, 12), (12, 20), (20, 35), (35, 60), (60, 90)]

# --- Boundary refinement ---
BOUNDARY_OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)

# Keyword patterns for boundary objective
START_HOOK_RE = re.compile(r"\b(why|how|stop|don['']?t|listen|watch|the truth|truth about|the secret|secrets|what no one|what nobody|this is why|here(?:'|')?s why|mistake|warning|avoid|if you|before you)\b", re.I)
END_CURIOSITY_RE = re.compile(r"(but|however|the catch|what you don't|what nobody|no one tells you|here(?:'|')?s why|until you|unless you|\?$|\.{3}$)", re.I)

def _band_for(dur: float) -> tuple[float, float]:
    """Get the length band for a given duration."""
    for lo, hi in BANDS:
        if lo <= dur <= hi:
            return (lo, hi)
    return BANDS[-1]

def _ep_get(ep, key, default=None):
    """Safe accessor for episode data - works with dict, Pydantic v1, and v2"""
    if isinstance(ep, dict):
        return ep.get(key, default)
    try:
        val = getattr(ep, key)
        return val if val is not None else default
    except AttributeError:
        pass
    try:
        d = ep.model_dump()  # pydantic v2
        return d.get(key, default)
    except Exception:
        try:
            d = ep.dict()  # pydantic v1
            return d.get(key, default)
        except Exception:
            return default

# ---- word helpers ------------------------------------------------------------

from bisect import bisect_right
import bisect
import math

def _episode_words(episode):
    """
    Returns a normalized list of words each shaped as {t, d, w}.
    Works for dict episodes and pydantic Episode models.
    """
    words = None
    if hasattr(episode, "words"):
        words = getattr(episode, "words")
    elif isinstance(episode, dict):
        words = episode.get("words")
    if not words:
        return []

    out = []
    for w in words:
        # Format A: {t, d, w}
        if "t" in w:
            t = float(w["t"])
            d = float(w.get("d", 0.0))
            text = w.get("w") or w.get("text", "")
            out.append({"t": t, "d": max(0.0, d), "w": text})
        # Format B: {t0, t1, w}
        elif "t0" in w and "t1" in w:
            t0 = float(w["t0"]); t1 = float(w["t1"])
            text = w.get("w") or w.get("text", "")
            out.append({"t": t0, "d": max(0.0, t1 - t0), "w": text})
        # Format C: {start, end, text} (ASR format)
        elif "start" in w and "end" in w:
            start = float(w["start"])
            end = float(w["end"])
            text = w.get("text") or w.get("w", "")
            out.append({"t": start, "d": max(0.0, end - start), "w": text})
        # Unknown shape → skip
    return out

def _episode_words_or_empty(ep):
    """Try common attachment points in order; adjust to your schema."""
    return (
        getattr(ep, "words", None)  # Primary: words attached after ASR
        or getattr(ep, "normalized_words", None)  # Fallback: legacy normalized_words
        or ep.get("words") if isinstance(ep, dict) else None  # Dict access
        or ep.get("normalized_words") if isinstance(ep, dict) else None  # Dict fallback
        or []
    )

def _word_starts(words):
    starts = []
    for w in words:
        try:
            starts.append(_w_start(w))
        except KeyError:
            continue
    return starts

def _idx_at_or_before(words, t):
    starts = _word_starts(words)
    i = bisect_right(starts, t) - 1
    return max(-1, min(i, len(words) - 1))

def _slice_words_text(words, t0, t1):
    if not words:
        return ""
    i = max(0, _idx_at_or_before(words, t0))
    acc = []
    for j in range(i, len(words)):
        try:
            wt = _w_start(words[j])
            if wt > t1:
                break
            acc.append(_w_text(words[j]))
        except KeyError:
            continue
    return " ".join(acc).strip()

# Back-compat shims (only if other modules still import these)
def _normalized_words_for_refine(ep): 
    return _episode_words(ep)

def _w_text(w): 
    return w.get("w", "")

def _w_start(w): 
    return float(w.get("t", w.get("t0", 0.0)))

def _w_end(w):
    t0 = _w_start(w)
    if "d" in w: 
        return t0 + float(w["d"])
    if "t1" in w: 
        return float(w["t1"])
    return t0

def _w_has_times(w): 
    return ("t" in w) or ("t0" in w and "t1" in w)

# ---- boundary cues -----------------------------------------------------------

_TERMINAL = (".", "!", "?", "…")
_CURIOSITY_PHRASES = (
    "but", "however", "here's why", "the catch", "what you don't", "what no one", "?", "…"
)

def _has_terminal_near(words, t, radius=0.25):
    """Is there a word ending with terminal punctuation within ±radius?"""
    if not words:
        return False
    i = _idx_at_or_before(words, t)
    for j in range(max(0, i-2), min(len(words), i+3)):
        w = words[j].get("w", "").strip().lower()
        if not w:
            continue
        if any(w.endswith(p) for p in _TERMINAL):
            wt_end = words[j]["t"] + words[j].get("d", 0.0)
            if abs(wt_end - t) <= radius:
                return True
    return False

def _gap_ms_around(words, t):
    """Return the local inter-word gap in seconds around t."""
    if not words:
        return 0.0
    i = _idx_at_or_before(words, t)
    prev_end = words[i]["t"] + words[i].get("d", 0.0) if i >= 0 else 0.0
    next_start = words[i+1]["t"] if i+1 < len(words) else prev_end
    return max(0.0, next_start - prev_end)

def _ends_on_curiosity_text(text_tail: str) -> bool:
    tl = text_tail.strip().lower()
    return any(k in tl for k in _CURIOSITY_PHRASES)

def _looks_like_boundary(t, episode):
    """Simple heuristic: big gap or terminal punctuation near t."""
    words = _episode_words(episode)
    if not words:
        return True  # be permissive if we don't have words
    if _gap_ms_around(words, t) >= 0.22:
        return True
    return _has_terminal_near(words, t, radius=0.25)

# ---- hill climb --------------------------------------------------------------

try:
    # reuse your centralized constants
    from backend.config.settings import CLIP_LEN_MIN, CLIP_LEN_MAX
except Exception:
    CLIP_LEN_MIN, CLIP_LEN_MAX = 8.0, 90.0

def _boundary_score_start(t, t0_orig, words, window):
    """Score a candidate START time."""
    # gap bonus (favor natural silence)
    gap = _gap_ms_around(words, t)
    gap_bonus = min(1.0, gap / 0.22)  # 0..1 once >=220ms

    # punctuation bonus
    punct = 1.0 if _has_terminal_near(words, t, 0.25) else 0.0

    # small lexical hook bonus in first 2s if we land right before a strong word
    first2 = _slice_words_text(words, t, t + 2.0)[:120].lower()
    hookish = any(h in first2 for h in ("why", "how", "stop", "the secret", "here's", "watch", "don't", "never"))
    hook_bonus = 0.25 if hookish else 0.0

    # context bonus - favor including important context words
    context_text = _slice_words_text(words, t, t + 8.0)[:200].lower()
    context_bonus = 0.0
    if any(phrase in context_text for phrase in ["ever watched", "webinar", "video that", "ramble on", "go into circles"]):
        context_bonus = 0.3

    # distance penalty (don't drift too far from original)
    dist_pen = min(0.35, 0.35 * abs(t - t0_orig) / max(0.001, window))

    return 0.5 * gap_bonus + 0.25 * punct + 0.15 * hook_bonus + 0.1 * context_bonus - dist_pen

def _boundary_score_end(t, t1_orig, words, window, start_t):
    """Score a candidate END time."""
    gap = _gap_ms_around(words, t)
    gap_bonus = min(1.0, gap / 0.22)

    punct = 1.0 if _has_terminal_near(words, t, 0.25) else 0.0

    tail_text = _slice_words_text(words, max(start_t, t - 4.0), t)[-120:]
    curiosity_bonus = 0.25 if _ends_on_curiosity_text(tail_text) else 0.0

    # Semantic end bonus (Phase 2)
    semantic_bonus = 0.0
    if FEATURE_SEMANTIC_END_BONUS:
        # Look at the last ~1.4 seconds for semantic cues
        semantic_text = _slice_words_text(words, max(start_t, t - 1.4), t)
        semantic_bonus = _semantic_end_bonus(semantic_text)

    dist_pen = min(0.35, 0.35 * abs(t - t1_orig) / max(0.001, window))

    return 0.55 * gap_bonus + 0.35 * punct + 0.25 * curiosity_bonus + semantic_bonus - dist_pen

def refine_bounds_with_hill_climb(clip, episode, max_nudge=1.0, step=0.05):
    """
    Locally search ±max_nudge around start and end to snap to nicer word boundaries.
    Returns (new_start, new_end). Never throws; falls back to original.
    """
    try:
        s0 = float(clip["start"]); e0 = float(clip["end"])
        dur0 = max(0.0, e0 - s0)
        if dur0 < CLIP_LEN_MIN:
            return s0, e0

        words = _episode_words(episode)
        if not words:
            return s0, e0

        # --- search start
        best_s, best_sscore = s0, -1e9
        s_lo = max(0.0, s0 - max_nudge)
        s_hi = min(e0 - CLIP_LEN_MIN, s0 + max_nudge)
        t = s_lo
        while t <= s_hi + 1e-9:
            sc = _boundary_score_start(t, s0, words, max_nudge)
            if sc > best_sscore:
                best_sscore = sc
                best_s = t
            t += step

        # --- search end (respect clamped duration limits)
        best_e, best_escore = e0, -1e9
        e_lo = max(best_s + CLIP_LEN_MIN, e0 - max_nudge)
        e_hi = min(best_s + CLIP_LEN_MAX, e0 + max_nudge)
        t = e_lo
        while t <= e_hi + 1e-9:
            sc = _boundary_score_end(t, e0, words, max_nudge, best_s)
            if sc > best_escore:
                best_escore = sc
                best_e = t
            t += step

        # final clamp + rounding
        new_s = round(max(0.0, min(best_s, best_e - CLIP_LEN_MIN)), 2)
        new_e = round(min(best_s + CLIP_LEN_MAX, max(best_e, new_s + CLIP_LEN_MIN)), 2)
        return new_s, new_e
    except Exception:
        # absolutely never break the pipeline
        return clip["start"], clip["end"]

# ---- clean start/end guards -------------------------------------------------

_TERMINALS = {'.','!','?','…'}
_START_DANGLERS = {'and','but','so','or','because','which','that','like','um','uh'}

def _is_terminal_char(ch: str) -> bool:
    return ch and ch[-1] in _TERMINALS

def _gap_after(words, i) -> float:
    """gap from end of words[i] to start of words[i+1]"""
    if i < 0 or i+1 >= len(words): return 9999.0
    end_i = _w_end(words[i])
    start_next = _w_start(words[i+1])
    return max(0.0, start_next - end_i)

def _gap_before(words, i) -> float:
    """gap from end of words[i-1] to start of words[i]"""
    if i <= 0 or i >= len(words): return 9999.0
    end_prev = _w_end(words[i-1])
    start_i = _w_start(words[i])
    return max(0.0, start_i - end_prev)

def _nearest_boundary_backward(words, t, max_seek=1.5, min_gap=0.35):
    """walk left to an EOS char or a long enough gap"""
    i = max(0, max(range(len(words)), key=lambda k: _w_start(words[k]) <= t and _w_start(words[k]) or -1))
    best = t
    anchor = t
    while i > 0 and (anchor - _w_start(words[i])) <= max_seek:
        token = _w_text(words[i])
        if _is_terminal_char(token.strip()[-1:]) or _gap_before(words, i) >= min_gap:
            best = _w_start(words[i])
            break
        i -= 1
    return best

def _nearest_boundary_forward(words, t, max_seek=2.0, min_gap=0.40):
    """walk right to an EOS char or a long enough gap"""
    i = min(len(words)-1, min(range(len(words)), key=lambda k: _w_start(words[k]) >= t and _w_start(words[k]) or 10**9))
    best = t
    anchor = t
    while i < len(words)-1 and (_w_start(words[i]) - anchor) <= max_seek:
        token = _w_text(words[i])
        if _is_terminal_char(token.strip()[-1:]) or _gap_after(words, i) >= min_gap:
            end_i = _w_end(words[i])
            best = end_i
            break
        i += 1
    return best

def _word_start(w):
    return float(w.get("t", w.get("start", math.nan)))

def _word_end(w):
    if "end" in w:
        return float(w["end"])
    t = w.get("t", w.get("start"))
    if t is None:
        return math.nan
    d = w.get("d", w.get("duration"))
    return float(t) + float(d) if d is not None else float(t)

def _extract_sorted_times(words):
    ts = []
    for w in words or []:
        t = _word_start(w)
        if not math.isnan(t):
            ts.append(t)
    ts.sort()
    return ts

def _clean_start_end(bounds, words, *, head_pad=0.05, tail_pad=0.25):
    """Return (start, end) refined against word boundaries. Safe on empty words."""
    try:
        s = float(bounds["start"]); e = float(bounds["end"])
    except Exception:
        return bounds.get("start", 0.0), bounds.get("end", 0.0)

    if not words:
        # Nothing to refine against; keep as-is.
        return s, e

    times = _extract_sorted_times(words)
    if not times:
        return s, e

    # START → previous word start (then tiny head pad)
    i = bisect.bisect_right(times, s) - 1
    if i >= 0:
        s = max(0.0, times[i] - head_pad)

    # END → next word end (then tiny tail pad)
    j = bisect.bisect_left(times, e)  # first word starting at/after e
    if j >= len(times):
        j = len(times) - 1
    end_word_end = _word_end(words[j]) if j >= 0 else math.nan
    if math.isnan(end_word_end):
        # Fallback: use the last word's start if end is missing
        end_word_end = times[min(j, len(times)-1)]
    e = max(e, end_word_end) + tail_pad

    # Ensure we maintain a positive duration
    if e <= s:
        e = s + 0.5

    # Small rounding is OK for UI / storage
    return round(s, 2), round(e, 2)

def _fmt_tc(seconds: float) -> str:
    """Format seconds as timecode (M:SS or H:MM:SS)"""
    s = int(round(max(0.0, seconds)))
    h, m = divmod(s // 60, 60)
    ss = s % 60
    return f"{h}:{m:02d}:{ss:02d}" if h else f"{m}:{ss:02d}"

def _fmt_ts(sec: float) -> str:
    """Format seconds as simple timestamp (M:SS)"""
    m = int(sec // 60)
    s = int(round(sec % 60))
    return f"{m}:{s:02d}"

def _nudge_to_punct_or_gap(end_time: float, words: list, max_nudge: float, min_silence: float) -> float:
    """Nudge end time to nearest punctuation or word gap within max_nudge"""
    if not words:
        return end_time
    
    # Find the word that contains or is closest to end_time
    best_end = end_time
    best_score = 0.0
    
    for i, word in enumerate(words):
        word_start = word.get("t", 0.0)
        word_end = word_start + word.get("d", 0.0)
        
        # Check if this word is within nudge range
        if abs(word_end - end_time) <= max_nudge:
            # Check for terminal punctuation
            text = word.get("w", "").strip()
            if text and text[-1] in ".!?":
                # Prefer punctuation endings
                if abs(word_end - end_time) <= max_nudge:
                    best_end = word_end
                    best_score = 1.0
                    break
            
            # Check for word gaps (silence between words)
            if i < len(words) - 1:
                next_start = words[i + 1].get("t", word_end)
                gap = next_start - word_end
                if gap >= min_silence and abs(word_end - end_time) <= max_nudge:
                    if gap > best_score:
                        best_end = word_end
                        best_score = gap
    
    return best_end

def _compute_total_duration(episode, base_segments, clips) -> float:
    """Compute total duration robustly from multiple sources."""
    # 1) Prefer explicit episode.duration if you have it
    dur = float(getattr(episode, "duration", 0.0) or 0.0)

    # 2) Fallback to words max end
    if dur <= 0.0:
        words = getattr(episode, "words", None) or []
        if words:
            try:
                dur = max(float(w.get("end", 0.0)) for w in words)
            except Exception:
                pass

    # 3) Fallback to base segments max end
    if dur <= 0.0 and base_segments:
        try:
            dur = max(float(s["end"]) for s in base_segments if "end" in s)
        except Exception:
            pass

    # 4) Fallback to clips max end (last resort)
    if dur <= 0.0 and clips:
        try:
            dur = max(float(c["end"]) for c in clips if "end" in c)
        except Exception:
            pass

    return max(0.0, dur)



def _robust_center(vals: list[float]) -> tuple[float, float]:
    """median + IQR, safe for tiny sets"""
    if not vals:
        return 0.0, 1.0
    vs = sorted(vals)
    n = len(vs)
    med = vs[n // 2]
    q1 = vs[n // 4]
    q3 = vs[(3 * n) // 4]
    iqr = max(1e-6, (q3 - q1))
    return float(med), float(iqr)

def _get_dur(c: dict) -> float:
    """Get duration from a candidate dict."""
    if "duration" in c:
        return float(c["duration"])
    return float(c.get("end", 0) - c.get("start", 0))

def _norm_or_sigmoid(raw: float, lo: float, hi: float, floor: float = 0.05) -> float:
    """Normalize with sigmoid fallback when variance is too low"""
    import math
    span = hi - lo
    if span < 1e-6:
        # Calibrated around a typical raw hook mean (tweak 0.35/0.08 to your corpus)
        sig = 1.0 / (1.0 + math.exp(-(raw - 0.35) / 0.08))
        return max(floor, min(1.0, sig))
    norm = (raw - lo) / span
    return max(floor, min(1.0, norm))

def _normalize_display_by_band(cands: list[dict]) -> list[dict]:
    """Normalize display_score within length bands to remove short-clip bias."""
    if not cands:
        return cands
    # group by band
    groups: dict[tuple[float, float], list[dict]] = {}
    for c in cands:
        dur = _get_dur(c)
        groups.setdefault(_band_for(dur), []).append(c)

    for band, group in groups.items():
        vals = [float(c.get("display_score", c.get("final_score", 0.0))*100.0) for c in group]
        med, iqr = _robust_center(vals)
        for c in group:
            v = float(c.get("display_score", c.get("final_score", 0.0)))*100.0
            # Use sigmoid fallback for low variance
            z = _norm_or_sigmoid(v, med - iqr, med + iqr, floor=0.05)
            c["_band"] = list(band)
            c["_band_norm_display"] = z
    # prefer normalized score for ranking, but keep originals for debugging
    for c in cands:
        if "_band_norm_display" in c:
            c["rank_score"] = c["_band_norm_display"]
        else:
            c["rank_score"] = float(c.get("display_score", c.get("final_score", 0.0)))
    return cands

def _mmr_select_jaccard(items, top_k, text_key="text", lam=0.7):
    """
    Null-safe MMR with Jaccard diversity over token sets.
    items: list of dicts with 'text' (or text_key) and 'rank_score'/'raw_score'
    Returns: list[dict] of chosen items
    """
    if not items:
        return []

    # Work on copies to avoid mutating source
    pool = [dict(x) for x in items]

    def tokset(s):
        s = (s or "").lower()
        # simple whitespace tokenization is fine for our use
        return set(s.split())

    def score_of(it):
        # prefer rank_score; fall back to raw_score; else 0.0
        return float(it.get("rank_score", it.get("raw_score", 0.0)) or 0.0)

    tokens = [tokset(it.get(text_key, it.get("text", ""))) for it in pool]
    chosen, chosen_tokens = [], []

    while pool and len(chosen) < top_k:
        best_i, best_val = 0, -1e9
        for i, it in enumerate(pool):
            rel = score_of(it)
            if chosen_tokens:
                # diversity = max Jaccard similarity to anything we've picked
                t = tokens[i]
                div = max(
                    (len(t & ct) / max(1, len(t | ct)) for ct in chosen_tokens),
                    default=0.0
                )
            else:
                div = 0.0
            # standard MMR: maximize lambda*relevance - (1-lambda)*diversity
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best_val, best_i = val, i

        chosen.append(pool.pop(best_i))
        chosen_tokens.append(tokens.pop(best_i))

    return chosen

# Import gate calibration constants
from services.secret_sauce_pkg.features import STRICT_FLOOR_DELTA, BALANCED_EXTRA_DELTA, MIN_FINALS

# Quality gate thresholds
THRESHOLDS = {
    "strict": {
        "payoff_min": 0.12,   # was ~0.20; allow more clips with moderate payoff
        "ql_max": 0.78,       # allow a bit more "QL" penalty on interviews
        "hook_min": 0.08,
        "arousal_min": 0.20,
    },
    "balanced": {
        "payoff_min": 0.08,
        "ql_max": 0.85,
        "hook_min": 0.06,
        "arousal_min": 0.15,
    },
    "relaxed": {
        "payoff_min": 0.05,
        "ql_max": 0.90,
        "hook_min": 0.04,
        "arousal_min": 0.10,
    }
}

def _run_quality_gates(cands, base_floor, log_ctx):
    """
    cands: list of candidate dicts that already passed safety & finished-like policy
    base_floor: your current dynamic floor (e.g., percentile or z-score transformed threshold)
    log_ctx: your running logging/telemetry dict
    """
    import numpy as np
    
    # Extract final scores for percentile calculation
    scores = [c.get("final_score", 0.0) for c in cands if isinstance(c.get("final_score"), (int, float))]
    if not scores:
        # Nothing usable; just keep top-k as a failsafe
        return sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True)[:MIN_FINALS], log_ctx
    
    # Calculate base floor from percentile (using existing logic from quality_filters.py)
    # Duration-based percentile filtering
    micro_candidates = [c for c in cands if (c.get('end', 0) - c.get('start', 0)) < 14.0]
    normal_candidates = [c for c in cands if 14.0 <= (c.get('end', 0) - c.get('start', 0)) <= 28.0]
    
    # Use the same logic as quality_filters.py for base floor calculation
    if micro_candidates and normal_candidates:
        # Mixed duration - use weighted approach
        micro_scores = [c.get("final_score", 0.0) for c in micro_candidates]
        normal_scores = [c.get("final_score", 0.0) for c in normal_candidates]
        
        if micro_scores and normal_scores:
            # Calculate percentiles for each group
            k_micro = max(1, int(0.10 * len(micro_scores)))
            k_normal = max(1, int(0.10 * len(normal_scores)))
            
            trimmed_micro = micro_scores[:-k_micro] if len(micro_scores) > 10 else micro_scores
            trimmed_normal = normal_scores[:-k_normal] if len(normal_scores) > 10 else normal_scores
            
            micro_cutoff = float(np.percentile(trimmed_micro, 70))
            normal_cutoff = float(np.percentile(trimmed_normal, 60))
            
            # Use the higher of the two as base floor
            base_floor = max(micro_cutoff, normal_cutoff)
        else:
            # Fallback to single percentile
            k = max(1, int(0.10 * len(scores)))
            trimmed_scores = scores[:-k] if len(scores) > 10 else scores
            base_floor = float(np.percentile(trimmed_scores, 60))
    else:
        # Single duration group - use existing logic
        k = max(1, int(0.10 * len(scores)))
        trimmed_scores = scores[:-k] if len(scores) > 10 else scores
        base_floor = float(np.percentile(trimmed_scores, 60))
    
    # Cap the base floor
    base_floor = min(base_floor, 0.80)
    
    # 1) Strict gate with lowered floor
    strict_floor = base_floor + STRICT_FLOOR_DELTA
    kept_strict = [c for c in cands if c.get("final_score", 0.0) >= strict_floor]

    log_ctx["strict_floor"] = round(strict_floor, 3)
    log_ctx["strict_kept"] = len(kept_strict)

    # 2) Balanced gate
    if len(kept_strict) == 0:
        # Don't reuse the strict floor; go a bit lower for balanced
        balanced_floor = strict_floor + BALANCED_EXTRA_DELTA
        kept_balanced = [c for c in cands if c.get("final_score", 0.0) >= balanced_floor]

        # Enforce MIN_FINALS when strict is empty.
        # Still keep safety/dup constraints intact – we're only relaxing the score floor.
        if len(kept_balanced) < MIN_FINALS:
            # take top-N by final_score (already safe) up to MIN_FINALS
            # NOTE: these are still from `cands` (already safety/gate-eligible)
            kept_balanced = sorted(cands, key=lambda x: x.get("final_score", 0.0), reverse=True)[:MIN_FINALS]

        log_ctx["balanced_floor"] = round(balanced_floor, 3)
        log_ctx["balanced_kept"] = len(kept_balanced)
        log_ctx["balanced_reason"] = "strict_empty_lowered"

        return kept_balanced, log_ctx

    else:
        # Strict had items — return strict as the finals for gate stage.
        return kept_strict, log_ctx

def apply_quality_gate(candidates: List[Dict], mode: str = "strict") -> List[Dict]:
    """Apply quality gate with auto-relaxation when too few clips remain"""
    if not candidates:
        return []
    
    # Initialize logging context
    log_ctx = {}
    
    # Run the quality gates with the new logic
    kept, log_ctx = _run_quality_gates(candidates, None, log_ctx)
    
    # Log the results
    logger.info(f"QUALITY_GATE[strict]: kept={log_ctx.get('strict_kept', 0)} floor={log_ctx.get('strict_floor', 'N/A')}")
    if 'balanced_floor' in log_ctx:
        logger.info(f"QUALITY_GATE[balanced]: kept={log_ctx.get('balanced_kept', 0)} floor={log_ctx.get('balanced_floor', 'N/A')} reason={log_ctx.get('balanced_reason', 'N/A')}")
    logger.info(f"MIN_FINALS: target={MIN_FINALS} applied_when_strict_empty={log_ctx.get('strict_kept', 0) == 0}")
    
    return kept

def notch_penalty(dur: float, already_at_notch: int) -> float:
    """Notch penalty disabled for length-agnostic scoring"""
    return 0.0  # NOTCH_PENALTY_WEIGHT = 0.0

def apply_micro_jitter_to_display(dur: float) -> float:
    """Apply tiny micro-jitter to displayed duration to break identical lengths"""
    import random
    # Add ±0.4s jitter to break identical display durations
    jitter = random.uniform(-0.4, 0.4)
    return round(dur + jitter, 1)

def _get_dur(c: dict) -> float:
    """Get duration from various possible field names"""
    return float(
        c.get("duration_sec") or
        c.get("duration") or
        c.get("dur") or
        (c.get("end", 0.0) - c.get("start", 0.0)) or
        0.0
    )

def _log_finals_summary(finals: list, logger) -> None:
    """Log structured summary of final clips"""
    if not finals:
        return
    
    n = len(finals)
    
    # Count buckets
    buckets = {"short": 0, "mid": 0, "long": 0}
    durs = []
    finished_count = 0
    
    for clip in finals:
        dur = _get_dur(clip)
        durs.append(dur)
        
        # Bucket classification
        if dur < 14.0:
            buckets["short"] += 1
        elif dur <= 22.0:
            buckets["mid"] += 1
        else:
            buckets["long"] += 1
        
        # Count finished
        if clip.get("finished_thought", False):
            finished_count += 1
    
    # Calculate duration stats
    durs_sorted = sorted(durs)
    dur_stats = {
        "min": durs_sorted[0] if durs_sorted else 0,
        "p50": durs_sorted[len(durs_sorted)//2] if durs_sorted else 0,
        "p90": durs_sorted[int(len(durs_sorted)*0.9)] if durs_sorted else 0,
        "max": durs_sorted[-1] if durs_sorted else 0
    }
    
    # Count reason codes
    reason_counts = {}
    for clip in finals:
        reason = clip.get("meta", {}).get("reason", "UNKNOWN")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    # Top 3 reasons
    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    reasons_str = ", ".join([f"{k}={v}" for k, v in top_reasons])
    
    # Log structured summary (one-liner per stage)
    def _fmt_list(arr, k=6):
        s = ", ".join(str(x) for x in arr[:k])
        return f"[{s}{', …' if len(arr) > k else ''}]"
    
    durs = [round(float(c.get("end",0))-float(c.get("start",0)), 1) for c in finals]
    virs = [round(float(c.get("display_score", c.get("score", 0.0))), 3) for c in finals]
    logger.info("FINALS: n=%d | durs=%s | virality=%s", len(finals), _fmt_list(durs), _fmt_list(virs))

def _log_pick_trace(finals: list, logger) -> None:
    """Log pick trace for each final clip"""
    for i, clip in enumerate(finals):
        virality = clip.get("display_score", clip.get("score", 0.0))
        payoff = clip.get("payoff_score", 0.0)
        finished = clip.get("finished_thought", False)
        dur = _get_dur(clip)
        
        # Get reasons from meta
        reasons = []
        if clip.get("meta", {}).get("ad_pen"):
            reasons.append("ad_pen")
        if finished:
            reasons.append("finished")
        if dur < 14.0:
            reasons.append("short")
        elif dur <= 22.0:
            reasons.append("mid")
        else:
            reasons.append("long")
        
        logger.info("PICK_TRACE #%d: virality=%.3f payoff=%.3f finished=%s dur=%.1fs reasons=[%s]",
                    i+1, virality, payoff, finished, dur, ",".join(reasons))

def _anti_uniform_tiebreak(picks, reserve, window=1.0, LOG=None):
    """Window rule: if ≥50% of finalists fall within ±window seconds of the median, swap the best finished candidate outside that band"""
    LOG = LOG or logger
    if len(picks) < 3: 
        return picks
    
    durs = [float(p.get("dur", p.get("end", 0) - p.get("start", 0))) for p in picks]
    med = sorted(durs)[len(durs)//2]
    
    # Count how many are within window of median
    in_window = [i for i, d in enumerate(durs) if abs(d - med) <= window]
    
    # If ≥50% are clustered, try to diversify
    if len(in_window) >= (len(picks) + 1) // 2:
        # Find best finished candidate outside the window
        outs = [c for c in reserve
                if abs(float(c.get("dur", c.get("end", 0) - c.get("start", 0))) - med) > window
                and c.get("finished_thought", False)]
        
        if outs:
            newcomer = max(outs, key=lambda c: c.get("display_score", c.get("score", 0.0)))
            worst_i = min(in_window, key=lambda i: picks[i].get("display_score", picks[i].get("score", 0.0)))
            
            LOG.info("DIVERSITY_SWAP: swap %s (%.1fs) with %s (%.1fs) | median=%.1fs window=%.1fs",
                     picks[worst_i].get("id"), durs[worst_i],
                     newcomer.get("id"), float(newcomer.get("dur", newcomer.get("end", 0) - newcomer.get("start", 0))),
                     med, window)
            picks[worst_i] = newcomer
    
    return picks

# Length-agnostic selection: pure top-K by virality with diversity tie-breaker
EPSILON = 0.03  # Virality difference threshold for tie-breaking

def pure_topk_pick(sorted_pool, reserve, want=4, LOG=None):
    """Pure top-K by virality with diversity tie-breaker only on near-ties"""
    picks = []
    
    # Pure top-K selection by virality
    for c in sorted_pool:
        if len(picks) >= want:
            break
        picks.append(c)
    
    # If we need more, add from reserve
    if len(picks) < want:
        from services.quality_filters import filter_low_quality
        reserve_ok = filter_low_quality(reserve, mode="balanced")
        
        for c in reserve_ok:
            if len(picks) >= want:
                break
            if c not in picks:
                picks.append(c)
    
    # Apply diversity tie-breaker only on near-ties (ε=0.03)
    picks = _diversity_tiebreaker(picks, reserve_ok if 'reserve_ok' in locals() else reserve, LOG=LOG)
    
    if LOG:
        LOG.info("PICK: pure_topk; DIVERSITY_TIEBREAK applied (eps=%.2f, window=±1.0s) only on near-ties", EPSILON)
    
    return picks

def _diversity_tiebreaker(picks, reserve, LOG=None):
    """Diversity tie-breaker: only when candidates are within ε=0.03 of each other"""
    if len(picks) < 3:
        return picks
    
    # Check if we have near-ties (within ε=0.03)
    virality_scores = [c.get("display_score", c.get("score", 0.0)) for c in picks]
    max_virality = max(virality_scores)
    near_ties = [i for i, score in enumerate(virality_scores) if abs(score - max_virality) <= EPSILON]
    
    if len(near_ties) < 2:
        return picks  # No near-ties, keep original order
    
    # Check for clustering in near-tie group
    durs = [float(picks[i].get("dur", 0)) for i in near_ties]
    med = sorted(durs)[len(durs)//2]
    
    # Count how many are within ±1.0s of median
    in_window = [i for i, d in enumerate(durs) if abs(d - med) <= 1.0]
    
    # If ≥50% are clustered, try to diversify
    if len(in_window) >= (len(near_ties) + 1) // 2:
        # Find best alternative outside the window
        outs = [c for c in reserve
                if abs(float(c.get("dur", 0)) - med) > 1.0
                and c.get("finished_thought", False)
                and c not in picks]
        
        if outs:
            # Find the best alternative
            best_alt = max(outs, key=lambda c: c.get("display_score", c.get("score", 0.0)))
            
            # Find worst clustered pick to replace
            worst_i = min(in_window, key=lambda i: picks[i].get("display_score", picks[i].get("score", 0.0)))
            
            if LOG:
                LOG.info("DIVERSITY_TIEBREAK: swap %s (%.1fs) with %s (%.1fs) | median=%.1fs",
                         picks[worst_i].get("id"), durs[worst_i],
                         best_alt.get("id"), float(best_alt.get("dur", 0)), med)
            
            picks[worst_i] = best_alt
    
    return picks

def adaptive_gate(candidates: List[Dict], min_count: int = 3, gate_mode=None) -> List[Dict]:
    """Adaptive gate with salvage pass - never relax finished_thought"""
    from services.quality_filters import essential_gates
    
    # infer fallback safely from gate_mode if provided
    fallback = bool(gate_mode.get("fallback")) if isinstance(gate_mode, dict) else False
    
    # Detect fallback mode from candidates if not provided
    if not fallback:
        fallback = any(
            bool(c.get("fallback") or c.get("meta", {}).get("fallback")) or
            "sparse" in (c.get("ft_classifier", "") or "")
            for c in candidates
        )
    
    # Apply fallback-aware gates
    from config import settings
    passed, reason_counts = essential_gates(
        candidates,
        fallback=fallback,
        tail_close_sec=1.5,
        cfg=settings
    )
    
    # Add one-line summary per pool
    fallback_count = reason_counts.get("KEEP_FALLBACK_FINISHED", 0)
    logger.info(f"GATE_SUMMARY: fallback={fallback}, accepted_finished_like={fallback_count}")
    
    if len(passed) >= min_count:
        logger.info(f"ADAPTIVE_GATE: {len(passed)} passed gates (target: {min_count})")
        return passed
    
    # Salvage pass when pool==0: auto-extend top hooks to next EOS
    if len(passed) == 0:
        logger.warning("ADAPTIVE_GATE: pool=0, attempting salvage pass")
        salvaged = _salvage_pass(
            candidates,
            fallback=bool(gate_mode and gate_mode.get("fallback")),
            tail_close_sec=gate_mode.get("tail_close_sec", 0.60) if gate_mode else 1.0,
            max_extend_sec=float(os.getenv("SALVAGE_MAX_EXTEND_SEC", "4.0")),
            LOG=logger,
        )
        if salvaged:
            logger.info(f"ADAPTIVE_GATE: salvage recovered {len(salvaged)} clips")
            return salvaged
        else:
            logger.warning("ADAPTIVE_GATE: salvage failed, returning empty")
            return []
    
    # Soft floor: guaranteed minimum when conditions are met
    SOFT_FLOOR_K = int(os.getenv("SOFT_FLOOR_K", "3"))
    
    # Calculate additional conditions for soft floor
    eos_density = 0.0  # Default, will be calculated if available
    ft_ratio_sparse_ok = 0.0  # Default, will be calculated if available
    
    # Try to get eos_density from candidates metadata
    for c in candidates:
        if "eos_density" in c.get("meta", {}):
            eos_density = c["meta"]["eos_density"]
            break
    
    # Try to get ft_ratio_sparse_ok from candidates metadata  
    for c in candidates:
        if "ft_ratio_sparse_ok" in c.get("meta", {}):
            ft_ratio_sparse_ok = c["meta"]["ft_ratio_sparse_ok"]
            break
    
    should_softfloor = (
        fallback or 
        eos_density < 0.015 or 
        ft_ratio_sparse_ok >= 0.70 or 
        len(passed) == 0
    )
    
    if should_softfloor and len(passed) < SOFT_FLOOR_K:
        logger.info(f"ADAPTIVE_GATE: applying soft floor (current={len(passed)}, target={SOFT_FLOOR_K})")
        
        # Sort reserve by utility_pre_gate desc, then by hook desc, then stable by id
        reserve_sorted = sorted(
            candidates, 
            key=lambda c: (
                c.get("utility_pre_gate", c.get("display_score", 0.0)), 
                c.get("hook", 0.0), 
                c.get("id", "")
            ), 
            reverse=True
        )
        
        # Apply quality guardrails
        def soft_floor_ok(c):
            virality = c.get("display_score", c.get("utility_pre_gate", 0.0))
            hook = c.get("hook", 0.0)
            payoff_ok = c.get("payoff_ok", False)
            ad_pen = c.get("ad_pen", 0.0)
            dur = c.get("dur", c.get("end", 0) - c.get("start", 0))
            
            return (
                (virality >= 0.55 or (hook >= 0.60 and payoff_ok)) and 
                ad_pen == 0 and 
                8.0 <= dur <= 30.0
            )
        
        pulled = [c for c in reserve_sorted if soft_floor_ok(c)]
        need = SOFT_FLOOR_K - len(passed)
        added = min(len(pulled), need)
        
        if added > 0:
            passed.extend(pulled[:added])
            logger.info(f"ADAPTIVE_GATE: soft floor added {added} clips (fallback={fallback}, eos_density={eos_density:.3f})")
    
    # Return fewer rather than auto-fill with failed candidates
    if len(passed) < min_count:
        logger.warning(f"ADAPTIVE_GATE: only {len(passed)} passed gates (target: {min_count}) - returning fewer")
    
    return passed

def apply_finish_adjustments(utility: float, dur: float, finished: bool, fallback_mode: bool) -> float:
    """Apply finish bonuses and penalties to utility score"""
    # bonus for actually finishing a thought
    if finished:
        utility += 0.15
        return utility

    # unfinished penalties (fallback-aware)
    if dur >= 18.0:
        utility += (-0.06 if fallback_mode else -0.12)
    return utility

def pre_gate_finish_normalization(cands, eos_times, flags, gate_mode, LOG):
    """Pre-gate extension + long-unfinished guard"""
    if not flags.get("EXTEND_MAX_5S", False):
        return

    from services.secret_sauce_pkg.features import extend_to_coherent_end

    for c in cands:
        dur = float(c.get("dur", 0.0))
        finished = bool(c.get("finished_thought", False))
        payoff = float(c.get("features", {}).get("payoff", 0.0))

        # 1) Attempt a single coherent extension for long unfinished
        if dur >= 22.0 and not finished:
            seg = {"t1": c["t1"], "t2": c["t2"], "meta": c.setdefault("meta", {})}
            _, ext = extend_to_coherent_end(seg, eos_times, max_extra=5.0, LOG=LOG)
            if ext:
                # update candidate window + duration
                c["t2"] = seg["t2"]
                c["dur"] = float(c["t2"]) - float(c["t1"])
                # mark finished if your classifier re-check says so (cheap heuristic):
                c["finished_thought"] = True  # or set after a quick tail-check

        # 2) Optional hard-ish gate: still long & unfinished with low payoff? mark reject
        if flags.get("LONG_UNFINISHED_GATE", False):
            cutoff = 0.45 if not gate_mode.get("fallback", False) else 0.40
            if dur >= 22.0 and (not c.get("finished_thought", False)) and payoff < cutoff:
                c.setdefault("reject_reasons", []).append("LONG_UNFINISHED_LOW_PAYOFF")

def slice_words_to_window(words, t1, t2):
    """Slice words to audio window to prevent text bleeding past audio"""
    # words: list of {"t": float, "w": "token"}
    out = []
    for w in words or []:
        tw = float(w.get("t", 0.0))
        if t1 - 1e-3 <= tw <= t2 + 1e-3:
            out.append(w.get("w", ""))
    return out

def _segmentation_preflight(transcript_segments, LOG):
    """
    Normalizes transcript segments so dynamic segmentation never sees None.
    - Ensures: words is a list, text is str (may be ""), and adds derived tokens
    """
    norm = []
    none_word_ids = []
    coerced_text_ids = []
    wrong_type_ids = []

    for i, s in enumerate(transcript_segments or []):
        if not s:
            continue
        seg = dict(s)  # shallow copy is enough
        seg.setdefault("id", i)
        
        # words
        words = seg.get("words")
        if words is None:
            none_word_ids.append(seg["id"])
            words = []
        elif not isinstance(words, list):
            wrong_type_ids.append(seg["id"])
            words = list(words)
        seg["words"] = words

        # text
        text = seg.get("text")
        if text is None:
            coerced_text_ids.append(seg["id"])
            # build from words if needed
            text = " ".join(w.get("w", "") if isinstance(w, dict) else str(w) for w in words) if words else ""
        elif not isinstance(text, str):
            coerced_text_ids.append(seg["id"])
            text = str(text)
        seg["text"] = text

        # derived tokens (never None)
        seg["tokens"] = (seg["text"].split() if seg["text"] else [])
        norm.append(seg)

    # emit compact warnings
    if none_word_ids:
        LOG.warning(
            "DYNSEG: %d segments had words=None; coerced to [] (first few ids=%s)",
            len(none_word_ids), none_word_ids[:5]
        )
    if wrong_type_ids:
        LOG.warning(
            "DYNSEG: %d segments had non-list words; coerced via list() (first few ids=%s)",
            len(wrong_type_ids), wrong_type_ids[:5]
        )
    if coerced_text_ids:
        LOG.warning(
            "DYNSEG: %d segments had non-str/None text; coerced/synthesized (first few ids=%s)",
            len(coerced_text_ids), coerced_text_ids[:5]
        )
    
    return norm

def near_time_dedup(cands, LOG):
    """Tighter duplicate filtering with near-time rule"""
    DEDUP_BY_TEXT_THRESH = 0.85 if FEATURES.get("DEDUP_TIGHTEN_V1", False) else 0.90
    NEAR_TIME_WINDOW_S = 12.0
    NEAR_TIME_OVERLAP = 0.75
    
    # assumes each c has t1,t2,utility
    cands = sorted(cands, key=lambda x: x.get("utility", 0.0), reverse=True)
    kept = []
    for c in cands:
        mid = 0.5 * (float(c["t1"]) + float(c["t2"]))
        ok = True
        for k in kept:
            kmid = 0.5 * (float(k["t1"]) + float(k["t2"]))
            if abs(mid - kmid) <= NEAR_TIME_WINDOW_S:
                # temporal overlap ratio
                a1,a2 = float(c["t1"]), float(c["t2"])
                b1,b2 = float(k["t1"]), float(k["t2"])
                inter = max(0.0, min(a2,b2) - max(a1,b1))
                union = max(a2,a1) - min(b1,a1) if b2>=a2 else (b2 - min(a1,b1))
                overlap = inter / max(1e-6, (a2-a1))
                if overlap >= NEAR_TIME_OVERLAP:
                    ok = False
                    break
        if ok:
            kept.append(c)
    LOG.info("DEDUP_NEARTIME: %d -> %d kept", len(cands), len(kept))
    return kept

# --- helper: unwrap outputs from _extend_to_eos ------------------------------
def _unwrap_extended(ext):
    # _extend_to_eos might return a dict, or (dict, meta), or a string/None on failure
    if ext is None:
        return None
    if isinstance(ext, dict):
        return ext
    if isinstance(ext, (list, tuple)):
        return ext[0] if ext and isinstance(ext[0], dict) else None
    # strings or anything else -> invalid
    return None

def unique_by_id(clips):
    """Remove duplicate clips by ID, preserving order"""
    seen = set()
    unique = []
    for clip in clips:
        clip_id = clip.get("id")
        if clip_id and clip_id not in seen:
            seen.add(clip_id)
            unique.append(clip)
    return unique

def dedup_by_timing(clips, time_window=0.15):
    """Remove clips with similar start/end times within time_window seconds"""
    if len(clips) <= 1:
        return clips
    
    unique = []
    for clip in clips:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", 0))
        
        # Check if this clip overlaps significantly with any already selected clip
        is_duplicate = False
        for existing in unique:
            existing_start = float(existing.get("start", 0))
            existing_end = float(existing.get("end", 0))
            
            # Check for temporal overlap within time_window
            if (abs(start - existing_start) <= time_window and 
                abs(end - existing_end) <= time_window):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(clip)
    
    return unique

def tighten_selection(clips):
    """Tighten selection after gating - drop unfinished clips unless they meet quality thresholds"""
    if not clips:
        return clips
    
    tightened = []
    for clip in clips:
        finished = bool(clip.get("finished_thought"))
        payoff = float(clip.get("payoff", 0.0))
        hook = float(clip.get("hook", 0.0))
        tail_gap = float(clip.get("tail_gap_sec", 999.0))
        
        # Keep finished clips
        if finished:
            tightened.append(clip)
            continue
        
        # For unfinished clips, apply stricter criteria
        if not finished:
            # Drop if (hook + payoff) < 0.28
            if (hook + payoff) < 0.28:
                continue
            
            # Drop unless payoff >= 0.25 and distance_to_next_EOS <= 0.6s
            if payoff >= 0.25 and tail_gap <= 0.6:
                tightened.append(clip)
                continue
        
        # If we get here, it's a borderline case - keep it
        tightened.append(clip)
    
    return tightened

def apply_duration_diversity(clips):
    """Apply soft duration diversity - prefer mix of short/mid/long clips"""
    if len(clips) <= 3:
        return clips  # Not enough clips to diversify
    
    # Categorize clips by duration
    short_clips = []  # 6-12s
    mid_clips = []    # 13-22s  
    long_clips = []   # 23-30s
    
    for clip in clips:
        dur = float(clip.get("dur", 0.0))
        if 6 <= dur <= 12:
            short_clips.append(clip)
        elif 13 <= dur <= 22:
            mid_clips.append(clip)
        elif 23 <= dur <= 30:
            long_clips.append(clip)
        else:
            # Other durations - add to mid by default
            mid_clips.append(clip)
    
    # Soft target mix: Short: 1-2, Mid: 2-3, Long: 0-1
    # Sort each category by virality score
    short_clips.sort(key=lambda c: c.get("display_score", c.get("score", 0.0)), reverse=True)
    mid_clips.sort(key=lambda c: c.get("display_score", c.get("score", 0.0)), reverse=True)
    long_clips.sort(key=lambda c: c.get("display_score", c.get("score", 0.0)), reverse=True)
    
    # Select diverse mix
    diverse = []
    diverse.extend(short_clips[:2])  # Up to 2 short clips
    diverse.extend(mid_clips[:3])    # Up to 3 mid clips
    diverse.extend(long_clips[:1])   # Up to 1 long clip
    
    # If we have fewer than original, add remaining high-scoring clips
    if len(diverse) < len(clips):
        all_clips = short_clips + mid_clips + long_clips
        remaining = [c for c in all_clips if c not in diverse]
        remaining.sort(key=lambda c: c.get("display_score", c.get("score", 0.0)), reverse=True)
        diverse.extend(remaining[:len(clips) - len(diverse)])
    
    return diverse

def _salvage_pass(candidates: List[Dict], *, fallback=False, tail_close_sec=1.0, max_extend_sec=4.0, LOG=None) -> List[Dict]:
    """Salvage pass: auto-extend top hooks to next EOS (≤90s), re-check finished_thought"""
    from services.quality_filters import essential_gates
    
    # Sort by hook score (top hooks first)
    top_hooks = sorted(candidates, key=lambda c: c.get("hook", 0.0), reverse=True)[:5]
    
    salvaged = []
    tried = 0
    
    for c in top_hooks:
        tried += 1
        
        # Strategy 1: End-extend to next EOS boundary (≤90s hard cap)
        try:
            extended_c = _extend_to_eos(c, max_dur=min(90.0, c.get("end", 0) + max_extend_sec))
        except Exception as e:
            if LOG: LOG.debug("SALVAGE: extend failed: %r", e)
            continue

        c_ext = _unwrap_extended(extended_c)
        if not isinstance(c_ext, dict):
            if LOG: LOG.debug("SALVAGE: extend returned non-candidate (%r)", type(extended_c))
            continue

        # Check duration after salvage - reject micro-clips
        dur = float(c_ext.get("dur", 0.0))
        MIN_FINAL_DUR = 6.0  # Platform-safe minimum duration
        
        if dur < MIN_FINAL_DUR:
            # Try extending again if still too short
            try:
                re_extended = _extend_to_eos(c_ext, max_dur=min(90.0, c_ext.get("end", 0) + 4.0))
                if isinstance(re_extended, dict):
                    c_ext = re_extended
                elif isinstance(re_extended, (list, tuple)) and re_extended:
                    c_ext = re_extended[0]
                
                dur = float(c_ext.get("dur", 0.0))
            except:
                pass
            
            # Reject if still too short
            if dur < MIN_FINAL_DUR:
                if LOG:
                    LOG.debug("SALVAGE: rejected micro-clip dur=%.1fs < %.1fs", dur, MIN_FINAL_DUR)
                continue

        from config import settings
        kept, _reasons = essential_gates([c_ext], fallback=fallback, tail_close_sec=tail_close_sec, cfg=settings)
        if kept:
            salvaged.append(kept[0])
            if LOG:
                LOG.info("SALVAGE: rescued id=%s new_end=%.2fs dur=%.1fs", c_ext.get("id"), c_ext.get("end", 0.0), dur)
            continue
        
        # Strategy 2: If ends with question, try including immediate answer (Q→A join)
        if _ends_with_question(c):
            try:
                qa_c = _join_question_answer(c, max_dur=90.0)
            except Exception as e:
                if LOG: LOG.debug("SALVAGE: Q→A join failed: %r", e)
                continue
                
            if qa_c:
                kept, _reasons = essential_gates([qa_c], fallback=fallback, tail_close_sec=tail_close_sec, cfg=settings)
                if kept:
                    salvaged.append(kept[0])
                    if LOG:
                        LOG.info("SALVAGE: Q→A rescued id=%s new_end=%.2fs", qa_c.get("id"), qa_c.get("end", 0.0))
                    continue
        
        # Strategy 3: Start-backoff to previous EOS (trim filler)
        try:
            backoff_c = _backoff_to_previous_eos(c, max_dur=90.0)
        except Exception as e:
            if LOG: LOG.debug("SALVAGE: backoff failed: %r", e)
            continue
            
        if backoff_c:
            kept, _reasons = essential_gates([backoff_c], fallback=fallback, tail_close_sec=tail_close_sec, cfg=settings)
            if kept:
                salvaged.append(kept[0])
                if LOG:
                    LOG.info("SALVAGE: backoff rescued id=%s new_end=%.2fs", backoff_c.get("id"), backoff_c.get("end", 0.0))
    
    if LOG:
        LOG.info("SALVAGE: tried=%d rescued=%d max_extend=5.0s heuristic=punct_or_boundary", tried, len(salvaged))
    return salvaged

def _finish_polish(c: dict, LOG=None) -> dict:
    """If the clip looks unfinished but an EOS/pause is close, extend and re-score."""
    try:
        finished = bool(c.get("finished_thought"))
        ft = (c.get("ft_classifier") or "").lower()
        if finished or ft in ("finished", "sparse_finished"):
            return c

        tail_close = float(c.get("tail_close_sec") or 999.0)
        if tail_close <= 5.0:
            ext = _extend_to_eos(c, max_dur=90.0)
            if isinstance(ext, dict) and ext.get("end") != c.get("end"):
                # re-score just the parts affected (payoff/loop/platform)
                _rescore_payoff_loop_platform(ext)
                if LOG:
                    LOG.info("FINISH_POLISH: +%.1fs → %s",
                             float(ext.get("end", 0) - c.get("end", 0)), ext.get("id"))
                return ext
    except Exception as e:
        if LOG:
            LOG.warning("FINISH_POLISH: skipped due to %s", e)
    return c

def _rescore_payoff_loop_platform(ext: dict):
    """Re-score only payoff/loop/platform terms after extension"""
    try:
        from services.secret_sauce_pkg.features import compute_virality
        # Recompute virality with updated duration
        start = ext.get("start", 0)
        end = ext.get("end", 0)
        pl_v2 = ext.get("platform_length_score_v2", 0.0)
        ext["display_score"] = compute_virality(ext, start, end, pl_v2)
    except Exception:
        pass

def _extend_to_eos(candidate: Dict, max_dur: float = 90.0) -> Dict:
    """Extend candidate to next EOS boundary or punctuation"""
    # Allow extend up to +4.0s (bounded by clip max)
    current_dur = candidate.get("dur", 0)
    extended_dur = min(max_dur, current_dur + 4.0)
    
    # Check if we can find EOS or punctuation in the transcript
    text = candidate.get("text", "")
    if text:
        # Look for sentence endings in the text
        sentences = text.split('.')
        if len(sentences) > 1:
            # Found a sentence boundary, extend to it
            extended_dur = min(max_dur, current_dur + 2.0)
    
    # If no EOS exists (gap fallback), extend to next segment boundary or +2.0s
    if candidate.get("fallback") or candidate.get("meta", {}).get("fallback"):
        extended_dur = min(max_dur, current_dur + 2.0)
    
    extended_c = candidate.copy()
    extended_c["dur"] = extended_dur
    extended_c["end"] = candidate.get("start", 0) + extended_dur
    extended_c["finished_thought"] = True
    extended_c["ft_classifier"] = "sparse_finished"  # Mark as sparse_finished for fallback gates
    
    # Recompute only payoff/loop subscores (don't rescore hook/arousal)
    from services.secret_sauce_pkg.features import compute_virality
    extended_c["display_score"] = compute_virality(extended_c, extended_c.get("start", 0), extended_c.get("end", 0), 0.0)
    
    return extended_c

def _ends_with_question(candidate: Dict) -> bool:
    """Check if candidate ends with a question"""
    text = candidate.get("text", "").lower()
    return text.endswith("?") or any(q in text[-20:] for q in ["what", "how", "why", "when", "where", "who"])

def _join_question_answer(candidate: Dict, max_dur: float = 90.0) -> Dict:
    """Join question with immediate answer segment"""
    # Simplified: extend by 15s for answer
    extended_dur = min(max_dur, candidate.get("dur", 0) + 15.0)
    
    qa_c = candidate.copy()
    qa_c["dur"] = extended_dur
    qa_c["end"] = candidate.get("start", 0) + extended_dur
    qa_c["finished_thought"] = True
    return qa_c

def _backoff_to_previous_eos(candidate: Dict, max_dur: float = 90.0) -> Dict:
    """Trim filler from start to previous EOS"""
    # Simplified: trim 5s from start
    trimmed_dur = max(8.0, candidate.get("dur", 0) - 5.0)
    
    backoff_c = candidate.copy()
    backoff_c["dur"] = trimmed_dur
    backoff_c["start"] = candidate.get("start", 0) + 5.0
    backoff_c["finished_thought"] = True
    return backoff_c

def enforce_soft_floor(candidates: List[Dict], min_count: int = 3) -> List[Dict]:
    """Enforce final soft floor after all gates"""
    if len(candidates) >= min_count:
        return candidates
    
    # Get all remaining candidates (non-ads)
    pool = [c for c in candidates if not c.get("is_ad", False)]
    
    # Sort by score
    pool.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
    
    # Add from pool until we reach min_count
    result = list(candidates)
    for c in pool:
        if c not in result:
            result.append(c)
        if len(result) >= min_count:
            break
    
    logger.info(f"SOFT_FLOOR: {len(candidates)} -> {len(result)} (target: {min_count})")
    return result

# --- helpers for authoritative finish-thought tracking ---
def authoritative_fallback_mode(result_meta):
    try:
        return bool(result_meta["ft"]["fallback_mode"])
    except Exception:
        # conservative default if meta missing
        return True

def normalize_ft_status(c):
    # ensure every candidate has an ft_status
    if not c.get("ft_status"):
        c["ft_status"] = "unresolved"
    return c

def _interval_iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0

def _finished_rank(meta):
    s = (meta or {}).get("ft_status")
    return 2 if s == "finished" else 1 if s == "sparse_finished" else 0

def _length_fit(start, end, target=18.0):
    return -abs((end - start) - target)

def _dedup_by_time_iou(cands, iou_thresh=0.85, target_len=18.0):
    if len(cands) <= 1: 
        return cands
    # sort: score desc, finished rank desc, length fit (closer to target) desc
    ordered = sorted(
        cands,
        key=lambda c: (
            c.get("display_score", 0.0),
            _finished_rank((c.get("meta") or {})),
            _length_fit(c["start"], c["end"], target_len),
        ),
        reverse=True,
    )
    kept = []
    for c in ordered:
        s, e = c["start"], c["end"]
        if any(_interval_iou(s, e, k["start"], k["end"]) >= iou_thresh for k in kept):
            continue
        kept.append(c)
    return kept

def _text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using simple word overlap"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize text: lowercase, remove punctuation, split into words
    import re
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0

def _dedup_by_text(cands, text_thresh=0.90):
    """Remove candidates with similar text content"""
    if len(cands) <= 1:
        return cands
    
    # Sort by score (best first)
    ordered = sorted(cands, key=lambda c: c.get("display_score", 0.0), reverse=True)
    
    kept = []
    for c in ordered:
        text = c.get("text", "") or c.get("transcript", "")
        if not text:
            kept.append(c)
            continue
            
        # Check against all kept candidates
        is_duplicate = False
        for k in kept:
            k_text = k.get("text", "") or k.get("transcript", "")
            if k_text and _text_similarity(text, k_text) >= text_thresh:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(c)
    
    return kept

def _format_candidate(seg):
    """Format a single segment into a candidate with ft_status propagation"""
    c = {
        "id": seg.get("id"),
        "start": float(seg.get("start", 0)),
        "end": float(seg.get("end", 0)),
        "duration": float(seg.get("end", 0) - seg.get("start", 0)),
        "text": seg.get("text", ""),
        "ft_status": seg.get("ft_status"),   # <-- carry through
        "ft_meta": seg.get("ft_meta", {}),   # optional but useful for logging
    }
    return c

def run_safety_or_shrink(candidate, eos_times, word_end_times, platform, fallback_mode):
    """Run safety/shrink logic on a single candidate and return status metadata"""
    from services.secret_sauce_pkg.features import finish_thought_normalize
    
    # Check if clip already ends on a finished thought
    if candidate.get("finished_thought", 0) == 1:
        return {"status": candidate.get("ft_status", "finished")}
    
    # Try to normalize this clip
    normalized_clip, result = finish_thought_normalize(candidate, eos_times, word_end_times, platform, fallback_mode)
    
    if result.get("status") == "unresolved":
        # Special handling for protected long clips
        if candidate.get("protected_long", False):
            dur = candidate.get('end', 0) - candidate.get('start', 0)
            if dur >= 18.0:  # Only for long clips
                return {"status": "unresolved", "reason": "protected_long"}
        
        # For non-protected clips or sparse EOS, be more lenient
        if fallback_mode:
            return {"status": "sparse_finished", "reason": "fallback_mode"}
        else:
            return {"status": "unresolved", "reason": "no_eos_hit"}
    else:
        # Clip was modified - update it
        candidate.update(normalized_clip)
        return result

from services.secret_sauce_pkg import compute_features_v4, score_segment_v4, explain_segment_v4, viral_potential_v4, get_clip_weights
from services.viral_moment_detector import ViralMomentDetector
from services.progress_writer import write_progress
from services.prerank import pre_rank_candidates, get_safety_candidates, pick_stratified
from services.quality_filters import fails_quality, filter_overlapping_candidates, filter_low_quality
from services.candidate_formatter import format_candidates

# Enhanced compute function with all Phase 1-3 features
from services.secret_sauce_pkg.features import compute_features_v4_enhanced as _compute_features

# Feature coverage logging
REQUIRED_FEATURE_KEYS = [
    "hook_score", "arousal_score", "payoff_score", "info_density",
    "loopability", "insight_score", "platform_len_match"
]
OPTIONAL_FEATURE_KEYS = [
    "insight_conf", "q_list_score", "prosody_arousal", "platform_length_score_v2", "emotion_score"
]

def _log_feature_coverage(logger, feats: dict):
    missing = [k for k in REQUIRED_FEATURE_KEYS if k not in feats]
    if missing:
        logger.warning(f"[features] Missing required: {missing}")
    soft_missing = [k for k in OPTIONAL_FEATURE_KEYS if k not in feats]
    if soft_missing:
        logger.info(f"[features] Optional not present: {soft_missing}")
from config_loader import get_config

# ensure v2 is ON everywhere
def is_on(_): return True  # temporary until flags are unified

logger = logging.getLogger(__name__)

def _soft_relax_on_zero(pool, min_keep=3):
    """
    If gating/pick produced zero, rescue top-K by score.
    Prefer display_score → fs → score as fallbacks, and use dur, finished, refined, 
    and terminal punctuation in text to define "finished-like."
    """
    if not pool:
        return []
    
    # Score field priority: display_score → fs → score
    key = None
    for score_field in ["display_score", "fs", "score"]:
        if score_field in pool[0]:
            key = score_field
            break
    
    if key is None:
        # Last resort: keep incoming order
        logger.warning("SOFT_RELAX: no score field found, keeping original order")
        return pool[:min_keep]
    
    # Sort by score (descending) and take top-K
    sorted_pool = sorted(pool, key=lambda c: float(c.get(key, 0.0)), reverse=True)
    rescued = sorted_pool[:min_keep]
    
    logger.info(f"SOFT_RELAX: rescued {len(rescued)} items using {key} field")
    return rescued

# Environment variable for soft-relax
import os
_SOFT_RELAX = os.getenv("FT_SOFT_RELAX_ON_ZERO", "1") != "0"

# Terminal punctuation patterns for finished-like detection
_TERMINAL_PUNCT = ('.', '!', '?', '…', '."', '!"', '?"')

def _score_of(x):
    """Best available display score first, then fs, then raw score"""
    return (x.get('display_score')
            or x.get('fs')
            or x.get('score')
            or 0.0)

def _is_finished_like(x):
    """Check if a candidate is finished-like based on various indicators"""
    if x.get('finished') or x.get('refined') or x.get('finished_thought'):
        return True
    txt = (x.get('text') or '').rstrip()
    if not txt:
        return False
    if txt.endswith(_TERMINAL_PUNCT):
        return True
    # Some pipelines annotate "terminal punctuation kept" as a cap/flag
    caps = x.get('caps') or []
    if isinstance(caps, (list, tuple)) and any('TERMINAL_PUNCT' in str(c).upper() for c in caps):
        return True
    return False

def _get_platform_mode():
    """
    Decide platform-mode flags from config/env in one place.
    Returns: (pl_v2_weight: float, platform_protect: bool, length_agnostic: bool)
    """
    import os
    # v2 weighting (0..1)
    pl_v2_weight = float(os.getenv("PL_V2_WEIGHT", "0.5") or 0.5)
    # keep at least one long clip in traditional/platform-biased modes
    platform_protect = os.getenv("PLATFORM_PROTECT", "1") in ("1", "true", "True")
    # when TRUE we treat all lengths neutrally and cap long-count
    length_agnostic = os.getenv("LENGTH_AGNOSTIC", "0") in ("1", "true", "True")
    
    logger.info(
        "PLATFORM_MODE: pl_v2_weight=%s, platform_protect=%s, length_agnostic=%s",
        pl_v2_weight, platform_protect, length_agnostic
    )
    return pl_v2_weight, platform_protect, length_agnostic

def test_limits_never_nameerror_or_empty():
    """Quick sanity test to prevent regressions."""
    _, _, length_agnostic = _get_platform_mode()
    finals = [
        {"id":"a","duration":48.0,"score":0.60},
        {"id":"b","duration":16.0,"score":0.58},
        {"id":"c","duration":27.5,"score":0.57},
    ]
    out = apply_length_bucket_cap(finals, length_agnostic=length_agnostic)
    assert out, "Limiter must never return empty when input non-empty"
    return True

def _normalize_episode_words(maybe_words):
    """Accept: list[dict], dict with 'words', or anything else → []"""
    if isinstance(maybe_words, list):
        return [w for w in maybe_words if isinstance(w, dict)]
    if isinstance(maybe_words, dict) and isinstance(maybe_words.get("words"), list):
        return [w for w in maybe_words["words"] if isinstance(w, dict)]
    return []

def _safe_eos(list_or_none):
    return list_or_none if isinstance(list_or_none, list) else []

def _word_end_times(words):
    """words: list[dict] with ('t'/'start','d'/'end')"""
    out = []
    if isinstance(words, list):
        for w in words:
            if not isinstance(w, dict):
                continue
            t0 = float(w.get("t", w.get("start", 0.0)) or 0.0)
            d  = float(w.get("d", (w.get("end", t0) - t0)) or 0.0)
            out.append(t0 + max(0.0, d))
    return out

def _telemetry_density(episode_words, eos_times):
    we = _word_end_times(episode_words)
    et = _safe_eos(eos_times)
    wc = len(we)
    ec = len(et)
    # Avoid bogus density; prefer 0.0 over inflated value
    dens = (ec / wc) if wc > 0 else 0.0
    return wc, ec, dens

def apply_length_bucket_cap(finals, *, length_agnostic: bool) -> list:
    """
    Apply length bucket cap safely - never returns empty list if input had clips.
    Only enforces 'max 1 long' policy when truly length-agnostic.
    """
    if not finals:
        return finals

    try:
        short, mid, long = [], [], []
        for c in finals:
            d = float(c.get("duration", 0))
            (short if d < 13 else mid if d <= 30 else long).append(c)

        # Only cap when truly length agnostic
        if length_agnostic and len(long) > 1:
            long_sorted = sorted(long, key=lambda x: x.get("score", 0), reverse=True)
            keep_one = {id(long_sorted[0])}
            # drop the rest of longs
            result = [c for c in finals if (c in short or c in mid or id(c) in keep_one)]
            logger.info("LENGTH_CAP: kept 1 long, dropped %d longs, total kept=%d", 
                       len(long_sorted) - 1, len(result))
            return result

        return finals

    except Exception as e:
        logger.exception("LENGTH_CAP_ERROR: %s", e)
        # On any error, DO NOT change finals
        return finals


def _log_final_summary(finals, logger):
    total = len(finals)
    strict_finished = sum(
        1 for c in finals
        if c.get("finished_thought") is True
        or c.get("finished") is True
        or c.get("ft_status") == "finished"
    )
    sparse_finished = sum(1 for c in finals if c.get("ft_status") == "sparse_finished")
    longest = max((float(c.get("end",0))-float(c.get("start",0)) for c in finals), default=0.0)
    ratio_strict = (strict_finished / total) if total else 0.0
    ratio_sparse_ok = ((strict_finished + sparse_finished) / total) if total else 0.0
    logger.info(
        "FT_SUMMARY_FINAL: finished=%d sparse=%d total=%d ratio_strict=%.2f ratio_sparse_ok=%.2f longest=%.1fs",
        strict_finished, sparse_finished, total, ratio_strict, ratio_sparse_ok, longest
    )

def choose_default_clip(clips: List[Dict[str, Any]]) -> Optional[str]:
    """Choose the default clip for UI display"""
    if not clips:
        return None
    
    longs = [c for c in clips if c.get("protected_long")]
    if longs:
        longs.sort(key=lambda c: (-c.get("pl_v2", 0), -c.get("duration", 0), -c.get("final_score", 0)))
        return longs[0]["id"]
    
    rest = sorted(clips, key=lambda c: (-c.get("final_score", 0), -c.get("finished_thought", 0), -c.get("pl_v2", 0)))
    if rest:
        return rest[0]["id"]
    
    return None

def rank_clips(clips: List[Dict[str, Any]], platform_neutral: bool = True) -> List[Dict[str, Any]]:
    """Rank clips using platform-neutral V_core scoring or traditional scoring with seed-first ranking"""
    from services.secret_sauce_pkg.features import compute_v_core
    
    # Separate seed clips from brute-force clips
    seed_clips = [c for c in clips if c.get('seed_idx') is not None]
    brute_clips = [c for c in clips if c.get('seed_idx') is None]
    
    logger.info("RANKING: %d seed clips, %d brute clips", len(seed_clips), len(brute_clips))
    
    if platform_neutral:
        # Platform-neutral ranking using V_core
        for clip in clips:
            start_s = clip.get("start", 0.0)
            end_s = clip.get("end", 0.0)
            v_core = compute_v_core(clip, start_s, end_s)
            clip["v_core_score"] = v_core
        
        # Sort seed clips by V_core score (highest first)
        ranked_seed = sorted(seed_clips, key=lambda c: -c.get("v_core_score", 0.0))
        
        # Sort brute clips by V_core score (highest first)
        ranked_brute = sorted(brute_clips, key=lambda c: -c.get("v_core_score", 0.0))
        
        # Combine: seed clips first, then brute clips
        sorted_clips = ranked_seed + ranked_brute
    else:
        # Traditional ranking with platform bias
        # Seed clips get priority
        seed_longs = sorted([c for c in seed_clips if c.get("protected_long")],
                           key=lambda c: (-c.get("pl_v2", 0), -c.get("duration", 0), -c.get("final_score", 0)))
        seed_rest = sorted([c for c in seed_clips if not c.get("protected_long")],
                          key=lambda c: (-c.get("final_score", 0), -c.get("finished_thought", 0), -c.get("pl_v2", 0)))
        
        # Brute clips with stricter quality requirements
        brute_longs = sorted([c for c in brute_clips if c.get("protected_long")],
                            key=lambda c: (-c.get("pl_v2", 0), -c.get("duration", 0), -c.get("final_score", 0)))
        brute_rest = sorted([c for c in brute_clips if not c.get("protected_long")],
                           key=lambda c: (-c.get("final_score", 0), -c.get("finished_thought", 0), -c.get("pl_v2", 0)))
        
        # Apply stricter quality filter to brute clips
        brute_rest = [c for c in brute_rest if c.get("final_score", 0) > 0.3]  # Higher threshold for brute clips
        
        sorted_clips = seed_longs + seed_rest + brute_longs + brute_rest
    
    # Apply dedupe to remove near-duplicates (Phase 3)
    deduped_clips = dedupe_keep_best(sorted_clips)
    
    # Add rank to each clip
    for i, c in enumerate(deduped_clips): 
        c["rank_primary"] = i
    
    logger.info("RANKING: final order has %d clips (%d seed + %d brute) after dedupe", 
                len(deduped_clips), len(seed_clips), len(brute_clips))
    
    return deduped_clips

def _enhanced_select_and_rank(candidates: List[Dict], words: List[Dict], platform_cfg: Dict) -> List[Dict]:
    """Enhanced selection with null-safe processing and adaptive gating"""
    from services.quality_filters import filter_low_quality
    
    # Sanitize first
    candidates = [_ensure_clip_fields(c) for c in candidates]
    for c in candidates:
        _blend_final_score(c)
    
    logger.info("ENHANCED: candidates=%d", len(candidates))
    
    # Apply quality filtering (strict mode first)
    filtered = filter_low_quality(candidates, mode="strict")
    if not filtered:
        logger.info("ENHANCED: strict gates empty → soft gates")
        filtered = filter_low_quality(candidates, mode="soft")
    if not filtered:
        logger.warning("ENHANCED: soft gates empty → top-K by score")
        # Be more lenient - take more candidates
        filtered = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)[:max(6, min(15, len(candidates)))]
    
    logger.info(f"ENHANCED: after filtering: {len(filtered)} candidates")
    
    # Choose finals safely
    desired_k = max(6, min(12, len(filtered)))
    finals = sorted(filtered, key=lambda x: x.get("final_score", 0.0), reverse=True)[:desired_k]
    
    logger.info("ENHANCED: finals=%d (after gates/MMR)", len(finals))
    return finals

def final_safety_pass(clips: List[Dict[str, Any]], eos_times: List[float], word_end_times: List[float], platform: str) -> List[Dict[str, Any]]:
    """
    Final safety pass to ensure all clips end on finished thoughts.
    This catches any clips that might have slipped through without proper EOS normalization.
    Always runs, even with sparse EOS data using fallback mode.
    """
    if not eos_times or not word_end_times:
        logger.warning("SAFETY_FALLBACK: No EOS data available, using relaxed mode")
        # In fallback mode, just mark all clips as finished to avoid drops
        for clip in clips:
            clip['finished_thought'] = 1
        return clips
    
    from services.secret_sauce_pkg.features import finish_thought_normalize, near_eos
    
    safety_updated = 0
    safety_dropped = 0
    safety_protected = 0
    
    # Check EOS density
    word_count = len(word_end_times)
    eos_density = len(eos_times) / max(word_count, 1)
    is_sparse = eos_density < 0.02 or word_count < 500
    
    if is_sparse:
        logger.warning(f"EOS_SPARSE: Using relaxed safety mode (density: {eos_density:.3f}, words: {word_count})")
    
    for clip in clips:
        # Check if clip already ends on a finished thought
        if clip.get("finished_thought", 0) == 1:
            continue
        
        # Try to normalize this clip
        # Determine fallback mode based on EOS density
        fallback_mode = len(eos_times) < 50 or len(word_end_times) < 500
        normalized_clip, result = finish_thought_normalize(clip, eos_times, word_end_times, platform, fallback_mode)
        
        if result == "unresolved":
            # Special handling for protected long clips
            if clip.get("protected_long", False):
                # Try last-resort extension with wider window
                dur = clip.get('end', 0) - clip.get('start', 0)
                if dur >= 18.0:  # Only for long clips
                    logger.warning(f"SAFETY_PROTECTED: keeping unresolved long clip {clip.get('id', 'unknown')} dur={dur:.1f}s")
                    clip['finished_thought'] = 0
                    clip['needs_review'] = True
                    safety_protected += 1
                    continue
            
            # For non-protected clips or sparse EOS, be more lenient
            if is_sparse:
                logger.warning(f"SAFETY_KEEP_SPARSE: keeping unresolved clip {clip.get('id', 'unknown')} due to sparse EOS")
                clip['finished_thought'] = 0
                clip['needs_review'] = True  # Mark for sparse EOS mode
                safety_protected += 1
                continue
            else:
                safety_dropped += 1
                logger.warning(f"SAFETY_DROP: clip {clip.get('id', 'unknown')} unresolved after safety pass")
                continue
        elif result != "snap_ok":
            # Clip was modified - update it
            clip.update(normalized_clip)
            safety_updated += 1
            
            # Safety updates must write ft_status back
            if isinstance(result, dict) and result.get("status"):
                clip["ft_status"] = result["status"]  # 'finished' | 'sparse_finished' | 'unresolved' | 'extended'
            
            logger.info(f"SAFETY_UPDATE: clip {clip.get('id', 'unknown')} -> {result}")
    
    # Calculate telemetry
    # Calculate episode-level telemetry (defensive)
    wc = len(word_end_times) if word_end_times else 0
    ec = len(eos_times) if eos_times else 0
    # Note: episode-level fallbacks are handled in the calling function
    # Avoid bogus density; prefer 0.0 over inflated value
    eos_density = (ec / wc) if wc > 0 else 0.0
    fallback_mode = wc < 500 or ec == 0
    
    # Count finish thought results
    finished_count = sum(1 for c in clips if c.get("finished_thought", 0) == 1)
    finished_ratio = finished_count / max(len(clips), 1)
    
    # Log episode-level telemetry
    logger.info(f"TELEMETRY: word_count={wc}, eos_count={ec}, eos_density={eos_density:.3f}")
    logger.info(f"TELEMETRY: fallback_mode={fallback_mode}, finished_ratio={finished_ratio:.2f}")
    logger.info(f"SAFETY_PASS: updated {safety_updated}, dropped {safety_dropped}, protected {safety_protected} clips")
    
    # Episode-level thresholds
    if wc < 500 or ec == 0:
        logger.warning("EOS_SPARSE: word_count < 500 or eos_count == 0")
    if finished_ratio < 0.95:
        logger.warning(f"FINISH_RATIO_LOW: {finished_ratio:.2f}")
    
    # Remove any clips that were marked for dropping (but keep protected ones)
    # In sparse EOS mode, be more lenient with finished_thought requirements
    if fallback_mode:
        return [c for c in clips if c.get("finished_thought", 0) == 1 or c.get("needs_review", False) or c.get("protected", False)]
    else:
        return [c for c in clips if c.get("finished_thought", 0) == 1 or c.get("needs_review", False) or near_eos(c.get("end", 0), eos_times, 0.5)]

class ClipScoreService:
    """Service for analyzing podcast episodes and scoring potential clip moments"""
    
    def __init__(self, episode_service):
        self.episode_service = episode_service

    def _best_audio_for_features(self, audio_path: str) -> str:
        """Prefer clean WAV if available to eliminate mpg123 errors"""
        import os
        base, ext = os.path.splitext(audio_path)
        wav_candidate = f"{base}.__asr16k.wav"
        if os.path.exists(wav_candidate):
            logger.debug(f"Using clean WAV for features: {wav_candidate}")
            return wav_candidate
        else:
            logger.debug(f"Using original audio for features: {audio_path}")
            return audio_path


    def pre_rank_candidates(self, segments: List[Dict], episode_id: str) -> List[Dict]:
        return pre_rank_candidates(segments, episode_id)

    def get_safety_candidates(self, segments: List[Dict]) -> List[Dict]:
        return get_safety_candidates(segments)

    def pick_stratified(self, candidates: List[Dict], target_count: int) -> List[Dict]:
        return pick_stratified(candidates, target_count)

    def analyze_episode(self, audio_path: str, transcript: List[TranscriptSegment], episode_id: str = None) -> List[MomentScore]:
        """Analyze episode and return scored moments"""
        try:
            logger.info(f"Starting episode analysis for {audio_path}")
            
            # Convert transcript to segments for ranking using improved moment detection
            segments = self._transcript_to_segments(transcript, genre='general', platform='tiktok')
            
            # NEW: expand to multi-scale candidates
            segments = self._generate_multiscale_candidates(segments)
            
            # Rank candidates using secret sauce V4 with genre awareness
            # SPEED: Reduce top_k for faster processing (still processes all segments, just ranks fewer)
            top_k = int(os.getenv("TOP_K_CANDIDATES", "15"))  # Default 15, was 10
            ranked_segments = self.rank_candidates(segments, audio_path, top_k=top_k, platform='tiktok', genre='general', episode_id=episode_id)
            
            # Convert back to MomentScore objects
            moment_scores = []
            for seg in ranked_segments:
                moment_score = MomentScore(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    duration=seg["end"] - seg["start"],
                    hook_score=seg["features"]["hook_score"],
                    emotion_score=seg["features"]["emotion_score"],
                    arousal_score=seg["features"]["arousal_score"],
                    payoff_score=seg["features"]["payoff_score"],
                    loopability_score=seg["features"]["loopability"],
                    question_or_list_score=seg["features"]["question_score"],
                    info_density_score=seg["features"]["info_density"],
                    total_score=seg["score"]
                )
                moment_scores.append(moment_score)
            
            logger.info(f"Found {len(moment_scores)} potential moments")
            return moment_scores
            
        except Exception as e:
            logger.error(f"Episode analysis failed: {e}")
            raise

    def _is_repetitive_content(self, text: str) -> bool:
        """Check if content is repetitive or filler (like 'Yeah. Yeah. Yeah...')"""
        if not text or len(text.strip()) < 5:
            return True
        
        words = text.lower().split()
        if len(words) < 5:
            return True
        
        # Check for repetitive patterns
        # Count unique words vs total words
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        # If more than 70% of words are repeated, it's likely filler
        if repetition_ratio < 0.3:
            return True
        
        # Check for specific repetitive patterns
        repetitive_patterns = [
            r'^(yeah\.?\s*){3,}',  # "Yeah. Yeah. Yeah..."
            r'^(uh\.?\s*){3,}',    # "Uh. Uh. Uh..."
            r'^(um\.?\s*){3,}',    # "Um. Um. Um..."
            r'^(ok\.?\s*){3,}',    # "Ok. Ok. Ok..."
            r'^(right\.?\s*){3,}', # "Right. Right. Right..."
            r'^(so\.?\s*){3,}',    # "So. So. So..."
            r'^(and\.?\s*){3,}',   # "And. And. And..."
        ]
        
        import re
        for pattern in repetitive_patterns:
            if re.match(pattern, text.lower()):
                return True
        
        # Check if the same word appears more than 50% of the time
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_count = max(word_counts.values()) if word_counts else 0
        if max_count > len(words) * 0.5:
            return True
        
        return False

    def _fails_quality(self, feats: dict) -> str | None:
        """Check if segment fails quality gates (soft reject) - Soft hook penalty, strong ad clamp"""
        # Conditional hook threshold based on payoff and arousal
        payoff = feats.get("payoff_score", 0.0)
        arousal = feats.get("arousal_score", 0.0)
        question = feats.get("question_score", 0.0)
        
        # Lower hook threshold when payoff and arousal are strong
        if payoff >= 0.6 and arousal >= 0.45:
            hook_threshold = 0.06  # τ_cond for high-quality clips
        elif payoff >= 0.4 and arousal >= 0.35:
            hook_threshold = 0.10  # τ_cond for good clips
        else:
            hook_threshold = 0.15  # τ for standard clips
        
        hook = feats.get("hook_score", 0.0)
        weak_hook = hook < hook_threshold
        has_early_question = question >= 0.50
        no_payoff = payoff < 0.25
        ad_like = feats.get("_ad_flag", False) or (feats.get("_ad_penalty", 0.0) >= 0.3)
        
        # Hard fail ONLY when (weak hook) AND (ad_like OR no_payoff)
        if hook < 0.08 and (ad_like or no_payoff):
            if ad_like and no_payoff:
                return "ad_like;weak_hook;no_payoff"
            elif ad_like:
                return "ad_like;weak_hook"
            else:
                return "weak_hook;no_payoff"
        
        # Soft penalties for weak hooks (never force 0.05 unless above condition)
        if hook < 0.08:
            return "weak_hook_very_soft"  # ×0.65 penalty
        elif hook < hook_threshold:
            return "weak_hook_mild_soft"  # ×0.90 penalty
        
        if no_payoff and not has_early_question:
            return "no_payoff"
        
        if arousal < 0.20:  # Increased from 0.15 to allow more variation
            return "low_energy"
        
        return None

    def rank_candidates(self, segments: List[Dict], audio_file: str, top_k=5, platform: str = 'tiktok', genre: str = 'general', episode_id: str = None) -> List[Dict]:
        """Rank candidates using two-stage scoring: pre-rank + full V4 scoring"""
        scored = []
        
        # Two-stage scoring: pre-rank first, then full scoring on top candidates
        if PRERANK_ENABLED and len(segments) > 10:  # Only use pre-rank for larger sets
            logger.info(f"Using two-stage scoring: {len(segments)} segments")
            
            # Stage 1: Pre-rank with cheap features
            prerank_candidates = self.pre_rank_candidates(segments, episode_id or "unknown")
            
            # Get safety candidates (obvious bangers)
            safety_candidates = self.get_safety_candidates(segments)
            
            # Combine and deduplicate
            safety_added = [s for s in safety_candidates if s not in prerank_candidates]
            all_candidates = prerank_candidates + safety_added
            
            logger.info(f"Pre-rank: {len(prerank_candidates)} candidates")
            logger.info(f"Safety net: {len(safety_candidates)} bangers found, {len(safety_added)} new ones added")
            logger.info(f"Combined: {len(all_candidates)} total candidates")
            
            # Apply stratified selection if enabled
            if STRATIFY_ENABLED:
                target_count = min(len(all_candidates), max(TOP_K_MIN, math.ceil(TOP_K_RATIO * len(segments))))
                final_candidates = self.pick_stratified(all_candidates, target_count)
            else:
                final_candidates = all_candidates
                
            logger.info(f"Two-stage: {len(segments)} -> {len(final_candidates)} candidates for full scoring")
            segments_to_score = final_candidates
        else:
            logger.info(f"Using full scoring: {len(segments)} segments")
            segments_to_score = segments
        
        # preload audio once for arousal (big speed-up)
        try:
            import librosa
            # Prefer clean WAV if available to eliminate mpg123 errors
            audio_for_features = self._best_audio_for_features(audio_file)
            y_sr = librosa.load(audio_for_features, sr=None)
        except Exception:
            y_sr = None

        # Compute episode-level prosody stats for arousal v2
        episode_stats = None
        if y_sr is not None:
            try:
                from services.audio_features import episode_prosody_stats
                y, sr = y_sr
                episode_stats = episode_prosody_stats(y, sr)
                logger.info("prosody.stats_computed", extra={"episode_id": episode_id})
            except Exception as e:
                logger.warning(f"Prosody stats failed: {e}")
                episode_stats = None

        # Stage 2: Full V4 scoring on selected candidates
        write_progress(episode_id or "unknown", "scoring:full", 20, f"Full scoring {len(segments_to_score)} candidates...")
        
        for i, seg in enumerate(segments_to_score):
            # Progress update every 10 items
            if i % 10 == 0:
                progress_pct = 20 + int((i / len(segments_to_score)) * 70)  # 20-90%
                write_progress(episode_id or "unknown", "scoring:full", progress_pct, f"Scoring candidate {i+1}/{len(segments_to_score)}...")
            
            # Use enhanced V4 feature computation with all Phase 1-3 features
            feats = None
            try:
                from config import settings
                feats = _compute_features(
                    segment=seg,
                    audio_file=audio_file,
                    y_sr=y_sr,
                    cfg=settings,
                    platform=platform,
                    genre=genre,
                    segments=segments_to_score,  # give enhanced fn episode context
                )
                _log_feature_coverage(logger, feats)
                
                # Apply low-activity first-second hook nudge
                hook = feats.get("hook_score", 0.0)
                if hook > 0.15 and self._low_activity_first_second(segments_to_score, seg["start"], seg["end"]):
                    hook = max(0.0, hook - 0.08)
                    feats["hook_score"] = hook
                
                seg["features"] = feats
                
                # Debug: log what features we got
                logger.info(f"Enhanced V4 Features computed: {list(feats.keys())}")
                logger.info(f"Feature values: hook={feats.get('hook_score', 0):.3f}, arousal={feats.get('arousal_score', 0):.3f}, payoff={feats.get('payoff_score', 0):.3f}")
                logger.info(f"Hook reasons: {feats.get('hook_reasons', 'none')}")
                logger.info(f"Payoff type: {feats.get('payoff_type', 'none')}")
                logger.info(f"Moment type: {feats.get('type', 'general')}")
                
                # Add prosody and trend scoring
                try:
                    if episode_stats is not None and y_sr is not None:
                        from services.audio_features import segment_prosody, blend_arousal_v2
                        y, sr = y_sr
                        t0 = float(seg.get("start", 0.0)); t1 = float(seg.get("end", t0))
                        prosody = segment_prosody(y, sr, t0, t1)
                        feats["prosody_rms"] = prosody["prosody_rms"]
                        feats["prosody_flux"] = prosody["prosody_flux"]
                        feats["arousal_score_v2"] = blend_arousal_v2(
                            feats.get("arousal_score", 0.0), feats, episode_stats, ab_key=episode_id
                        )
                        feats["_arousal_src"] = "v2"
                        logger.debug("prosody.v2_ok", extra={"segment_id": seg.get("id")})
                except Exception as e:
                    logger.debug("prosody.v2_skipped", extra={"segment_id": seg.get("id"), "err": str(e)})
                
                try:
                    from services.trending_provider import trend_match_score, detect_categories, collect_hashtags
                    raw_text = seg.get("text", "")
                    
                    # Merge hashtags from text + episode meta (if any)
                    try:
                        # Get episode metadata from the episode object if available
                        episode_meta = None
                        if hasattr(self, 'current_episode') and self.current_episode and hasattr(self.current_episode, 'meta'):
                            episode_meta = self.current_episode.meta
                        
                        hashtags = collect_hashtags(raw_text, episode_meta)
                        if hashtags:
                            feats["hashtags"] = hashtags
                    except Exception:
                        hashtags = feats.get("hashtags", [])
                    
                    cats = detect_categories(raw_text)
                    
                    # Optional: use episode ID for AB bucket stability
                    ab_key = episode_id if 'episode_id' in locals() else None
                    
                    # Enhanced trend computation
                    new_trend = trend_match_score(raw_text, hashtags, categories=cats, ab_key=ab_key)
                    old_trend = feats.get("trend_match_score", 0.0)
                    feats["trend_match_score"] = new_trend if new_trend > 0 else old_trend
                    feats["trend_categories"] = cats
                    
                    # AB test logging
                    enabled = new_trend > 0
                    logger.info("trend.ab", extra={"episode_id": episode_id, "enabled": enabled})
                    
                    if feats["trend_match_score"] > 0:
                        logger.info("trend.hit", extra={"cats": cats, "score": round(feats["trend_match_score"], 3)})
                except Exception as e:
                    logger.debug(f"Trend score skipped: {e}")
                
                # Diagnostic: log what features we have
                logger.debug("features.computed", extra={
                    "segment_id": seg.get("id"),
                    "has_arousal_v2": "arousal_score_v2" in feats,
                    "has_trend": "trend_match_score" in feats,
                    "arousal_v2_val": feats.get("arousal_score_v2", None),
                    "trend_val": feats.get("trend_match_score", None)
                })
            except Exception as e:
                logger.warning(f"Enhanced features failed for segment {seg.get('id', i)}: {e}; falling back to v4")
                try:
                    from config import settings
                    feats = compute_features_v4(seg, audio_file, y_sr=y_sr, platform=platform, genre=genre, cfg=settings)
                    _log_feature_coverage(logger, feats)
                    
                    # Apply low-activity first-second hook nudge
                    hook = feats.get("hook_score", 0.0)
                    if hook > 0.15 and self._low_activity_first_second(segments_to_score, seg["start"], seg["end"]):
                        hook = max(0.0, hook - 0.08)
                        feats["hook_score"] = hook
                    
                    seg["features"] = feats
                    
                    # Debug: log what features we got
                    logger.info(f"Fallback V4 Features computed: {list(feats.keys())}")
                    logger.info(f"Feature values: hook={feats.get('hook_score', 0):.3f}, arousal={feats.get('arousal_score', 0):.3f}, payoff={feats.get('payoff_score', 0):.3f}")
                    
                    # Add prosody and trend scoring (fallback path)
                    try:
                        if episode_stats is not None and y_sr is not None:
                            from services.audio_features import segment_prosody, blend_arousal_v2
                            y, sr = y_sr
                            t0 = float(seg.get("start", 0.0)); t1 = float(seg.get("end", t0))
                            prosody = segment_prosody(y, sr, t0, t1)
                            feats["prosody_rms"] = prosody["prosody_rms"]
                            feats["prosody_flux"] = prosody["prosody_flux"]
                            feats["arousal_score_v2"] = blend_arousal_v2(
                                feats.get("arousal_score", 0.0), feats, episode_stats, ab_key=episode_id
                            )
                            feats["_arousal_src"] = "v2"
                            logger.debug("prosody.v2_ok", extra={"segment_id": seg.get("id")})
                    except Exception as e:
                        logger.debug("prosody.v2_skipped", extra={"segment_id": seg.get("id"), "err": str(e)})
                    
                    try:
                        from services.trending_provider import trend_match_score, detect_categories, collect_hashtags
                        raw_text = seg.get("text", "")
                        
                        # Merge hashtags from text + episode meta (if any)
                        try:
                            # Get episode metadata from the episode object if available
                            episode_meta = None
                            if hasattr(self, 'current_episode') and self.current_episode and hasattr(self.current_episode, 'meta'):
                                episode_meta = self.current_episode.meta
                            
                            hashtags = collect_hashtags(raw_text, episode_meta)
                            if hashtags:
                                feats["hashtags"] = hashtags
                        except Exception:
                            hashtags = feats.get("hashtags", [])
                        
                        cats = detect_categories(raw_text)
                        
                        # Optional: use episode ID for AB bucket stability
                        ab_key = episode_id if 'episode_id' in locals() else None
                        
                        # Enhanced trend computation
                        new_trend = trend_match_score(raw_text, hashtags, categories=cats, ab_key=ab_key)
                        old_trend = feats.get("trend_match_score", 0.0)
                        feats["trend_match_score"] = new_trend if new_trend > 0 else old_trend
                        feats["trend_categories"] = cats
                        
                        # AB test logging
                        enabled = new_trend > 0
                        logger.info("trend.ab", extra={"episode_id": episode_id, "enabled": enabled})
                        
                        if feats["trend_match_score"] > 0:
                            logger.info("trend.hit", extra={"cats": cats, "score": round(feats["trend_match_score"], 3)})
                    except Exception as e:
                        logger.debug(f"Trend score skipped: {e}")
                    
                    # Diagnostic: log what features we have (fallback path)
                    logger.debug("features.computed", extra={
                        "segment_id": seg.get("id"),
                        "has_arousal_v2": "arousal_score_v2" in feats,
                        "has_trend": "trend_match_score" in feats,
                        "arousal_v2_val": feats.get("arousal_score_v2", None),
                        "trend_val": feats.get("trend_match_score", None)
                    })
                except Exception as e2:
                    logger.exception(f"Both enhanced and fallback feature compute failed for segment {seg.get('id', i)}: {e2}")
                    continue
            
            # Skip if no features were computed
            if feats is None:
                continue
            
            # Get raw score using V4 multi-path scoring with genre awareness
            current_weights = get_clip_weights()
            logger.info(f"Available weights: {list(current_weights.keys())}")
            
            # Use V4 multi-path scoring with genre awareness
            scoring_result = score_segment_v4(feats, current_weights, genre=genre, platform=platform)
            raw_score = scoring_result["final_score"]
            winning_path = scoring_result["winning_path"]
            path_scores = scoring_result["path_scores"]
            synergy_multiplier = scoring_result["synergy_multiplier"]
            bonuses_applied = scoring_result["bonuses_applied"]
            bonus_reasons = scoring_result["bonus_reasons"]
            
            seg["winning_path"] = winning_path
            seg["path_scores"] = path_scores
            seg["synergy_multiplier"] = synergy_multiplier
            seg["bonuses_applied"] = bonuses_applied
            seg["bonus_reasons"] = bonus_reasons
            
            logger.info(f"V4 Scoring: {raw_score:.3f}, Path: {winning_path}, Synergy: {synergy_multiplier:.3f}, Bonuses: {bonuses_applied:.3f}")
            logger.info(f"Bonus reasons: {bonus_reasons}")
            
            # Apply ad penalty AFTER scoring, BEFORE calibration
            ad_flag = feats.get("_ad_flag", False)
            ad_pen = feats.get("_ad_penalty", 0.0)
            ad_reason = feats.get("_ad_reason", "none")
            
            # hard-drop ads
            if ad_flag or ad_pen >= 0.3:
                raw_score = -1.0  # Hard drop ads
            else:
                raw_score *= (1.0 - ad_pen * 0.40)  # Increased penalty for borderline ads
            seg["ad_penalty"] = float(ad_pen)
            seg["_ad_flag"] = ad_flag
            seg["_ad_reason"] = ad_reason
            
            # Features are already clamped in compute_features for ads
            # This ensures consistency between feature calculation and scoring
            
            # Soft quality gate with genre-specific penalties
            reason = fails_quality(feats)
            seg["discard_reason"] = reason
            
            if reason:
                # Get genre-specific quality gate config
                from config_loader import get_config
                config = get_config()
                quality_config = config.get("quality_gate", {})
                global_config = quality_config.get("global", {})
                genre_config = quality_config.get(genre, {})
                
                # Apply genre-specific penalties
                if reason == "weak_hook_very_soft":
                    penalty_mult = global_config.get("weak_hook_very_soft", 0.65)
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"very_weak_hook_{penalty_mult}"
                    logger.info(f"Clip got soft penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif reason == "weak_hook_mild_soft":
                    penalty_mult = global_config.get("weak_hook_mild_soft", 0.90)
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"mild_weak_hook_{penalty_mult}"
                    logger.info(f"Clip got mild penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif reason == "no_payoff":
                    # Check for insight/question fallback
                    has_insight = feats.get("insight_score", 0.0) >= 0.70
                    has_question = feats.get("question_score", 0.0) >= 0.50
                    
                    if has_insight or has_question:
                        # Use genre-specific soften for insights/questions
                        penalty_mult = genre_config.get("no_payoff_soften", global_config.get("no_payoff_soften", 0.85))
                        raw_score *= penalty_mult
                        seg["soft_penalty"] = f"no_payoff_with_insight_{penalty_mult}"
                        logger.info(f"Clip softened for insight/question: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                    else:
                        # Standard no_payoff penalty
                        penalty_mult = genre_config.get("no_payoff_soften", global_config.get("no_payoff_soften", 0.70))
                        raw_score *= penalty_mult
                        seg["soft_penalty"] = f"no_payoff_{penalty_mult}"
                        logger.info(f"Clip got no_payoff penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif reason == "low_energy":
                    penalty_mult = global_config.get("low_energy_soften", 0.75)
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"low_energy_{penalty_mult}"
                    logger.info(f"Clip got low_energy penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif "ad_like" in reason:
                    # Hard floor only for ads
                    logger.info(f"Clip failed quality gate: {reason}, setting score to 0.05")
                    raw_score = 0.05
                else:
                    # Fallback for other reasons
                    penalty_mult = 0.70
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"other_{penalty_mult}"
                    logger.info(f"Clip got other penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                seg["raw_score"] = raw_score
            else:
                seg["raw_score"] = raw_score
                logger.info(f"Clip passed quality gate, final score: {raw_score:.3f}")
            
            # Gentle energy dampener for low-arousal clips (optional)
            arousal = feats.get("arousal_score", 0.0)
            if arousal < 0.38:
                energy_dampener = 0.95  # small nudge for low energy
                raw_score *= energy_dampener
                seg["energy_dampener"] = f"low_energy_{energy_dampener}"
                logger.info(f"Applied energy dampener: {energy_dampener}x for arousal={arousal:.3f}, final score: {raw_score:.3f}")
            
            # Apply platform blend for neutral mode (small platform influence)
            pl_v2 = feats.get("platform_length_score_v2", 0.0)
            if platform in ["shorts", "tiktok", "reels"] or platform is None:
                # Small platform blend: 90% core + 10% platform
                raw_score = raw_score * 0.9 + 0.1 * pl_v2
                seg["platform_blend"] = f"neutral_{0.1}"
            
            # Set final_score field for consistency
            seg["final_score"] = raw_score
            
            # Calibrate score for better user experience with wider range
            calibrated_score = self._calibrate_score_for_ui(raw_score)
            seg["display_score"] = calibrated_score["score"]
            seg["clip_score_100"] = calibrated_score["score"]  # Set clip_score_100 for frontend
            seg["confidence"] = calibrated_score["confidence"]
            # Expose full virality values for frontend
            seg["virality_calibrated"] = calibrated_score.get("virality_calibrated", raw_score)
            seg["virality_pct"] = calibrated_score.get("virality_pct", int(round(raw_score * 100)))
            seg["confidence_color"] = calibrated_score["color"]
            
            # Add platform fit information
            platform_fit = seg.get("platform_length_score_v2", 0.0)
            seg["platform_fit"] = round(platform_fit, 2)
            seg["platform_fit_pct"] = int(round(platform_fit * 100))
            
            seg["score"] = seg["raw_score"]  # For backward compatibility
            
            # Use V4 explanation system
            seg["explain"] = explain_segment_v4(feats, genre=genre)
            
            # Add V4 viral score and platform recommendations
            length_s = float(seg["end"] - seg["start"])
            seg["viral"] = viral_potential_v4(feats, length_s, "general")
            
            # -------- DEBUG JSON LINE (feed to logs for analysis) --------
            if logger.isEnabledFor(logging.INFO):
                dbg = {
                    "episode_id": episode_id or "unknown",
                    "clip_id": seg.get("id", f"clip_{i}"),
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                    "duration_sec": round(length_s, 3),
                    "final_score": round(raw_score, 4),
                    "hook": round(feats.get("hook_score", 0.0), 4),
                    "arousal": round(feats.get("arousal_score", 0.0), 4),
                    "payoff": round(feats.get("payoff_score", 0.0), 4),
                    "emotion": round(feats.get("emotion_score", 0.0), 4),
                    "info_density": round(feats.get("info_density", 0.0), 4),
                    "question_score": round(feats.get("question_score", 0.0), 4),
                    "platform_length": round(feats.get("platform_length_score_v2", feats.get("platform_len_match", 0.0)), 4),
                    "insight_score": round(feats.get("insight_score", 0.0), 4),
                    "loopability": round(feats.get("loopability", 0.0), 4),
                    "_synergy_boost": round(feats.get("_synergy_boost", 0.0), 4),
                    "_bonuses": {
                        "insight": round(feats.get("insight_score", 0.0) * 0.15 if feats.get("insight_score", 0.0) >= 0.7 else 0.0, 4),
                        "question": round(feats.get("question_score", 0.0) * 0.10 if feats.get("question_score", 0.0) >= 0.6 else 0.0, 4),
                        "platform_fit": round(0.02 if feats.get("platform_length_score_v2", feats.get("platform_len_match", 0.0)) >= 0.90 else 0.0, 4)
                    },
                    "_penalties": {
                        "ad_penalty": round(feats.get("_ad_penalty", 0.0), 4),
                        "soft_penalty": seg.get("soft_penalty", "none"),
                        "energy_dampener": seg.get("energy_dampener", "none")
                    },
                    "winning_path": winning_path,
                    "path_scores": {k: round(v, 4) for k, v in path_scores.items()},
                    "title": seg.get("title", ""),
                    "transcript_preview": (seg.get("text", "") or "").strip()[:240]
                }
                # Single-line JSON: greppable and easy to paste
                logger.info("CLIP_SCORING %s", json.dumps(dbg, ensure_ascii=False))
            
            scored.append(seg)
        
        # Completion progress update
        write_progress(episode_id or "unknown", "scoring:completed", 100, f"Scoring complete: {len(scored)} candidates processed")
        
        # Apply band normalization for fair ranking across length bands
        scored = _normalize_display_by_band(scored)
        
        # Sort by rank_score (band-normalized) descending but prefer non-discarded
        scored_sorted = sorted(scored, key=lambda s: (s.get("discard_reason") is not None, -s.get("rank_score", s.get("raw_score", 0))))
        
        # Apply MMR selection for diversity (remove duplicates)
        scored_sorted = _mmr_select_jaccard(scored_sorted, top_k=top_k, text_key="text", lam=0.7)
        return scored_sorted
    
    def _calibrate_score_for_ui(self, raw_score: float) -> dict:
        """Transform raw 0-1 scores into user-friendly 45-95 range with confidence bands"""
        
        # Wider range for better differentiation: 0.1 (weak) to 0.8 (excellent)
        min_score = 0.1   # weakest passing clip
        max_score = 0.8   # best observed clip
        
        if raw_score < min_score:
            return {"score": 40, "confidence": "⚠️ Fair", "color": "text-red-600"}
        else:
            # Map 0.1 → 45%, 0.8 → 95%, cap at 100%
            normalized = (raw_score - min_score) / (max_score - min_score)
            calibrated = min(int(normalized * 50 + 45), 100)  # Cap at 100%
            
            # Confidence bands
            if calibrated >= 90:
                confidence = "🔥 Exceptional"
                color = "text-green-600"
            elif calibrated >= 80:
                confidence = "⭐ Premium"
                color = "text-blue-600"
            elif calibrated >= 70:
                confidence = "✅ Strong"
                color = "text-yellow-600"
            elif calibrated >= 60:
                confidence = "👍 Good"
                color = "text-orange-600"
            else:
                confidence = "⚠️ Fair"
                color = "text-red-600"
            
            return {
                "score": calibrated,
                "confidence": confidence,
                "color": color,
                "virality_calibrated": raw_score,  # 0-1 range
                "virality_pct": int(round(raw_score * 100))  # 0-100 range
            }

    def _find_topic_boundaries(self, transcript: List[TranscriptSegment]) -> List[int]:
        """Find natural topic transitions in conversation"""
        boundaries = [0]
        
        topic_markers = [
            "so anyway", "moving on", "let me tell you", 
            "here's the thing", "the point is", "basically",
            "now", "so", "well", "okay", "right",
            "let's talk about", "speaking of", "by the way",
            "the thing is", "what I mean is", "the bottom line",
            "to be honest", "frankly", "honestly",
            "you know what", "here's what", "this is why",
            "the reason is", "because", "since",
            "first of all", "secondly", "finally",
            "in conclusion", "to sum up", "overall"
        ]
        
        for i, seg in enumerate(transcript):
            text = (getattr(seg, "text", None) if not isinstance(seg, dict) else seg.get("text", "")) or ""
            text = text.lower()
            # Check for topic transition markers
            if any(marker in text for marker in topic_markers):
                boundaries.append(i)
        
        # Add the end if not already included
        if boundaries[-1] != len(transcript) - 1:
            boundaries.append(len(transcript) - 1)
        
        return boundaries

    def _transcript_to_segments_aligned(self, transcript: List[TranscriptSegment]) -> List[Dict]:
        """Create topic-based segments that are more self-contained"""
        segments = []
        
        if not transcript:
            return segments
        
        # Find topic boundaries
        boundaries = self._find_topic_boundaries(transcript)
        logger.info(f"Found {len(boundaries)} topic boundaries")
        
        # Create segments from topic boundaries with non-overlapping logic
        current_end = 0.0  # Track where the last segment ended
        
        # Sort boundaries by start time to ensure proper ordering
        boundary_times = [(transcript[boundaries[i]].start, boundaries[i]) for i in range(len(boundaries))]
        boundary_times.sort()
        
        for i in range(len(boundary_times) - 1):
            start_idx = boundary_times[i][1]
            end_idx = boundary_times[i + 1][1]
            
            # Get the segment
            segment_start = transcript[start_idx].start
            segment_end = transcript[end_idx].end
            
            # Only create segment if it doesn't overlap with previous ones
            if segment_start >= current_end:
                # Calculate duration
                duration = segment_end - segment_start
                
                # More flexible duration range to get more segments
                if 8.0 <= duration <= 90.0:  # Expanded range for more variety
                    segment_text = " ".join([t.text for t in transcript[start_idx:end_idx+1]])
                    
                    # Less restrictive self-contained check for topic-based segments
                    if self._is_self_contained_topic(segment_text):
                        segment = {
                            "start": segment_start,
                            "end": segment_end,
                            "text": segment_text
                        }
                        segments.append(segment)
                        current_end = segment_end  # Update the end point
                        logger.info(f"Created topic segment: {segment_start:.1f}s - {segment_end:.1f}s ({duration:.1f}s)")
        
        # Final check: ensure no overlaps exist by filtering overlapping segments
        if len(segments) > 1:
            non_overlapping = [segments[0]]
            for seg in segments[1:]:
                if seg["start"] >= non_overlapping[-1]["end"]:
                    non_overlapping.append(seg)
            segments = non_overlapping
            logger.info(f"Filtered to {len(segments)} non-overlapping topic segments")
        
        # If we didn't find enough topic-based segments, fall back to the original method
        if len(segments) < 3:  # Lowered threshold from 5 to 3
            logger.info("Topic-based segmentation produced too few segments, falling back to window-based approach")
            return self._transcript_to_segments(transcript, platform='tiktok')
        
        logger.info(f"Created {len(segments)} topic-based segments")
        return segments

    def _is_self_contained(self, text: str) -> bool:
        """Check if a segment is self-contained and doesn't require prior context"""
        text_lower = text.lower()
        
        # Context-dependent patterns that indicate the segment needs prior context
        context_patterns = [
            r'\b(he|she|they|it|this|that|these|those)\b.*\b(too|also|as well|either)\b',
            r'\b(so|therefore|thus|hence|consequently)\b.*\b(that\'s why|this is why)\b',
            r'\b(you need to|you should|you have to)\b.*\b(this way|that way)\b',
            r'\b(they feel|they think|they know)\b.*\b(the care|the love|the attention)\b',
            r'\b(that\'s why|this is why)\b.*\b(we\'re|you\'re|they\'re)\b.*\b(successful|good|better)\b',
            r'\b(oh yeah|yeah|right)\b.*\b(and how|and what|and why)\b',
            r'\b(you know|you see|you understand)\b.*\b(what I mean|what I\'m saying)\b',
            r'\b(like I said|as I mentioned|as we discussed)\b',
            r'\b(going back to|returning to|coming back to)\b',
            r'\b(in the same way|similarly|likewise)\b'
        ]
        
        # Check for context-dependent patterns
        for pattern in context_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for incomplete thoughts
        incomplete_patterns = [
            r'\b(and|but|so|because|since|while|although)\s*$',
            r'\b(is|are|was|were|has|have|had)\s*$',
            r'\b(you|he|she|they|it|this|that)\s*$',
            r'\.{3}$',  # Ends with ellipsis
            r'\?$'  # Ends with question (often indicates incomplete thought)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for minimum meaningful content
        words = text.split()
        if len(words) < 8:  # Too short to be meaningful
            return False
        
        # Check for proper sentence structure
        if not re.search(r'[.!?]$', text):  # Doesn't end with proper punctuation
            return False
        
        return True

    def _is_self_contained_topic(self, text: str) -> bool:
        """Less restrictive self-contained check for topic-based segments"""
        text_lower = text.lower()
        
        # Only check for the most obvious context-dependent patterns
        obvious_context_patterns = [
            r'\b(like I said|as I mentioned|as we discussed)\b',
            r'\b(going back to|returning to|coming back to)\b',
            r'\b(in the same way|similarly|likewise)\b',
            r'\b(oh yeah|yeah|right)\b.*\b(and how|and what|and why)\b'
        ]
        
        # Check for obvious context-dependent patterns
        for pattern in obvious_context_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for incomplete thoughts (less strict)
        incomplete_patterns = [
            r'\b(and|but|so|because|since|while|although)\s*$',
            r'\.{3}$'  # Ends with ellipsis
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for minimum meaningful content (reduced from 8 to 6)
        words = text.split()
        if len(words) < 6:  # Less strict minimum
            return False
        
        # Check for proper sentence structure (more flexible)
        if not re.search(r'[.!?]$', text):  # Doesn't end with proper punctuation
            # Allow segments that end with incomplete thoughts if they're long enough
            if len(words) < 15:  # Only require punctuation for short segments
                return False
        
        return True

    def _is_repetitive_content(self, text: str) -> bool:
        """Check if content is repetitive/filler (like 'Yeah. Yeah. Yeah...')"""
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        # Check for repetitive patterns
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 50% of the time, it's repetitive
        max_count = max(word_counts.values())
        if max_count > len(words) * 0.5:
            return True
        
        # Check for specific repetitive patterns
        repetitive_patterns = [
            "yeah yeah yeah",
            "uh uh uh",
            "um um um",
            "so so so",
            "and and and",
            "the the the"
        ]
        
        text_lower = text.lower()
        for pattern in repetitive_patterns:
            if pattern in text_lower:
                return True
        
        return False

    def _is_intro_content(self, text: str) -> bool:
        """Check if content is intro/greeting material"""
        text_lower = text.lower().strip()
        
        intro_patterns = [
            r"^(yo|hey|hi|hello|what's up|how's it going|good morning|good afternoon|good evening)",
            r"^(it's|this is) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"^(i'm|my name is) \w+",
            r"^(welcome to|thanks for|thank you for)",
            r"^(hope you|hope everyone)",
            r"^(let's get|let's start|let's begin)",
            r"^(today we're|today i'm|today let's)"
        ]
        
        import re
        for pattern in intro_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False

    def _is_natural_ending(self, text: str) -> bool:
        """Check if text ends naturally"""
        text = text.strip()
        # Good endings
        if text.endswith(('.', '!', '?')):
            # Check it's not mid-sentence
            last_words = text.split()[-3:]
            if not any(w in ['the', 'a', 'an', 'to', 'of', 'and', 'but'] for w in last_words):
                return True
        return False

    def _transcript_to_segments(self, transcript: List[TranscriptSegment], genre: str = 'general', platform: str = 'tiktok', words: List[Dict] = None) -> List[Dict]:
        """Create dynamic segments based on natural content boundaries and platform optimization"""
        try:
            # Convert TranscriptSegment objects to dicts
            transcript_dicts = []
            for seg in transcript:
                if hasattr(seg, '__dict__'):
                    transcript_dicts.append({
                        'text': seg.text,
                        'start': seg.start,
                        'end': seg.end
                    })
                else:
                    transcript_dicts.append(seg)
            
            logger.info(f"Starting with {len(transcript_dicts)} raw transcript segments")
            
            # Filter out intro/filler content first, preserving words
            filtered_segments = []
            for seg in transcript_dicts:
                if not (self._is_intro_content(seg['text']) or self._is_repetitive_content(seg['text'])):
                    # Preserve words through filtering
                    filtered_seg = {
                        "id": seg.get("id"),
                        "text": seg.get("text", ""),
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "words": (seg.get("words") or []),     # <- preserve; never None
                        # ... any other fields you rely on
                    }
                    filtered_segments.append(filtered_seg)
            
            logger.info(f"Filtered to {len(filtered_segments)} non-intro segments")
            
            # Use dynamic segmentation based on natural content boundaries
            from services.secret_sauce_pkg.features import create_dynamic_segments
            
            # One-line guard at call site (extra insurance)
            filtered_segments = [s for s in (filtered_segments or []) if s is not None]
            
            # Null-safety preflight for segmentation
            filtered_segments = _segmentation_preflight(filtered_segments, logger)
            
            # Note: EOS times will be provided later in the pipeline
            try:
                dynamic_segments = create_dynamic_segments(filtered_segments, platform, eos_times=None, words=words)
            except Exception as e:
                logger.exception("Dynamic segmentation failed despite preflight: %s; falling back", e)
                dynamic_segments = None
            
            logger.info(f"Created {len(dynamic_segments)} dynamic segments based on natural boundaries")
            
            # Log segment details for debugging
            for i, seg in enumerate(dynamic_segments[:3]):
                duration = seg['end'] - seg['start']
                boundary_type = seg.get('boundary_type', 'unknown')
                confidence = seg.get('confidence', 0.0)
                logger.info(f"Dynamic Segment {i+1}: {duration:.1f}s ({boundary_type}, conf={confidence:.2f}) - {seg['text'][:60]}...")
            
            return dynamic_segments
            
        except Exception as e:
            logger.error(f"Dynamic segmentation failed: {e}, falling back to window-based")
            return self._window_based_segments(transcript_dicts, window=30, step=20)
    
    def _window_based_segments(self, transcript: List[Dict], window: float = 25, step: float = 15) -> List[Dict]:
        """Fallback window-based segmentation with larger steps"""
        segments = []
        
        # Find total duration
        if transcript:
            total_duration = max(seg.get('end', 0) for seg in transcript)
        else:
            return segments
        
        for start_time in range(0, int(total_duration), int(step)):
            end_time = min(start_time + window, total_duration)
            
            if end_time - start_time < 8.0:  # Minimum duration (expanded to 8s)
                continue
            
            segment_text = self._get_transcript_segment_text(transcript, start_time, end_time)
            
            if len(segment_text.strip()) > 20:  # Minimum text length
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': segment_text,
                    'type': 'window'
            })
        
        return segments
    
    def _get_transcript_segment_text(self, transcript: List[Dict], start_time: float, end_time: float) -> str:
        """Get transcript text for a specific time window from dict format"""
        segment_texts = []
        
        for seg in transcript:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check if this segment overlaps with our window
            if seg_start < end_time and seg_end > start_time:
                # Calculate overlap
                overlap_start = max(seg_start, start_time)
                overlap_end = min(seg_end, end_time)
                
                # If significant overlap, include the text
                if overlap_end - overlap_start > 0.5:  # At least 0.5 seconds overlap
                    segment_texts.append(seg.get('text', ''))
        
        return ' '.join(segment_texts)
    
    def _snap_to_sentence_or_pause(self, t: float, prefer: str = "start") -> float:
        """
        Snap a timestamp toward a nearby 'good' boundary:
        - prefer punctuation end/start in the nearest transcript segment
        - else prefer a local word gap >= 220ms within ±1.0s window
        """
        ep = getattr(self, "_current_episode", None)
        words = getattr(ep, "words", None) or []
        if not words:
            return t

        # 1) word-gap snap
        win = 1.0
        nearest = t
        best_gap = 0.0
        for i in range(1, len(words)):
            a, b = words[i-1], words[i]
            ta = float(a.get("t", 0.0)); tb = float(b.get("t", 0.0))
            if ta < t - win or tb > t + win:
                continue
            gap = tb - ta
            if gap >= 0.22 and abs(((ta+tb)/2.0) - t) < abs(nearest - t):
                nearest = (ta+tb)/2.0
                best_gap = gap

        # 2) punctuation bias (if we have segment text nearby)
        # Optional: look up transcript segment spanning nearest and bias to its end '.' '!' '?'
        return nearest

    def _is_sentence_boundary(self, t: float, radius: float = 0.35) -> bool:
        """
        Heuristic: boundary if there is a word gap >= 220ms or the previous word ends with [.?!]
        within ±radius of t.
        """
        ep = getattr(self, "_current_episode", None)
        words = (getattr(ep, "words", None) or [])
        if not words: return False
        prev = None
        for w in words:
            ws, we = float(w.get("start",0.0)), float(w.get("end",0.0))
            if we <= t - radius: prev = w; continue
            if ws >= t + radius: break
            # if there is a big gap across t
            if prev:
                gap = ws - float(prev.get("end", ws))
                if gap >= 0.22: return True
                if re.search(r"[\.!\?]$", str(prev.get("text") or prev.get("word") or "")): 
                    return True
            prev = w
        return False

    def _low_rms_valley_near(self, t: float, radius: float = 0.25) -> bool:
        """
        Proxy for low RMS: no words (or tiny coverage) in a ±radius window around t,
        or a single inter-word gap ≥ 220ms overlapping t.
        """
        ep = getattr(self, "_current_episode", None)
        words = (getattr(ep, "words", None) or [])
        if not words: return False
        win_s, win_e = t - radius, t + radius
        covered = 0.0
        last_end = None
        for w in words:
            ws, we = float(w.get("start",0.0)), float(w.get("end",0.0))
            if we < win_s: last_end = we; continue
            if ws > win_e: break
            # gap check
            if last_end is not None and ws > last_end and ws < win_e and last_end > win_s:
                if (ws - last_end) >= 0.22: 
                    return True
            last_end = we
            # coverage
            overlap = max(0.0, min(we, win_e) - max(ws, win_s))
            covered += overlap
        return covered <= (2*radius) * 0.35  # <=35% of the window has speech

    def _low_activity_first_second(self, base, s: float, e: float) -> bool:
        """Check if first second has low speech activity (dead air)"""
        end = min(s + 1.0, e)
        span = end - s
        if span <= 0.05: return True
        covered = 0.0
        for seg in base:
            xs, xe = float(seg["start"]), float(seg["end"])
            if xe <= s: continue
            if xs >= end: break
            covered += max(0.0, min(xe, end) - max(xs, s))
        return (span - covered) >= 0.6

    def _base_from_words(self):
        """Fallback: rebuild base segments from episode words"""
        ep = getattr(self, "_current_episode", None)
        words = (getattr(ep, "words", None) or [])
        # Treat words as micro-segments; _get_transcript_segment_text works fine on these
        return [{"start": float(w.get("start", 0.0)),
                 "end":   float(w.get("end",   0.0)),
                 "text":  str(w.get("text") or w.get("word") or "")}
                for w in words]

    def _local_boundary_objective(self, base, s: float, e: float) -> float:
        """Lightweight: favor clean boundary + hook @ start + payoff/curiosity @ end"""
        text_start = self._get_transcript_segment_text(base, s, min(s+2.0, e))
        text_end   = self._get_transcript_segment_text(base, max(e-2.0, s), e)
        score = 0.0
        # boundary bonuses
        if self._is_sentence_boundary(s): score += 0.04
        if self._is_sentence_boundary(e): score += 0.04
        if self._low_rms_valley_near(s, radius=0.25): score += 0.02
        if self._low_rms_valley_near(e, radius=0.25): score += 0.02
        # start hookiness
        if START_HOOK_RE.search(text_start):
            score += 0.06
        # end payoff/curiosity
        if END_CURIOSITY_RE.search(text_end) or re.search(r"[.!?]$", text_end.strip()):
            score += 0.06
        return score

    def _refine_boundaries(self, base, start: float, end: float, total: float) -> tuple[float,float]:
        """Hill-climb around winner to find better boundaries"""
        best_s, best_e = start, end
        best = self._local_boundary_objective(base, start, end)
        for ds in BOUNDARY_OFFSETS:
            for de in BOUNDARY_OFFSETS:
                s = max(0.0, start + ds)
                e = min(total,  end   + de)
                if e - s < CLIP_LEN_MIN or e - s > CLIP_LEN_MAX: 
                    continue
                val = self._local_boundary_objective(base, s, e)
                if val > best:
                    best, best_s, best_e = val, s, e
        return round(best_s, 2), round(best_e, 2)
    
    def _generate_multiscale_candidates(self, source: list, *, step_frac: float = 0.6) -> list[dict]:
        """
        Accepts either a raw transcript (word/segment entries) OR prebuilt segments [{start,end,text},...].
        If it's already segments, we use them directly. Otherwise we build base segments from transcript.
        """
        if not source:
            return []

        def _as_seg_dict(x):
            if isinstance(x, dict) and "start" in x and "end" in x and "text" in x:
                return {"start": float(x["start"]), "end": float(x["end"]), "text": str(x["text"])}
            # Objects with attributes
            if hasattr(x, "start") and hasattr(x, "end") and hasattr(x, "text"):
                return {"start": float(x.start), "end": float(x.end), "text": str(x.text)}
            return None

        # If input already looks like segments, don't call transcript->segments again
        first = _as_seg_dict(source[0])
        if first is not None:
            base = [first] + [sd for sd in (_as_seg_dict(s) for s in source[1:]) if sd]
        else:
            # Treat as transcript; derive base segments using your existing helpers
            derived = self._transcript_to_segments_aligned(source) or self._transcript_to_segments(source)
            base = []
            for s in derived:
                sd = _as_seg_dict(s)
                if sd:
                    base.append(sd)

        # Now sweep multi-scale windows over 'base'
        if not base:
            return []

        total = max(b["end"] for b in base)
        out: list[dict] = []
        BANDS = [(8,12),(12,20),(20,35),(35,60),(60,90)]
        for (lo, hi) in BANDS:
            win = (lo + hi) / 2.0
            # Denser sweep for longer bands to reduce missed 45-75s spans
            step_frac = 0.5 if hi >= 35 else 0.6
            step = max(2.0, win * step_frac)
            t = 0.0
            while t < total - lo + 1e-6:
                start = t
                end = min(t + win, total)
                dur = end - start
                # Apply hard length clamps during generation
                if dur < CLIP_LEN_MIN or dur > CLIP_LEN_MAX:
                    t += step
                    continue
                text = self._get_transcript_segment_text(base, start, end)  # works on [{start,end,text}]
                if text.strip():
                    out.append({"start": start, "end": end, "text": text, "type": f"multiscale_{lo}-{hi}"})
                t += step
        return self._dedup_by_timing(out, time_window=0.15)
    
    def _dedup_by_timing(self, candidates: list[dict], time_window: float = 0.15) -> list[dict]:
        """Remove duplicate candidates based on timing overlap"""
        if not candidates:
            return candidates
        
        # Sort by start time
        sorted_candidates = sorted(candidates, key=lambda x: x.get("start", 0))
        deduped = []
        
        for candidate in sorted_candidates:
            # Check if this overlaps with any already selected
            overlaps = False
            for selected in deduped:
                if (candidate.get("start", 0) < selected.get("end", 0) + time_window and 
                    candidate.get("end", 0) > selected.get("start", 0) - time_window):
                    overlaps = True
                    break
            
            if not overlaps:
                deduped.append(candidate)
        
        return deduped
    
    def _filter_overlapping_candidates(self, candidates: List[Dict], min_gap: float = 15.0) -> List[Dict]:
        """Remove overlapping candidates, keeping highest scoring ones"""
        if not candidates:
            return candidates
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        filtered = []
        
        for candidate in sorted_candidates:
            # Check if this overlaps with any already selected
            overlaps = False
            for selected in filtered:
                # Check for time overlap
                if (candidate['start'] < selected['end'] and 
                    candidate['end'] > selected['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(candidate)
        
        return filtered
    
    def _filter_low_quality(self, candidates: List[Dict], min_score: int = 40) -> List[Dict]:
        """Filter out low-quality candidates"""
        filtered = []
        
        for candidate in candidates:
            # Skip if score too low
            if candidate.get('display_score', 0) < min_score:
                continue
                
            # Skip if text too short or repetitive
            text = candidate.get('text', '')
            if len(text.split()) < 20:
                continue
                
            # Skip if starts mid-sentence (basic check)
            if text and text[0].islower():
                continue
                
            filtered.append(candidate)
        
        # If we filtered too many, relax criteria
        if len(filtered) < 3 and len(candidates) >= 3:
            return candidates[:5]  # Return top 5 regardless
        
        return filtered

    def _get_transcript_segment(self, transcript: List[TranscriptSegment], 
                               start_time: float, end_time: float) -> str:
        """Get transcript text for a specific time window"""
        segment_texts = []
        
        for segment in transcript:
            # More precise overlap check
            if segment.end > start_time and segment.start < end_time:
                segment_texts.append(segment.text)
        
        return ' '.join(segment_texts)

    def select_best_moments(self, moment_scores: List[MomentScore], 
                           target_count: int = 3, min_duration: int = 12, 
                           max_duration: int = 30) -> List[MomentScore]:
        """Select the best moments ensuring diversity and constraints"""
        if not moment_scores:
            return []
        
        # Filter by duration constraints
        valid_moments = [
            m for m in moment_scores 
            if min_duration <= m.duration <= max_duration
        ]
        
        if not valid_moments:
            return []
        
        # Sort by score
        valid_moments.sort(key=lambda x: x.total_score, reverse=True)
        
        # Select moments ensuring temporal and content diversity
        selected_moments = []
        min_gap = 45  # Increased minimum gap between clips
        
        for moment in valid_moments:
            if len(selected_moments) >= target_count:
                break
            
            # Check if this moment is far enough from already selected ones
            is_diverse = True
            for selected in selected_moments:
                time_gap = abs(moment.start_time - selected.start_time)
                if time_gap < min_gap:
                    is_diverse = False
                    break
                
                # Also check for content similarity (basic check)
                if hasattr(moment, 'text') and hasattr(selected, 'text'):
                    moment_words = set(moment.text.lower().split()[:10])  # First 10 words
                    selected_words = set(selected.text.lower().split()[:10])
                    if len(moment_words.intersection(selected_words)) > 5:  # If more than 5 words overlap
                        is_diverse = False
                        break
            
            if is_diverse:
                selected_moments.append(moment)
        
        return selected_moments[:target_count]

    def _safe_get(self, d, key, default=None):
        """Safely get value from dict with normalization for common numeric fields."""
        v = d.get(key, default)
        # normalize common numeric fields
        if key in ("virality_raw", "platform_len", "display_score") and v is None:
            return 0
        return v

    def _format_one(self, c):
        """Format a single clip with bulletproof field access."""
        # Extract virality and individual scores
        virality = float(self._safe_get(c, "virality_raw", 0.0))
        
        # Individual score fields (0-1)
        scores = {
            "hook_score": float(self._safe_get(c, "hook_score", 0.0)),
            "arousal_score": float(self._safe_get(c, "arousal_score", 0.0)),
            "emotion_score": float(self._safe_get(c, "emotion_score", 0.0)),
            "question_score": float(self._safe_get(c, "question_score", 0.0)),
            "payoff_score": float(self._safe_get(c, "payoff_score", 0.0)),
            "info_density": float(self._safe_get(c, "info_density", 0.0)),
            "loopability": float(self._safe_get(c, "loopability", 0.0)),
            "platform_len_match": float(self._safe_get(c, "platform_len_match", 0.0)),
        }
        
        # Extract seed and payoff sentences for enhanced title generation (Phase 3)
        seed_sentence = None
        payoff_sentence = None
        if c.get('seed_idx') is not None:
            # Try to get sentence_spans from the clip or episode context
            sentence_spans = c.get('sentence_spans') or getattr(self, 'sentence_spans', None)
            if sentence_spans and c['seed_idx'] < len(sentence_spans):
                seed_sentence = sentence_spans[c['seed_idx']].text
        payoff_sentence = last_sentence_of(c.get('text', ''))
        
        # Add to clip for title service
        c['seed_sentence'] = seed_sentence
        c['payoff_sentence'] = payoff_sentence
        
        # Generate eager title
        title = None
        title_suggestions = []
        try:
            from services.titles_service import TitlesService
            ts = TitlesService()
            # Use the existing ensure_titles_for_clip method
            title_data = ts.ensure_titles_for_clip(c, platform="shorts")
            if title_data:
                title = title_data.get("chosen") or title_data.get("overlay")
                title_suggestions = title_data.get("variants", [])
        except Exception as e:
            logger.warning("TITLE_GEN_FAIL %s: %s", c.get("id", "unknown"), e)
        
        return {
            "id": c["id"],
            "start": c.get("start"),
            "end": c.get("end"),
            "duration": round(c.get("duration", (c.get("end", 0) - c.get("start", 0))), 1),
            "display_score": int(round(self._safe_get(c, "display_score", 50))),
            "text": (c.get("text") or "")[:240],
            
            # Virality fields (both 0-1 and 0-100)
            "virality": virality,
            "virality_calibrated": virality,
            "virality_pct": int(round(virality * 100)),
            
            # Platform fit fields
            "platform_fit": float(self._safe_get(c, "platform_len", 0.0)),
            "platform_fit_pct": int(round(self._safe_get(c, "platform_len", 0.0) * 100)),
            
            # Individual scores (0-1)
            **scores,
            
            # Score percentages (belt-and-suspenders)
            "score_pct": {k: int(round(v * 100)) for k, v in scores.items()},
            
            # Title fields
            "title": title,
            "title_suggestions": title_suggestions,
            
            # keep any extra fields safely
            "meta": {k: v for k, v in c.items() if k not in {"id", "start", "end", "duration", "text"}}
        }

    def _minimal_candidate_dict(self, clip):
        """Lowest-common-denominator fields so the UI can render something."""
        return {
            "id": clip["id"],
            "start": clip.get("start"),
            "end": clip.get("end"),
            "duration": round(clip.get("duration", (clip.get("end", 0) - clip.get("start", 0))), 1),
            "display_score": int(round(clip.get("display_score", 50))),
            "virality_pct": int(round(clip.get("virality_raw", 0.0) * 100)),
            "platform_fit_pct": int(round(clip.get("platform_len", 0.0) * 100)),
            "text": clip.get("text", "")[:240],
        }

    def _format_candidates_safe(self, finals, *args, **kwargs):
        """Format candidates with fallback to minimal dicts on error."""
        try:
            formatted = [self._format_one(c) for c in finals]
            return formatted
        except Exception:
            logger.exception("FORMAT_CANDIDATES_FAIL: falling back to minimal dicts")
            return [self._minimal_candidate_dict(c) for c in finals]

    async def get_candidates(self, episode_id: str, platform: str = "tiktok_reels", genre: str = None) -> Tuple[List[Dict], Dict]:
        """Get AI-scored clip candidates for an episode with platform-neutral selection and post-selection platform recommendations"""
        
        # Platform flags: always defined unconditionally at the start
        pl_v2_weight, platform_protect, length_agnostic = _get_platform_mode()
        logger.info("PLATFORM_MODE: pl_v2_weight=%.2f, platform_protect=%s, length_agnostic=%s", 
                   pl_v2_weight, platform_protect, length_agnostic)
        
        # Initialize final_clips for salvage guard
        final_clips = []
        
        try:
            from services.secret_sauce_pkg import (
                find_viral_clips_enhanced, resolve_platform, detect_podcast_genre
            )
            from services.secret_sauce_pkg.features import (
                compute_v_core, PLATFORM_NEUTRAL_SELECTION, FINISHED_THOUGHT_REQUIRED,
                LENGTH_SEARCH_BUCKETS, LENGTH_MAX_HARD
            )
            from services.platform_recommender import add_platform_recommendations_to_clips

            # Get episode with words loaded (async version)
            episode = await self.episode_service.get_episode(episode_id, with_words=True)
            if not episode or not episode.transcript:
                logger.error(f"Episode {episode_id} not found or has no transcript")
                return [], {"reason": "episode_not_found", "episode_id": episode_id}
            
            # Fallback word loading if not present
            if not getattr(episode, "words", None):
                loaded = self.episode_service._load_words_from_disk(episode_id)
                if loaded:
                    episode.words = loaded
                    episode.eos_from_words = True
                    logger.info("CLIP_INPUT: words=present count=%d (from disk)", len(loaded))
                else:
                    logger.warning("CLIP_INPUT: words missing (episode_id=%s)", episode_id)
            else:
                logger.info("CLIP_INPUT: words=present count=%d (from memory)", len(episode.words))
            
            # Set current episode for boundary snapper
            self._current_episode = episode

            # Resolve platform
            backend_platform = resolve_platform(platform)

            # Detect or resolve genre first
            detected_genre = detect_podcast_genre(episode.transcript)
            if genre:
                # User explicitly selected a genre
                final_genre = genre
            else:
                # Use auto-detected genre
                final_genre = detected_genre

            logger.info(f"Using genre: {final_genre} (detected: {detected_genre}, user_selected: {genre})")
            
            # Log length policy
            logger.info(f"LENGTH_POLICY: search={LENGTH_SEARCH_BUCKETS}, clamp={LENGTH_MAX_HARD}")
            logger.info(f"QUALITY_GATE: finished_required={FINISHED_THOUGHT_REQUIRED}")

            # Convert transcript to segments using intelligent moment detection
            segments = self._transcript_to_segments(episode.transcript, genre=final_genre, platform=platform, words=getattr(episode, 'words', None))
            
            # Store base segments for boundary refinement
            self._base_segments = segments.copy() if segments else []
            
            # Store original segments as fallback
            original_segments = segments.copy() if segments else []
            
            # NEW: expand to multi-scale candidates with fail-safe
            try:
                segments = self._generate_multiscale_candidates(segments)
                logger.info(f"Generated {len(segments)} multiscale candidates")
            except Exception as e:
                logger.exception("Multiscale generation failed; falling back to dynamic segments: %s", e)
                # Fallback to previously computed dynamic segments
                segments = original_segments
                logger.info(f"Using fallback segments: {len(segments)} candidates")

            # Build EOS index early to determine fallback mode
            from services.secret_sauce_pkg.features import build_eos_index
            episode_words = getattr(episode, 'words', None) if episode else None
            episode_raw_text = getattr(episode, 'raw_text', None) if episode else None
            eos_times, word_end_times, eos_source = build_eos_index(segments, episode_words, episode_raw_text)
            
            # One-time fallback mode decision per episode (freeze this value)
            eos_density = len(eos_times) / max(len(word_end_times), 1)
            episode_fallback_mode = eos_density < 0.020
            logger.info(f"EOS_UNIFIED: count={len(eos_times)}, src={eos_source}, density={eos_density:.3f}, fallback={episode_fallback_mode}")

            # Use the enhanced viral clips pipeline with all Phase 1-3 improvements
            logger.info(f"Using enhanced viral clips pipeline with {len(segments)} segments")
            try:
                viral_result = find_viral_clips_enhanced(
                    segments,
                    episode.audio_path,
                    genre=final_genre,
                    platform=backend_platform,
                    fallback_mode=episode_fallback_mode,
                    effective_eos_times=eos_times,
                    effective_word_end_times=word_end_times,
                    eos_source=eos_source,
                    episode_words=episode_words,
                )
                if not viral_result or not viral_result.get("clips"):
                    raise RuntimeError("ENHANCED_EMPTY")
            except Exception:
                logger.error("CANDIDATE_PIPELINE_FAIL: enhanced pipeline threw; invoking legacy fallback")
                logger.exception("CANDIDATE_PIPELINE_FAIL_STACK")
                # Legacy fallback on segments → simple pass-through as minimal candidates
                legacy_candidates = []
                for seg in segments or []:
                    try:
                        s = float(seg.get("start", 0.0))
                        e = float(seg.get("end", s))
                        legacy_candidates.append({
                            "start": s,
                            "end": e,
                            "text": seg.get("text", ""),
                            "duration": max(0.0, e - s),
                            "features": {},
                            "meta": {"legacy_fallback": True},
                        })
                    except Exception:
                        continue
                viral_result = {"clips": legacy_candidates, "meta": {"fallback": True}}
            
            # Debug: log ft_status from enhanced pipeline
            if viral_result and "clips" in viral_result:
                ft_statuses = [c.get("ft_status", "missing") for c in viral_result.get("clips", [])]
                logger.info(f"Enhanced pipeline ft_statuses: {ft_statuses}")
            
            # Use authoritative fallback mode from enhanced pipeline
            episode_fallback_mode = authoritative_fallback_mode(viral_result)
            
            # Optional emergency override while testing:
            import os
            env_force = os.getenv("FT_FORCE_FALLBACK")
            if env_force is not None:
                episode_fallback_mode = (env_force == "1")
            
            logger.info(f"GATE_MODE: fallback={episode_fallback_mode} (authoritative from enhanced pipeline)")
            
            # Compute gate_mode for adaptive gate
            gate_mode = {
                # authoritative from enhanced pipeline; keep your existing signal if set elsewhere
                "fallback": episode_fallback_mode,
                # make tail-close more forgiving in fallback/balanced runs
                "tail_close_sec": float(os.getenv("TAIL_CLOSE_SEC", "0.60")),
            }
            
            if "error" in viral_result:
                logger.error(f"Enhanced viral clips pipeline failed: {viral_result.get('error', 'Unknown error')}")
                return [], {"reason": "viral_pipeline_error", "error": viral_result.get('error', 'Unknown error'), "episode_id": episode_id}
            
            clips = viral_result.get('clips', [])
            logger.info(f"Found {len(clips)} viral clips using enhanced pipeline")
            
            # Convert episode transcript to full text for display
            full_episode_transcript = ""
            if episode.transcript:
                full_episode_transcript = " ".join([seg.text for seg in episode.transcript if hasattr(seg, 'text') and seg.text])
            
            # EOS index already built above for fallback mode determination
            
            # Sanity assertions
            assert eos_times is not None, "EOS times should never be None"
            if len(eos_times) == 0:
                logger.warning("EOS_FALLBACK_ONLY: No EOS markers found, using fallback mode")
            
            # Add fallback flag to each candidate for hook honesty cap
            for clip in clips:
                clip_meta = clip.get("meta") or {}
                clip_meta["is_fallback"] = bool(episode_fallback_mode)
                clip["meta"] = clip_meta
            
            # Convert to candidate format with title generation and grades
            try:
                candidates = format_candidates(clips, final_genre, backend_platform, episode_id, full_episode_transcript, episode)
                logger.info(f"Formatted {len(candidates)} candidates")
            except Exception as e:
                logger.exception("FORMAT_CANDIDATES_ERROR: %s", e)
                # Fallback: persist clips with safe titles
                candidates = []
                for c in clips:
                    text = c.get("text") or c.get("display_text") or ""
                    from services.candidate_formatter import _fallback_title
                    candidates.append({**c, "title": _fallback_title(text)})
                logger.warning(f"Used fallback titles for {len(candidates)} clips due to formatting error")
            
            # PRE-NMS: Log top 10 before NMS with proper tie-breaking
            def sort_key(c):
                return (
                    round(c.get('final_score', 0), 3),
                    round(c.get('platform_length_score_v2', 0.0), 3),
                    round(c.get('end', 0) - c.get('start', 0), 2)
                )
            
            for c in sorted(candidates, key=sort_key, reverse=True)[:10]:
                logger.info("PRE-NMS: fs=%.2f dur=%.1f pl_v2=%.2f text='%s'",
                    c.get("final_score", 0), c.get("end", 0) - c.get("start", 0), 
                    c.get("platform_length_score_v2", 0.0),
                    (c.get("text", "")[:50]).replace("\n", " ")
                )
            
            # A) Prove what each top clip scored (one-line instrumentation)
            for i, c in enumerate(sorted(candidates, key=sort_key, reverse=True)[:10]):
                # Enhanced diagnostics
                text = c.get("text", "")
                is_q = text.strip().endswith("?")
                has_payoff = c.get("payoff_score", 0.0) >= 0.20
                looks_ad = c.get("ad_penalty", 0.0) >= 0.3
                words = c.get("text_length", 0)
                final_score = c.get("final_score", 0)
                
                # Verification asserts
                if is_q and words < 12 and c.get("payoff_score", 0) < 0.20 and not c.get("_has_answer", False):
                    assert final_score <= 0.55 + 1e-6, f"Question cap failed: {final_score} '{text[:60]}'"
                
                logger.info(
                    "Top #%d: score=%.2f w=%s hook=%.2f arous=%.2f arous_src=%s payoff=%.2f info=%.2f ql=%.2f pl_v2=%.2f ad_pen=%.2f q=%s payoff_ok=%s ad=%s caps=%s text='%s'",
                    i+1,
                    final_score,
                    words,
                    c.get("hook_score", 0),
                    c.get("arousal_score", 0),
                    c.get("_arousal_src", "v1"),
                    c.get("payoff_score", 0),
                    c.get("info_density", 0),
                    c.get("q_list_score", 0),
                    c.get("platform_length_score_v2", 0),
                    c.get("ad_penalty", 0),
                    is_q,
                    has_payoff,
                    looks_ad,
                    c.get("flags", {}).get("caps_applied", []),
                    (text[:80]).replace("\n", " ")
                )
            
            # Debug: Log candidate scores
            for i, candidate in enumerate(candidates[:3]):  # Log first 3 candidates
                logger.info(f"Candidate {i}: display_score={candidate.get('display_score', 'MISSING')}, text_length={len(candidate.get('text', '').split())}")
            
            # Collapse consecutive question runs (keep only the strongest)
            from services.secret_sauce_pkg.scoring_utils import collapse_question_runs
            candidates = collapse_question_runs(candidates)
            logger.info(f"After question collapse: {len(candidates)} candidates")
            
            # Filter overlapping candidates
            filtered_candidates = filter_overlapping_candidates(candidates)
            logger.info(f"After overlap filtering: {len(filtered_candidates)} candidates")
            
            # Normalize candidates before any gates
            filtered_candidates = [normalize_ft_status(c) for c in filtered_candidates]
            
            # Duration/np.float64 guard (prevents weird types later)
            try:
                import numpy as np
                for c in filtered_candidates:
                    if isinstance(c.get("duration"), np.floating):
                        c["duration"] = float(c["duration"])
            except Exception:
                pass
            
            # Create reserve pool before quality gates (for auto-relax)
            import heapq
            RESERVE_TOP_K = 24
            reserve_pool = heapq.nlargest(RESERVE_TOP_K, filtered_candidates, key=lambda c: c.get("utility_pre_gate", c.get("final_score", 0.0)))
            logger.info(f"POOL: pre_gate={len(filtered_candidates)} reserve={len(reserve_pool)}")
            
            # Apply quality filtering with safety net
            quality_filtered = filter_low_quality(filtered_candidates, min_score=15)
            logger.info(f"QUALITY_FILTER: kept={len(quality_filtered)} of {len(filtered_candidates)}")
            
            # CANDIDATES_BEFORE_SAFETY: Log before safety pass
            logger.info(f"CANDIDATES_BEFORE_SAFETY: {len(quality_filtered)}")
            
            # --- Safety / shrink / finish-thought pass (updates ft_status) ---
            updated = 0
            for c in quality_filtered:
                meta = run_safety_or_shrink(c, eos_times, word_end_times, backend_platform, episode_fallback_mode)
                if isinstance(meta, dict) and meta.get("status"):
                    c["ft_status"] = meta["status"]                 # 'finished' | 'sparse_finished' | 'unresolved' | 'extended'
                    c["ft_meta"] = {**c.get("ft_meta", {}), **meta}
                    updated += 1
            logger.info(f"SAFETY_PASS: updated {updated}/{len(quality_filtered)}")
            
            # --- Apply quality gate with auto-relaxation ---
            quality_filtered = apply_quality_gate(quality_filtered, mode="strict" if not episode_fallback_mode else "fallback")
            
            # Apply adaptive gate if we have too few clips (use reserve pool)
            if len(quality_filtered) < 3:
                quality_filtered = adaptive_gate(reserve_pool, min_count=3, gate_mode=gate_mode)
                logger.info(f"POOL: STRICT={len(quality_filtered)} BALANCED(from=reserve)={len(quality_filtered)}")
            
            # Finish polish: extend borderline clips to coherent stops
            quality_filtered = [_finish_polish(c, LOG=logger) for c in quality_filtered]
            
            # One-line telemetry for monitoring
            logger.info(
                "TELEMETRY: eos=%d(%.3f) fallback=%s | seeds=%d → strict=%d → balanced=%d | "
                "rescued=%d | finals=%d durs=%s",
                len(eos_times), eos_density, episode_fallback_mode,
                len(reserve_pool), len(quality_filtered), len(quality_filtered),
                0, len(quality_filtered), [round(_get_dur(c),1) for c in quality_filtered]
            )
            
            # Apply temporal deduplication to remove near-duplicate clips
            iou = float(os.getenv("DEDUP_IOU_THRESHOLD", "0.85"))
            target_len = float(os.getenv("DEDUP_TARGET_LEN", "18.0"))
            before = len(quality_filtered)
            quality_filtered = _dedup_by_time_iou(quality_filtered, iou_thresh=iou, target_len=target_len)
            after = len(quality_filtered)
            logger.info(f"DEDUP_BY_TIME: {before} → {after} kept ({before-after} removed), thresh={iou:.2f}")
            
            # Apply text-based deduplication to remove semantic duplicates
            text_thresh = float(os.getenv("DEDUP_TEXT_THRESHOLD", "0.90"))
            before_text = len(quality_filtered)
            quality_filtered = _dedup_by_text(quality_filtered, text_thresh=text_thresh)
            after_text = len(quality_filtered)
            logger.info(f"DEDUP_BY_TEXT: {before_text} → {after_text} kept ({before_text-after_text} removed), thresh={text_thresh:.2f}")
            
            # Apply duration clamping to prevent overlong clips (platform-neutral)
            from services.secret_sauce_pkg.features import LENGTH_MAX_HARD
            platform_max_sec = LENGTH_MAX_HARD  # 90.0s
            clamped_count = 0
            for c in quality_filtered:
                old_dur = c.get("end", 0) - c.get("start", 0)
                if old_dur > platform_max_sec:
                    c["end"] = c.get("start", 0) + platform_max_sec
                    new_dur = c["end"] - c.get("start", 0)
                    logger.info(f"CLIP_DURATION_CLAMP: {c.get('id', 'unknown')} {old_dur:.2f}s → {new_dur:.2f}s")
                    clamped_count += 1
            if clamped_count > 0:
                logger.info(f"CLIP_DURATION_CLAMP: {clamped_count} clips clamped to {platform_max_sec}s max")
            
            # POST-NMS: Log final durations (cast to float to avoid numpy dtype issues)
            safe_durs = [float(round(c.get("end", 0) - c.get("start", 0), 1)) for c in quality_filtered]
            logger.info("POST-NMS: durs=%s", safe_durs)
            
            # Diversity check: warn if all candidates have same duration
            durs = [round(c.get("end", 0) - c.get("start", 0), 1) for c in quality_filtered]
            if len(quality_filtered) >= 6 and len(set(durs)) < 2:
                logger.warning("DIVERSITY: all candidates have same duration=%s; favoring longer pl_v2 ties", durs[0] if durs else "unknown")
            
            # Snapshot after quality filtering, before selection
            post_nms = list(quality_filtered) if quality_filtered else []
            
            # Log length policy
            logger.info("LENGTH_POLICY: max=90.0, quotas=off, gaussian_weight=0.00, notch_weight=0.00")
            
            # Pure top-K selection by virality (no forced buckets)
            finals = pure_topk_pick(quality_filtered, reserve_pool, want=10, LOG=logger)
            
            # Enforce final soft floor after all gates
            finals = enforce_soft_floor(finals, min_count=3)
            
            # Snapshot after selection, before later transforms
            finals_pre_limits = list(finals) if finals else []
            
            # Soft relax for empty results
            if not finals:
                from config.settings import FT_SOFT_RELAX_ON_ZERO, FT_SOFT_RELAX_TOPK
                if FT_SOFT_RELAX_ON_ZERO:
                    # Get all candidates that passed initial filtering
                    pool = sorted(
                        quality_filtered,
                        key=lambda c: (
                            c.get("rank_score", c.get("display_score", 0.0)),
                            c.get("features", {}).get("payoff_score", 0.0)
                        ),
                        reverse=True,
                    )
                    # Filter out ads from salvage pool
                    pool = [c for c in pool if not c.get("features", {}).get("ad", False)]
                    finals = pool[:FT_SOFT_RELAX_TOPK]
                    logger.warning("SOFT_SALVAGE: rescued %d clips", len(finals))
                else:
                    logger.warning("REASON=EMPTY_AFTER_SALVAGE: returning empty list")
                    return [], {"reason": "empty_after_salvage", "episode_id": episode_id}
            
            # ---- FINISHED-THOUGHT BACKSTOP -----------------------------------------
            # If finals contain zero "finished_thought", promote the best finished clip
            # from prior survivors by swapping out the weakest final (bounded swap).
            try:
                has_finished = any(bool(c.get("finished_thought")) for c in finals)
                if not has_finished:
                    import re
                    _BAN_CHATTER = re.compile(
                        r"\b(appreciate|honored|thanks?|thank you|glad|pleased|great to be here|"
                        r"see you|next week|today we('| )?re|interview(ed)?|enjoyed)\b",
                        re.I,
                    )
                    pool = []
                    for c in candidates:
                        if not c.get("finished_thought"):
                            continue
                        text = (c.get("transcript") or c.get("text") or "").lower()
                        if _BAN_CHATTER.search(text):
                            continue  # avoid pleasantries/outro banter
                        pool.append(c)
                    if pool:
                        # Prefer payoff_ok=True, then by score
                        pool.sort(key=lambda c: (
                            0 if not c.get("payoff_ok") else 1,
                            float(c.get("final_score", c.get("score", 0.0)))
                        ), reverse=True)
                        candidate = pool[0]
                        if finals:
                            finals.sort(key=lambda c: float(c.get("final_score", c.get("score", 0.0))))
                            weakest = finals[0]
                            if float(candidate.get("final_score", candidate.get("score", 0.0))) >= \
                               float(weakest.get("final_score", weakest.get("score", 0.0))) - 0.05:
                                finals[0] = candidate
                                logger.info("QUALITY_BACKSTOP: promoted finished_thought clip %s", candidate.get("id"))
            except Exception as e:
                logger.error("QUALITY_BACKSTOP_ERROR: %s", e)
            # -----------------------------------------------------------------------
            
            # Mark all selected clips with protected status
            for c in finals:
                c["protected"] = c.get("protected", False)
            
            # Stable final ordering: protected clips first, then by score
            finals.sort(key=lambda c: (c.get("protected", False), c.get("final_score", 0)), reverse=True)
            
            # Apply temporal deduplication to remove near-duplicate clips
            if finals:
                iou_thresh = float(os.getenv("DEDUP_IOU_THRESHOLD", "0.85"))
                platform_target = 18.0  # Default target duration
                before_count = len(finals)
                finals = _dedup_by_time_iou(finals, iou_thresh=iou_thresh, target_len=platform_target)
                after_count = len(finals)
                if before_count != after_count:
                    logger.info(f"DEDUP_BY_TIME: {before_count} → {after_count} kept ({before_count - after_count} removed), thresh={iou_thresh}")
            
            # Debug: Log final saved set with durations and protected flags
            final_info = []
            for c in finals:
                dur = round(c.get('end', 0) - c.get('start', 0), 1)
                # Apply micro-jitter to display duration to break identical lengths
                display_dur = apply_micro_jitter_to_display(dur)
                prot = "prot=True" if c.get("protected", False) else ""
                final_info.append(f"{display_dur}s {prot}".strip())
            logger.info(f"FINAL_PICK_BEFORE_LIMITS: n={len(finals)} :: [{', '.join(final_info)}]")
            
            # Tighten selection after gating
            finals = tighten_selection(finals)
            
            # Apply duration diversity (soft target mix)
            finals = apply_duration_diversity(finals)
            
            # Dedup before saving
            finals = unique_by_id(finals)  # preserve order
            finals = dedup_by_timing(finals, time_window=0.15)  # remove clones within ±150ms
            
            # Safety pass already completed before quality gate
            candidates_after = len(finals)
            
            # Pre-limit backup: stash finals before any potential zero-out
            pre_limit_finals = list(finals) if finals else []
            
            # Compute finished-like for resilience when finished_required is True but none are marked finished
            finished_like = [c for c in finals if _is_finished_like(c)]
            
            # Check if we require finished but ended with none
            finals_finished = [c for c in finals if c.get("finished_thought", False)]
            finished_required = FINISHED_THOUGHT_REQUIRED
            
            # If we require finished but ended with none, try finished-like first
            if finished_required and not finals_finished:
                if finished_like:
                    logger.warning("FINISHED_LIKE_RESCUE: using finished-like items when finished_required=True but finished=0")
                    finals = finished_like
                    finals_finished = finished_like  # treat as finished-like for return symmetry
                else:
                    # Optional safety net via env flag
                    if _SOFT_RELAX:
                        logger.warning("SOFT_RELAX_RESCUE: finished_required=True but no finished-like items, applying soft relax")
                        finals = _soft_relax_on_zero(pre_limit_finals, min_keep=3)
                        finals_finished = []  # still be honest: these aren't truly "finished"
            
            # If still empty, fall back to what we had before any post-limit gates
            if not finals and pre_limit_finals:
                logger.warning("FALLBACK_RESCUE: using pre-limit finals as last resort")
                finals = pre_limit_finals
                # finals_finished can remain as-is; downstream code should not crash if empty
            
            # Pre-save logging with enhanced diagnostics
            logger.info(f"PRE_SAVE: final_count={len(finals)} ids={[c.get('id', 'unknown') for c in finals]}")
            logger.info(f"PRE_SAVE_DEBUG: finals={len(finals)} finished={len(finals_finished)} finished_like={len([c for c in finals if _is_finished_like(c)])} finished_required={finished_required}")
            logger.debug(f"PRE_SAVE_DEBUG: pre_limit_backup={len(pre_limit_finals)}, soft_relax_enabled={_SOFT_RELAX}")
            
            # Remove hard assert on zero; degrade gracefully
            if candidates_after == 0:
                logger.error("NO_CANDIDATES: returning empty set after gating")
                
                # Try soft-relax rescue if enabled
                if _SOFT_RELAX and pre_limit_finals:
                    logger.warning("SOFT_RELAX: attempting rescue from pre-limit finals")
                    rescued = _soft_relax_on_zero(pre_limit_finals, min_keep=3)
                    if rescued:
                        logger.warning("SOFT_RELAX: rescued %d finals from pre-limit pool", len(rescued))
                        # Calculate finished count for rescued items
                        rescued_finished = [c for c in rescued if c.get("finished_thought", False)]
                        return rescued, rescued_finished
                
                return [], []
            
            # Check finished count and apply early salvage if needed
            pool_finished_count = sum(1 for c in finals if c.get("finished_thought", False))
            finished_ratio_strict = pool_finished_count / max(len(finals), 1)
            logger.info(f"FT_COVERAGE: pre_gate_finished={pool_finished_count}/{len(finals)}")
            logger.info(f"GATE_RESULT: kept={len(finals)} (finished={pool_finished_count}, unfinished={len(finals) - pool_finished_count})")
            
            # Early salvage if finished ratio is too low
            if finished_ratio_strict < 0.15:
                logger.warning(f"EARLY_SALVAGE: finished_ratio_strict={finished_ratio_strict:.2f} < 0.15, running salvage_to_EOS")
                from services.secret_sauce_pkg.features import extend_to_coherent_end
                
                # Run salvage on top 2×K seeds
                top_seeds = sorted(finals, key=lambda c: c.get("v_core_score", c.get("final_score", 0)), reverse=True)[:6]
                early_salvaged = []
                
                for seed in top_seeds:
                    if not seed.get("finished_thought", False):
                        extended = extend_to_coherent_end(
                            seed, eos_times, word_end_times, 
                            max_extend=8.0, platform=backend_platform
                        )
                        if extended and extended.get("end", 0) > seed.get("end", 0):
                            extended["finished_thought"] = True
                            extended["finish_reason"] = "early_salvage_eos"
                            # Re-score with V_core
                            from services.secret_sauce_pkg.features import compute_v_core
                            start_s = extended.get("start", 0)
                            end_s = extended.get("end", 0)
                            extended["v_core_score"] = compute_v_core(extended, start_s, end_s)
                            early_salvaged.append(extended)
                            logger.debug(f"EARLY_SALVAGE: extended {seed.get('id', 'unknown')} by {end_s - seed.get('end', 0):.1f}s")
                
                # Add early salvaged to finals
                finals.extend(early_salvaged)
                pool_finished_count = sum(1 for c in finals if c.get("finished_thought", False))
                logger.info(f"FT_COVERAGE: post_early_salvage_finished={pool_finished_count}/{len(finals)}")
            
            if pool_finished_count == 0:
                logger.warning("SOFT_RELAX: pool_finished_count=0, running salvage_to_EOS on top seeds")
                from services.secret_sauce_pkg.features import extend_to_coherent_end
                
                # Run salvage on top K seeds
                top_seeds = sorted(finals, key=lambda c: c.get("v_core_score", c.get("final_score", 0)), reverse=True)[:6]
                salvaged = []
                
                for seed in top_seeds:
                    if not seed.get("finished_thought", False):
                        extended = extend_to_coherent_end(
                            seed, eos_times, word_end_times, 
                            max_extend=8.0, platform=backend_platform
                        )
                        if extended and extended.get("end", 0) > seed.get("end", 0):
                            extended["finished_thought"] = True
                            extended["finish_reason"] = "salvage_eos"
                            # Re-score with V_core
                            from services.secret_sauce_pkg.features import compute_v_core
                            start_s = extended.get("start", 0)
                            end_s = extended.get("end", 0)
                            extended["v_core_score"] = compute_v_core(extended, start_s, end_s)
                            salvaged.append(extended)
                            logger.debug(f"SOFT_RELAX: salvaged {seed.get('id', 'unknown')} by {end_s - seed.get('end', 0):.1f}s")
                
                # Add salvaged to finals
                finals.extend(salvaged)
                pool_finished_count = sum(1 for c in finals if c.get("finished_thought", False))
                logger.info(f"FT_COVERAGE: post_salvage_finished={pool_finished_count}/{len(finals)}")
                
                # If still 0, apply FT_SOFT_RELAX_ON_ZERO
                if pool_finished_count == 0:
                    logger.warning("SOFT_RELAX: still 0 finished, applying FT_SOFT_RELAX_ON_ZERO")
                    finished_like = [c for c in finals if _is_finished_like(c)]
                    # Filter out ads from finished_like
                    finished_like = [c for c in finished_like if not c.get("features", {}).get("ad", False)]
                    if finished_like:
                        finals = finished_like[:min(3, len(finished_like))]
                        logger.info(f"SOFT_RELAX: using {len(finished_like)} finished_like candidates")
            
            # Add ranking and default clip selection (platform-neutral)
            ranked_clips = rank_clips(finals, platform_neutral=PLATFORM_NEUTRAL_SELECTION)
            
            # Apply payoff rescue to boost clips with strong CTAs, punchlines, or insights
            from services.payoff_rescue import apply_payoff_rescue
            rescue_items = [{"calibrated": c.get("final_score", 0.0), "features": c.get("features", {}), "clip": c} for c in ranked_clips]
            rescue_items.sort(key=lambda x: x["calibrated"], reverse=True)
            rescue_items = apply_payoff_rescue(rescue_items)
            
            # Update clips with rescued scores and re-sort
            for item in rescue_items:
                clip = item["clip"]
                if "calibrated_rescued" in item:
                    clip["final_score_rescued"] = item["calibrated_rescued"]
                    clip["rescue_reason"] = item.get("rescue_reason")
            
            # Re-sort by rescued score if any rescues were applied
            if any("calibrated_rescued" in item for item in rescue_items):
                ranked_clips.sort(key=lambda c: c.get("final_score_rescued", c.get("final_score", 0.0)), reverse=True)
            
            # Apply anti-uniform tiebreak if we have a reserve pool
            if hasattr(self, 'reserve_pool') and self.reserve_pool:
                ranked_clips = _anti_uniform_tiebreak(ranked_clips, self.reserve_pool, window=0.5, LOG=logger)
            
            # Add platform recommendations after selection
            ranked_clips = add_platform_recommendations_to_clips(ranked_clips)
            
            # Apply semantic MMR for diversity (content-based only in length-agnostic mode)
            from services.quality_filters import mmr_select_semantic
            if length_agnostic:
                # In length-agnostic mode, use content diversity only (no duration bias)
                ranked_clips = mmr_select_semantic(ranked_clips, K=min(6, len(ranked_clips)), lam=0.8)  # Higher content diversity
            else:
                ranked_clips = mmr_select_semantic(ranked_clips, K=min(6, len(ranked_clips)), lam=0.7)
            
            
            # Platform flags already defined at function start
            
            if length_agnostic:
                logger.info("LENGTH_AGNOSTIC: enabled - best clip wins regardless of duration")
            
            # Apply platform length scoring if enabled
            if pl_v2_weight > 0:
                for clip in ranked_clips:
                    pl_v2 = clip.get("platform_length_score_v2", 0.0)
                    # Blend platform score with final score
                    clip["final_score"] = clip.get("final_score", 0) * (1 - pl_v2_weight) + pl_v2 * pl_v2_weight
                    clip["display_score"] = clip["final_score"]
            
            # Log platform-aware selection info
            logger.info(f"SELECTION: platform_neutral={PLATFORM_NEUTRAL_SELECTION}")
            logger.info(f"PLATFORM_MODE: pl_v2_weight={pl_v2_weight}, platform_protect={platform_protect}")
            if ranked_clips:
                logger.info(f"PLATFORM_REC: {ranked_clips[0].get('platform_recommendations', [])[:3]}")
            
            # Save guard - never crash on empty clips
            if not ranked_clips:
                logger.warning(f"FINAL_SAVE_EMPTY: episode={episode_id}")
                # still return the correct shape (two lists)
                return [], []
            
            # Log final summary with pick trace
            finals_finished = sum(1 for c in ranked_clips if c.get("finished_thought", False))
            logger.info(f"PICK_RESULT: finals={len(ranked_clips)}, finals_finished={finals_finished}")
            
            if ranked_clips:
                _log_finals_summary(ranked_clips, logger)
                _log_pick_trace(ranked_clips, logger)
            
            default_clip_id = choose_default_clip(ranked_clips)
            
            
            # --- boundary refinement pass (safe, best-effort) -----------------------------
            refined = []
            
            # Explicit words loading with fallback to disk
            episode_words = getattr(episode, "words", None)
            if not episode_words:
                # Try to load from disk as fallback
                try:
                    from services.episode_service import EpisodeService
                    episode_service = EpisodeService()
                    episode_words = episode_service._load_words_from_disk(episode.id)
                    if episode_words:
                        logger.info("BOUNDARY_REFINEMENT: loaded %d words from disk for episode %s", len(episode_words), episode.id)
                except Exception as e:
                    logger.warning("BOUNDARY_REFINEMENT: failed to load words from disk: %s", e)
                    episode_words = []
            
            logger.info("BOUNDARY_REFINEMENT: using %d episode words", len(episode_words or []))
            
            for clip in ranked_clips:
                try:
                    # Store original end time for A/B metrics
                    old_end = clip.get("end", 0.0)
                    
                    # Step 1: Hill-climb refinement (increased max_nudge for better context)
                    s1, e1 = refine_bounds_with_hill_climb(clip, episode, max_nudge=8.0, step=0.05)
                    
                    # Step 1.5: Apply soft start pad and gentle filler trim (Phase 3)
                    if episode_words:
                        # Preserve all metadata from original clip
                        temp_clip = dict(clip)  # Copy all fields including seed_idx, seed_sentence, etc.
                        temp_clip['start'] = s1
                        temp_clip['end'] = e1
                        temp_clip = soft_start_pad_and_trim(temp_clip, episode_words, pad=settings.SEED_START_PAD_SEC, max_trim=0.6)
                        s1, e1 = temp_clip['start'], temp_clip['end']
                        
                        # Preserve the refined metadata back to the original clip
                        clip.update(temp_clip)
                    
                    # Track end delta for A/B metrics
                    clip["end_delta_ms"] = int(1000 * (e1 - old_end))
                    
                    # Step 2: Clean start/end guards
                    if not episode_words:
                        logger.warning("BOUNDARY_REFINEMENT: no episode words; skipping clean for this clip")
                        s2, e2 = s1, e1
                    else:
                        s2, e2 = _clean_start_end({'start': s1, 'end': e1}, episode_words)
                    clip["start"], clip["end"] = s2, e2
                    clip["duration"] = round(max(0.0, e2 - s2), 2)
                    # Add timecode formatting
                    clip["start_tc"] = _fmt_tc(s2)
                    clip["end_tc"] = _fmt_tc(e2)
                    clip["start_sec"] = s2
                    clip["end_sec"] = e2
                    clip["display_range"] = f"{_fmt_ts(s2)}–{_fmt_ts(e2)}"
                    refined.append(clip["id"])
                except Exception as ex:
                    logger.exception("REFINE_BOUNDS_ERROR: %s", ex)
                    # keep original clip untouched
            
            # Step 3: Smart extension to natural sentence endings (if words available)
            if episode_words:
                from services.util import extend_to_natural_end
                extended_count = 0
                for clip in ranked_clips:
                    old_end = clip.get("end", 0.0)
                    clip = extend_to_natural_end(clip, episode_words, max_extend_sec=3.0)
                    if clip.get("extended"):
                        new_end = clip.get("end", 0.0)
                        delta = clip.get("extension_delta", 0.0)
                        logger.debug("BOUNDARY_REFINEMENT: extended end %.2f->%.2f (+%.2fs)", old_end, new_end, delta)
                        extended_count += 1
                        # Update derived fields
                        clip["duration"] = round(max(0.0, new_end - clip.get("start", 0.0)), 2)
                        clip["end_tc"] = _fmt_tc(new_end)
                        clip["end_sec"] = new_end
                        clip["display_range"] = f"{_fmt_ts(clip.get('start', 0.0))}–{_fmt_ts(new_end)}"
                
                if extended_count > 0:
                    logger.info("BOUNDARY_REFINEMENT: extended %d/%d clips to natural endings", extended_count, len(ranked_clips))
            
            logger.info("BOUNDARY_REFINEMENT: refined=%d/%d", len(refined), len(ranked_clips))
            
            # Normalize episode words for consistent processing
            episode_words_raw = _episode_words_or_empty(episode)
            episode_words = _normalize_episode_words(episode_words_raw)
            
            # Normalize eos_times early to prevent NoneType crashes
            eos_times = _safe_eos(eos_times)
            
            # Universal micro-extension (≤3s tail snap) - safe and cheap
            # This runs BEFORE gates to rescue borderline unfinished clips
            from services.util import extend_to_natural_end
            tail_snap_count = 0
            for clip in ranked_clips:
                try:
                    # Universal micro-tail snap (idempotent: only extends to clean EOS)
                    original_end = clip.get("end", 0.0)
                    extend_to_natural_end(clip, episode_words, max_extension=3.0)
                    if clip.get("end", 0.0) != original_end:
                        tail_snap_count += 1
                except Exception as e:
                    logger.debug("TAIL_SNAP: skipped for %s due to %s", clip.get("id"), e)
            
            if tail_snap_count > 0:
                logger.info("TAIL_SNAP: extended %d/%d clips to natural endings", tail_snap_count, len(ranked_clips))
            
            # EOS confidence scoring and smart extension (post-boundary refinement)
            from services.util import calculate_finish_confidence, finish_threshold_for, extend_to_natural_end
            
            for clip in ranked_clips:
                try:
                    # Calculate EOS confidence for the clip end
                    if episode_words:
                        end_confidence = calculate_finish_confidence(
                            clip.get("text", ""), 
                            episode_words, 
                            clip.get("end", 0.0)
                        )
                        clip["eos_confidence"] = end_confidence
                        
                        # Smart extension to natural end if confidence is low
                        # Build indicators for adaptive thresholding
                        speech_stats = (viral_result or {}).get("speech_stats") or {}
                        indicators = {
                            "conversational_ratio": float(speech_stats.get("conversational_ratio", 0.0)),
                            "interview_format": bool(speech_stats.get("interview_format", False)),
                        }
                        threshold = finish_threshold_for(final_genre, indicators)
                        if end_confidence < threshold:
                            extended_end = extend_to_natural_end(
                                clip.get("end", 0.0),
                                episode_words,
                                max_extension=2.0  # Allow up to 2 seconds extension
                            )
                            if extended_end > clip.get("end", 0.0):
                                old_end = clip.get("end", 0.0)
                                clip["end"] = extended_end
                                clip["extension_delta"] = extended_end - old_end
                                clip["smart_extended"] = True
                                logger.debug("SMART_EXTENSION: extended clip %s from %.2f to %.2f (+%.2fs)", 
                                           clip.get("id", "unknown"), old_end, extended_end, extended_end - old_end)
                    else:
                        clip["eos_confidence"] = 0.5  # Default confidence when no words available
                        clip["smart_extended"] = False
                except Exception as e:
                    logger.warning("EOS_CONFIDENCE: failed to calculate for clip %s: %s", clip.get("id", "unknown"), e)
                    clip["eos_confidence"] = 0.5
                    clip["smart_extended"] = False
            
            # Apply trail padding and nudge to punctuation
            from config.settings import HEAD_PAD_SEC, TRAIL_PAD_SEC, REFINE_SNAP_MAX_NUDGE, REFINE_MIN_TAIL_SILENCE
            episode_words = _episode_words(episode)
            base = getattr(self, "_base_segments", None) or self._base_from_words()
            total_duration = _compute_total_duration(episode, base, ranked_clips)
            
            for clip in ranked_clips:
                # Nudge to punctuation or gap
                end_time = _nudge_to_punct_or_gap(
                    clip["end"], 
                    episode_words, 
                    REFINE_SNAP_MAX_NUDGE, 
                    REFINE_MIN_TAIL_SILENCE
                )
                
                # Apply padding
                start = max(0.0, clip["start"] - HEAD_PAD_SEC)
                end = min(total_duration, end_time + TRAIL_PAD_SEC)
                
                clip["start"] = round(start, 2)
                clip["end"] = round(end, 2)
                clip["duration"] = round(end - start, 2)
            
            # Add metadata to each clip and apply final length clamps
            final_clips = []
            for clip in ranked_clips:
                clip["protected_long"] = clip.get("protected", False)
                # Round and clamp timing
                clip["start"] = round(max(0.0, clip["start"]), 2)
                clip["end"] = round(max(clip["start"], clip["end"]), 2)
                clip["duration"] = round(clip["end"] - clip["start"], 2)
                # Enforce hard length policy
                if clip["duration"] < CLIP_LEN_MIN or clip["duration"] > CLIP_LEN_MAX:
                    continue  # Skip clips outside bounds
                clip["pl_v2"] = clip.get("platform_length_score_v2", 0.0)
                
                # Recompute finished_thought detection after all refinements
                from services.secret_sauce_pkg.features import likely_finished_text
                from services.util import calculate_finish_confidence, finish_threshold_for
                
                # Use both text-based and EOS confidence-based detection
                text_finished = likely_finished_text(clip.get("text", ""))
                eos_confidence = clip.get("eos_confidence", 0.5)
                
                # Calculate finish confidence with EOS proximity boost
                # eos_times should already be normalized by _safe_eos
                try:
                    finish_conf = calculate_finish_confidence(clip, episode_words, eos_times)
                except Exception as e:
                    logger.warning("EOS_CONFIDENCE: failed to calculate for clip %s: %s", clip.get("id"), e)
                    finish_conf = 0.0
                
                # Finish confidence + sparse-finish promotion
                if finish_conf >= 0.75:
                    clip["finished_thought"] = True
                    clip["ft_status"] = "finished"
                else:
                    # Check for sparse-finish (near EOS + decent payoff)
                    from services.util import _nearest_eos_after
                    nearest = _nearest_eos_after(float(clip.get("end", 0.0)), eos_times)
                    payoff = clip.get("features", {}).get("payoff", 0.0)
                    if nearest is not None and 0.0 <= nearest <= 1.2 and payoff >= 0.45:
                        clip["ft_status"] = "sparse_finished"
                        clip["finished_thought"] = False
                    else:
                        # Build indicators for adaptive thresholding
                        speech_stats = (viral_result or {}).get("speech_stats") or {}
                        indicators = {
                            "conversational_ratio": float(speech_stats.get("conversational_ratio", 0.0)),
                            "interview_format": bool(speech_stats.get("interview_format", False)),
                        }
                        threshold = finish_threshold_for(final_genre, indicators)
                        eos_finished = 1 if eos_confidence >= threshold else 0
                        
                        # Combine both signals (either can indicate finished)
                        clip["finished_thought"] = max(text_finished, eos_finished)
                        clip["text_finished"] = text_finished
                        clip["eos_finished"] = eos_finished
                        
                        # Set ft_status based on the combined result
                        if clip.get("finished_thought", 0) == 1:
                            clip["ft_status"] = "finished"
                        else:
                            clip["ft_status"] = "unresolved"
                
                final_clips.append(clip)
            
            # Apply length bucket cap safely with fallback
            finals_before_limits = list(final_clips)  # shallow copy before caps
            
            try:
                final_clips = apply_length_bucket_cap(final_clips, length_agnostic=length_agnostic)
            except Exception as e:
                logger.exception("LIMITS_PIPE_ERROR: %s", e)
                final_clips = finals_before_limits
            
            # Hard guarantee: never end up empty due to post-pick logic
            if not final_clips and finals_before_limits:
                logger.warning("LIMITS_PIPE_EMPTY: restoring %d finals", len(finals_before_limits))
                final_clips = finals_before_limits
            
            # Backend compatibility shim: ensure canonical score fields for frontend
            for clip in final_clips:
                # Canonicalize virality field (0-1 range)
                v = clip.get("display_score", clip.get("final_score", clip.get("score", 0.0)))
                try:
                    v = float(v)
                except Exception:
                    v = 0.0
                clip["virality"] = v  # 0–1 canonical
                clip["virality_pct"] = int(round(v * 100))  # convenience for UI
                
                # Optional: expose other scores similarly for UI convenience
                for k in ("hook_score", "arousal_score", "payoff_score", "info_density", "q_list_score"):
                    if k in clip:
                        try:
                            clip[f"{k}_pct"] = int(round(float(clip[k]) * 100))
                        except Exception:
                            clip[f"{k}_pct"] = 0
            
            # Log final summary from the actual finals we're saving
            try:
                _log_final_summary(final_clips, logger)
            except Exception as e:
                logger.warning("FINAL_SUMMARY_LOG_ERROR: %s", e)
            
            # Log enhanced run summary with platform context and payoff rescue info
            try:
                durs = [round(float(c.get("end", 0)) - float(c.get("start", 0)), 1) for c in final_clips]
                durs_str = "[" + ",".join(map(str, durs)) + "]"
                mode = "platform-aware" if pl_v2_weight > 0 else "neutral"
                
                # Update mode if length-agnostic
                if length_agnostic:
                    mode = "length-agnostic"
                
                # Calculate additional summary metrics
                strict_finished = len([c for c in final_clips if c.get("ft_status") == "finished"])
                balanced_finished = len([c for c in final_clips if c.get("ft_status") in ["finished", "sparse_finished"]])
                # Safe telemetry calculation with fallbacks
                avg_virality = sum(c.get("virality_pct", c.get("display_score", 50)) for c in final_clips) / max(len(final_clips), 1)
                avg_platform_fit = sum(c.get("platform_fit", c.get("platform_fit_pct", 0)) for c in final_clips) / max(len(final_clips), 1)
            except Exception as e:
                logger.warning("TELEMETRY_CALC_ERROR: %s", e)
                # Safe fallbacks
                durs_str = "[]"
                mode = "neutral"
                strict_finished = balanced_finished = 0
                avg_virality = avg_platform_fit = 0.0
            
            # Count rescued clips
            rescued_count = sum(1 for c in final_clips if c.get("rescue_reason"))
            
            # Get telemetry values safely
            try:
                wc, ec, dens = _telemetry_density(episode_words, eos_times)
            except Exception:
                wc, ec, dens = 0, 0, 0.0
            
            logger.info(
                "RUN_SUMMARY: seeds=%d strict=%d balanced=%d finals=%d durs=%s eos=%d dens=%.3f pl_v2_w=%.2f protect=%s mode=%s fallback=%s avg_virality=%.1f avg_platform_fit=%.2f rescued=%d",
                len(ranked_clips), strict_finished, balanced_finished, len(final_clips), durs_str, ec, dens, 
                pl_v2_weight, platform_protect, mode, episode_fallback_mode, avg_virality, avg_platform_fit, rescued_count
            )
            
            # Final sanity check before save
            assert len(final_clips) > 0, "Should have at least one clip"
            
            # Calculate longest clip duration
            longest_dur = max((c.get("end", 0) - c.get("start", 0)) for c in final_clips) if final_clips else 0.0
            
            # FT summary already logged above via _log_final_summary
            
            # Telemetry already calculated above
            logger.info("TELEMETRY: word_count=%d, eos_count=%d, eos_density=%.3f", wc, ec, dens)
            logger.info(f"TELEMETRY: fallback_mode={episode_fallback_mode} (authoritative)")
            
            # Candidate-level FT summary (post-refinement, post-extension)
            cand_total = len(final_clips)
            cand_finished = sum(1 for c in final_clips if c.get("ft_status") == "finished")
            cand_sparse = sum(1 for c in final_clips if c.get("ft_status") == "sparse_finished")
            cand_ratio_strict = (cand_finished / cand_total) if cand_total else 0.0
            cand_ratio_sparse_ok = ((cand_finished + (cand_sparse if episode_fallback_mode else 0)) / cand_total) if cand_total else 0.0
            
            logger.info("FT_SUMMARY_CANDIDATES: total=%d finished=%d sparse=%d ratio_strict=%.2f ratio_sparse_ok=%.2f",
                        cand_total, cand_finished, cand_sparse, cand_ratio_strict, cand_ratio_sparse_ok)
            logger.info(f"POST_SAFETY: kept={len(final_clips)} ids={[c.get('id', 'unknown') for c in final_clips]}")
            
            # Candidate-based warning (use env tunable threshold; defaults are sensible)
            warn_min_strict = float(os.getenv("FT_WARN_MIN_CAND_RATIO_STRICT", "0.60"))
            warn_min_fallback = float(os.getenv("FT_WARN_MIN_CAND_RATIO_FALLBACK", "0.70"))
            if not episode_fallback_mode and cand_ratio_strict < warn_min_strict:
                logger.warning("FINISH_RATIO_LOW_CAND: %.2f < %.2f (strict)", cand_ratio_strict, warn_min_strict)
            elif episode_fallback_mode and cand_ratio_sparse_ok < warn_min_fallback:
                logger.warning("FINISH_RATIO_LOW_CAND: %.2f < %.2f (fallback)", cand_ratio_sparse_ok, warn_min_fallback)
            
            # Upstream/auth summary removed to avoid telemetry confusion
            # Use _log_final_summary() for authoritative reporting
            
            # Build metadata for return
            meta = {
                "reason": "ok" if ranked_clips else "no_candidates",
                "episode_id": episode_id,
                "platform": platform,
                "genre": final_genre,
                "fallback_mode": episode_fallback_mode,
                "eos_density": eos_density,
                "candidate_count": len(final_clips),
                "default_clip_id": default_clip_id,
                # Platform flags for debugging
                "platform_neutral": True,
                "pl_v2_weight": pl_v2_weight,
                "platform_protect": platform_protect,
                "length_agnostic": length_agnostic,
            }
            
            # Calculate finals_finished for consistent return type
            finals_finished = [c for c in final_clips if c.get("finished_thought", False)]
            return final_clips, finals_finished
            
        except Exception as e:
            logger.exception("CANDIDATE_PIPELINE_FAIL: returning finals as-is")
            meta = {
                "reason": "exception",
                "episode_id": episode_id,
                "error": str(e)[:200],
                "platform": platform,
                "genre": genre
            }
            # Return minimally formatted finals instead of zero.
            # This guarantees: if we've already computed finals, we get clips back
            if final_clips:
                logger.info(f"SALVAGE_GUARD: returning {len(final_clips)} clips despite error")
                rescued_finished = [c for c in final_clips if c.get("finished_thought", False)]
                return [self._minimal_candidate_dict(c) for c in final_clips], rescued_finished
            else:
                logger.warning("SALVAGE_GUARD: no finals to salvage, returning empty")
                return [], []

async def score_episode(episode, segments):
    """
    Score an episode and return ranked clips.
    This is a standalone function for compatibility with the existing main.py usage.
    """
    try:
        # Create a ClipScoreService instance
        from services.episode_service import EpisodeService
        episode_service = EpisodeService()
        clip_service = ClipScoreService(episode_service)
        
        # Get audio path for the episode
        audio_path = episode.audio_path
        if not audio_path:
            logger.error(f"No audio path found for episode {episode.id}")
            return []
        
        # Convert segments to the format expected by the service
        transcript_segments = []
        for seg in segments:
            transcript_segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', ''),
                'confidence': seg.get('confidence', 0.0)
            })
        
        # Get candidates using the service (await the async call)
        candidates, default_clip_id = await clip_service.get_candidates(
            episode_id=episode.id,
            platform="tiktok_reels",
            genre="general"
        )
        
        # Convert candidates to the expected format
        scored_clips = []
        for candidate in candidates:
            scored_clips.append({
                'id': candidate.get('id', ''),
                'start': candidate.get('start', 0),
                'end': candidate.get('end', 0),
                'text': candidate.get('text', ''),
                'score': candidate.get('score', 0),
                'confidence': candidate.get('confidence', ''),
                'genre': candidate.get('genre', 'general'),
                'platform': candidate.get('platform', 'tiktok'),
                'features': candidate.get('features', {}),
                'meta': candidate.get('meta', {})
            })
        
        return scored_clips
        
    except Exception as e:
        logger.error(f"Failed to score episode {episode.id}: {e}", exc_info=True)
        return []
