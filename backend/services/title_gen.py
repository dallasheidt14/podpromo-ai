# services/title_gen.py
from __future__ import annotations
from typing import Dict, List, Tuple
import re
import math

MAX_LEN = 72
CLICKBAIT = re.compile(r"\b(insane|crazy|shocking|unbelievable|magical|instant)\b", re.I)
STRONG_VERB = re.compile(r"\b(win|save|avoid|learn|unlock|beat|grow|double|prove|fix|crush|build|master|improve|train)\b", re.I)
NUM_TOKEN   = re.compile(r"\b\d+\b")
JUNK_PREFIX = re.compile(r"^(so|well|look|listen|okay|you know)[, ]+", re.I)

def _clean(s: str) -> str:
    s = (s or "").strip()
    s = JUNK_PREFIX.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _first_clause(text: str) -> str:
    if not text: return ""
    # split by punctuation; take the most informative sentence
    pieces = re.split(r"[.!?]", text)
    pieces = [p.strip() for p in pieces if len(p.split()) >= 6]
    base = pieces[0] if pieces else text.strip()
    # take first clause before long ramble
    clause = re.split(r"[-–—;:]", base)[0]
    return clause.strip()

def summarize_clip(text: str) -> str:
    return _first_clause(text)

def _patterns(summary: str) -> List[str]:
    s = _clean(summary)
    opts = [
        f"Train to Win: {s}",
        f"Stop Wasting Games—Train Like This: {s}",
        f"The Overlooked Rule That Accelerates Development: {s}",
        f"How to Outperform the Competition: {s}",
        f"{s} (Coach Tip)",
    ]
    # trim
    out = []
    for c in opts:
        c = c.strip(" -–—")
        if len(c) > MAX_LEN:
            c = c[:MAX_LEN-1].rstrip() + "…"
        out.append(c)
    # de-dupe while preserving order
    seen = set(); dedup = []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k); dedup.append(t)
    return dedup[:6]

def _score_candidate(title: str, feats: Dict) -> float:
    """Heuristic scorer: higher is better (0..1)."""
    # 1) length sweet spot (42–66 chars)
    L = len(title)
    len_score = math.exp(-((L - 54) ** 2) / (2 * (12 ** 2)))  # Gaussian peak ~54

    # 2) verb strength & numbers
    verb_bonus = 0.08 if STRONG_VERB.search(title) else 0.0
    num_bonus  = 0.06 if NUM_TOKEN.search(title) else 0.0

    # 3) match to clip features: payoff > hook > arousal
    payoff = float(feats.get("payoff_score", 0.0))
    hook   = float(feats.get("hook_score", 0.0))
    ar     = float(feats.get("arousal_score", feats.get("arousal", 0.0)))
    match_bonus = 0.20 * payoff + 0.12 * hook + 0.06 * ar

    # 4) penalties
    clickbait_pen = 0.15 if CLICKBAIT.search(title) else 0.0
    all_caps_pen  = 0.10 if (len(title) >= 6 and title.upper() == title) else 0.0

    score = len_score + verb_bonus + num_bonus + match_bonus - clickbait_pen - all_caps_pen
    return max(0.0, min(1.0, score))

def best_title_for_clip(clip: Dict) -> Tuple[str, List[Tuple[str,float]]]:
    """Return (best_title, [(cand,score), ...]) for debug."""
    text  = clip.get("text", "") or ""
    feats = clip.get("features", {}) or {}
    summary = summarize_clip(text)
    cands = _patterns(summary)
    ranked = sorted(((c, _score_candidate(c, feats)) for c in cands), key=lambda x: x[1], reverse=True)
    best = ranked[0][0] if ranked else (summary[:MAX_LEN] or "Untitled")
    return best, ranked
