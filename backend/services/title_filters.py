"""
Title filtering & repair helpers.

Usage:
- from services.title_filters import (
      clean_for_keywords, fix_or_reject_title, dedup_titles,
      get_anchor_set
  )
"""

from __future__ import annotations
import os, re
from typing import Iterable, Tuple, Optional, List, Set

# ----------------------------
# Env helpers
# ----------------------------

def _env_csv(name: str) -> Set[str]:
    val = os.getenv(name, "")
    if not val:
        return set()
    return {t.strip().lower() for t in val.split(",") if t.strip()}

DESTUTTER_PASSES = max(1, int(os.getenv("TITLE_DESTUTTER_PASSES", "3")))
FILTER_STRICT = os.getenv("TITLE_FILTER_STRICT", "1") not in ("0", "false", "False")

# ----------------------------
# Core lexical sets
# ----------------------------

DETERMINERS   = {"a","an","the","this","that","these","those","my","your","our","their","his","her","its"}
QUESTIONERS   = {"why","what","how","when"}
PRONOUNS      = {"i","i'm","you","you're","we","we're","they","they're","he","she","it"}
FUNC_TAILS    = {"of","to","for","with","on","in","at","by","about"}

# Compact, high-signal anchors (extend via env TITLE_ANCHORS_EXTRA="benchmark,latency,...")
BASE_ANCHORS = {
    # leadership/ops
    "feedback","leadership","team","manager","culture","accountability",
    "meeting","1:1","model","framework","strategy","process","system",
    # biz/performance
    "revenue","growth","market","audience","platform","search","algorithm",
    "pipeline","retention","conversion","engagement","quality","performance",
    # neutral ops nouns
    "project","plan","method","approach","workflow","metric","kpi",
    "result","outcome","impact","risk","opportunity","problem","solution",
    "initiative","program","roadmap","architecture","design","deployment"
}
ANCHOR_OVERRIDES = _env_csv("TITLE_ANCHORS_EXTRA")
GLOBAL_ANCHORS: Set[str] = set(a.lower() for a in (BASE_ANCHORS | ANCHOR_OVERRIDES))

def get_anchor_set() -> Set[str]:
    """Expose effective anchors (lowercased)."""
    return GLOBAL_ANCHORS

# ----------------------------
# Regexes
# ----------------------------

# single-token repeats: "really really good" -> "really good"
DEDUP_TOKEN_RE   = re.compile(r"\b([A-Za-z]+)\b(?:\s+\1\b)+", re.IGNORECASE)
# ABA pattern: "good luck good" -> "good luck"
DEDUP_ABA_RE     = re.compile(r"\b([A-Za-z]+)\s+([A-Za-z]+)\s+\1\b", re.IGNORECASE)
# bigram repeats: "of the of the" -> "of the"
DEDUP_BIGRAM_RE  = re.compile(r"\b([A-Za-z]+)\s+([A-Za-z]+)(?:\s+\1\s+\2\b)+", re.IGNORECASE)

NON_WORD_RE      = re.compile(r"[^A-Za-z0-9'\-]+")
MULTI_SPACE_RE   = re.compile(r"\s+")
MULTI_DASH_RE    = re.compile(r"\s*[-–—]\s*")
TRAILING_PUNC_RE = re.compile(r"[–—\-:,;]+$")

# Extra "bad" fragments you may want to ban outright (env-extendable).
BAD_FRAGMENTS = {
    "all all", "good luck good", "first question i'm", "you're giving positive",
    "frame up feedback",
}
BAD_FRAGMENTS |= _env_csv("TITLE_BAD_FRAGMENTS_EXTRA")

# Grammar fixes for better title quality
GERUND_FIX = {
    "give": "giving",
    "receive": "receiving", 
    "build": "building",
    "create": "creating",
}

# Filler prefix removal
FILLER_PREFIX_RE = re.compile(
    r"^(?:so|well|yeah|um|uh|like|quote|listen|look|okay|alright|you know)\b[^\w]*",
    re.IGNORECASE
)

# ----------------------------
# Utilities
# ----------------------------

def _destutter(text: str, passes: int = DESTUTTER_PASSES) -> str:
    t = DEDUP_TOKEN_RE.sub(r"\1", text)
    for _ in range(max(1, passes)):
        new = DEDUP_BIGRAM_RE.sub(r"\1 \2", t)
        new = DEDUP_ABA_RE.sub(r"\1 \2", new)
        if new == t:
            break
        t = new
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t

def _normalize_spaces_dashes(s: str) -> str:
    # normalize em/en dashes to " — " with single spaces
    parts = MULTI_DASH_RE.split(s)
    return " — ".join(p.strip() for p in parts if p.strip())

def _lower_tokens(s: str) -> List[str]:
    s = s.strip()
    s = re.sub(TRAILING_PUNC_RE, "", s)
    s = MULTI_SPACE_RE.sub(" ", s)
    return s.lower().split()

def _strip_dets_edges(tokens: List[str]) -> List[str]:
    while tokens and tokens[0] in DETERMINERS: tokens.pop(0)
    while tokens and (tokens[-1] in DETERMINERS or tokens[-1] in FUNC_TAILS): tokens.pop()
    return tokens

def _has_anchor_tail(tokens: List[str]) -> bool:
    return bool(tokens) and (tokens[-1] in GLOBAL_ANCHORS)

def _hard_ban(tokens: List[str]) -> bool:
    if not tokens: return True
    head, tail = tokens[0], tokens[-1]
    return (
        head in PRONOUNS or head in QUESTIONERS or
        tail in PRONOUNS or tail in QUESTIONERS or
        tail in FUNC_TAILS
    )

def _soft_penalty(tokens: List[str], anchored_tail: bool) -> float:
    # penalize if pronoun/question appears inside and tail isn't anchored
    inner = tokens[1:-1] if len(tokens) > 2 else []
    bad_inside = any(t in PRONOUNS or t in QUESTIONERS for t in inner)
    return -0.5 if (bad_inside and not anchored_tail) else 0.0

def _looks_nonsense(s: str) -> Optional[str]:
    raw = s.strip()
    if not raw:
        return "empty"
    low = " ".join(_lower_tokens(raw))
    if any(f in low for f in BAD_FRAGMENTS):
        return "bad_fragment"
    # very short or single stopword
    if len(low) < 3 or len(low.split()) == 1 and low in (DETERMINERS | PRONOUNS | QUESTIONERS | FUNC_TAILS):
        return "too_short"
    # repeated same word ≥3 times
    toks = low.split()
    if len(toks) >= 3 and len(set(toks)) == 1:
        return "repeated_single"
    # dangling dash or punctuation
    if raw.endswith(("-", "–", "—", ":", ";", ",")):
        return "dangling_punct"
    return None

def _grammar_touchup(tokens: List[str]) -> List[str]:
    """Apply grammar fixes to improve title quality"""
    # 1) give/receive feedback -> giving/receiving feedback
    if len(tokens) >= 2 and tokens[1] == "feedback" and tokens[0] in GERUND_FIX:
        tokens[0] = GERUND_FIX[tokens[0]]

    # 2) drop odd "frame up" bigram
    for i in range(len(tokens) - 1):
        if tokens[i] == "frame" and tokens[i+1] == "up":
            tokens.pop(i+1)
            break
    return tokens

def strip_filler_prefix(s: str) -> str:
    """Remove filler prefixes like 'So, yeah...', 'Quote...', etc."""
    return FILLER_PREFIX_RE.sub("", s).strip()

def _clean_phrase_edges(s: str) -> str:
    toks = _lower_tokens(_normalize_spaces_dashes(_destutter(s)))
    toks = _strip_dets_edges(toks)
    toks = _grammar_touchup(toks)
    return " ".join(toks)

# ----------------------------
# Public API
# ----------------------------

def clean_for_keywords(text: str) -> str:
    """
    Normalize clip text for cache keys / keyword extraction.
    Lowercase, remove non-word except quotes/dashes, destutter, collapse spaces.
    """
    t = _destutter(text)
    t = _normalize_spaces_dashes(t)
    t = NON_WORD_RE.sub(" ", t.lower())
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t

def score_title_candidate(title: str, clip_text: Optional[str] = None) -> Tuple[float, str]:
    """
    Returns (score, normalized_title). Higher is better.
    Will clean determiners at edges and destutter.
    """
    # 0) quick nonsense check before heavy ops
    ns = _looks_nonsense(title)
    if ns:
        return (-10.0, "")  # hard reject with strong negative

    # 1) clean
    cleaned_edges = _clean_phrase_edges(title)
    if not cleaned_edges:
        return (-10.0, "")
    tokens = cleaned_edges.split()

    # 2) hard ban on shards
    if _hard_ban(tokens):
        return (-5.0, "")

    # 3) base score by length prior (prefer 2–6 tokens)
    n = len(tokens)
    length_prior = {1: 0.0, 2: 1.0, 3: 1.2, 4: 1.15, 5: 1.1, 6: 1.05}.get(n, 0.8 if n < 2 else 0.9)

    # 4) anchor bonus / soft penalties
    anchor_tail = _has_anchor_tail(tokens)
    bonus = 0.5 if anchor_tail else 0.0
    penalty = _soft_penalty(tokens, anchor_tail)

    score = length_prior + bonus + penalty
    norm = " ".join(tokens)

    # 5) final sanity: avoid titles ending with boring modals/aux
    if tokens[-1] in {"be","do","have","get","make","go"}:
        score -= 0.2

    return (score, norm)

def fix_or_reject_title(title: str, clip_text: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Try to repair a generated title; if unrecoverable, reject.

    Returns (ok, final_title, reason)
      - ok=True: final_title is usable (normalized case; still lowercase words; let caller apply casing/titlecase)
      - ok=False: reason in {"nonsense","hard_ban"} and title is ""
    """
    # quick destutter + dash normalization (visible cleanup)
    t0 = _normalize_spaces_dashes(_destutter(title))
    if (r := _looks_nonsense(t0)):
        return (False, "", "nonsense:" + r)

    score, norm = score_title_candidate(t0, clip_text)
    if score < (0.8 if FILTER_STRICT else 0.5) or not norm:
        return (False, "", "hard_ban")
    return (True, norm, "ok")

def dedup_titles(titles: Iterable[str]) -> List[str]:
    """
    Deduplicate by normalized key (destuttered + lowercase + stripped punctuation).
    Preserves order.
    """
    seen = set()
    out: List[str] = []
    for t in titles:
        key = NON_WORD_RE.sub(" ", _destutter(t).lower())
        key = MULTI_SPACE_RE.sub(" ", key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
