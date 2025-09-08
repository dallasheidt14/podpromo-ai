# services/title_service.py
from __future__ import annotations
import re, math
from collections import Counter
from typing import Dict, List, Optional

STOP = {
    "a","an","and","the","to","of","in","on","for","with","that","this","is","are",
    "as","at","by","it","from","be","or","if","you","your","i","we","our","they",
    "their","them","me","my","so","but","not","have","has","had","do","does","did",
    "will","would","should","can","could","about","into","over","than","then"
}

POWER = {
    "mistake","truth","secret","strategy","simple","surprising","little-known",
    "science","why","how","stop","avoid","never","always","faster","smarter",
    "hack","blueprint","playbook","formula","framework"
}

HOOK_PATTERNS = [
    r"^how to\b",
    r"^why\b",
    r"^stop\b",
    r"^never\b",
    r"\bthe truth\b",
    r"\bdo this\b",
    r"\bavoid\b",
    r"\bsecrets?\b",
]

def _sentences(text: str) -> List[str]:
    # light segmentation; keeps sentences reasonably short
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]

def _keywords(text: str, topn: int = 6) -> List[str]:
    # simple frequency keywords + bigrams
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z']+", text)]
    toks = [t for t in tokens if t not in STOP and len(t) > 2]
    counts = Counter(toks)
    # add bigrams
    bigrams = [" ".join([toks[i], toks[i+1]]) for i in range(len(toks)-1)]
    for bg in bigrams:
        if all(w not in STOP for w in bg.split()):
            counts[bg] += 1
    return [w for w,_ in counts.most_common(topn)]

def _titlecase(s: str) -> str:
    s = s.strip(" -–—.:;")
    words = s.split()
    out = []
    for i,w in enumerate(words):
        lw = w.lower()
        if i not in (0, len(words)-1) and lw in STOP:
            out.append(lw)
        else:
            out.append(w[0:1].upper() + w[1:])
    return " ".join(out)

def _len_score(s: str, max_len: int) -> float:
    # reward being close to ~90% of max_len
    L = len(s)
    target = max_len * 0.9
    return math.exp(-((L - target) ** 2) / (2 * (0.25 * max_len) ** 2))

def _pattern_bonus(s: str) -> float:
    s_low = s.lower()
    return 1.2 if any(re.search(p, s_low) for p in HOOK_PATTERNS) else 1.0

def _compress_sentence(s: str, max_len: int) -> str:
    # trim filler & keep strong words
    kws = _keywords(s, topn=8)
    if kws:
        base = " ".join(_titlecase(k) for k in kws[:5])
        if 20 < len(base) <= max_len:
            return base
    # fallback: remove parentheticals & shorten
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,!?:;")
    return s[:max_len]

def _templates(topic: List[str], best: str) -> List[str]:
    # topic are keyword phrases already lowercased
    t = [k.title() for k in topic]
    core = " ".join(t[:3]).strip() or best
    return [
        f"How to {core}",
        f"Stop {t[0]} — Do This Instead" if t else f"Do This Instead",
        f"The Truth About {core}",
        f"{core}: What Everyone Gets Wrong",
        f"{core} in 60 Seconds",
        f"{core} — The Simple Playbook",
        best,  # include the compressed best sentence as-is
    ]

def generate_titles(
    transcript: str,
    features: Optional[Dict[str, float]] = None,
    *,
    max_len: int = 80
) -> List[str]:
    """
    Cheap, deterministic title generator.
    Returns 4–8 distinct, social-ready titles <= max_len.
    """
    features = features or {}
    sents = _sentences(transcript)
    if not sents:
        return []

    # 1) score sentences quickly
    scored = []
    for s in sents:
        l = s.lower()
        score = 0.0
        score += 1.0 if "?" in s else 0.2  # questions tend to hook
        score += 0.5 if any(p in l for p in POWER) else 0.0
        score += 0.4 if re.search(r"\b\d+", s) else 0.0
        score += 0.3 if re.search(r"\byou|your|we|our\b", l) else 0.0
        score += 0.4 if re.search(r"\b(stop|avoid|never|don'?t)\b", l) else 0.0
        score *= _pattern_bonus(s)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_sentence = scored[0][1]
    best_sentence = _compress_sentence(best_sentence, max_len)

    # 2) extract topic & build templates
    topic = _keywords(" ".join(sents), topn=8)
    candidates = _templates(topic, best_sentence)

    # 3) clean & enforce length; remove substrings of transcript head
    head = re.sub(r"\s+", " ", transcript.strip())[:90].lower()
    unique: List[str] = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" .,!?:;")
        c = _titlecase(c)
        if len(c) > max_len:
            c = _compress_sentence(c, max_len)
            c = _titlecase(c)
        if len(c) < 12:
            continue
        # avoid "first 80 chars of transcript"
        if c.lower() in head or head.startswith(c.lower()):
            continue
        if c not in unique:
            unique.append(c)

    # 4) soft re-rank by length fitness
    unique.sort(key=lambda s: _len_score(s, max_len), reverse=True)
    return unique[:6]
