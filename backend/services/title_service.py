# backend/services/title_service.py
# Improved title generation with normalization, relaxed bans, dedup, and fallbacks

from __future__ import annotations
import re, math, itertools, hashlib, time
from collections import Counter
from typing import List, Dict, Iterable, Tuple, Optional, Set, Any
import logging
from datetime import datetime, timezone
from functools import lru_cache
from config.settings import PLAT_LIMITS, TITLE_ENGINE_V2

logger = logging.getLogger("titles_service")

def _hash_text(s: str) -> str:
    """Generate a short hash for text content"""
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def _cache_key(clip_id: str, platform: str, text: str, kw: list[str]) -> tuple:
    """Generate cache key for title pack generation"""
    return (clip_id or "", platform or "shorts", _hash_text(text or ""), tuple(sorted(kw or [])))

@lru_cache(maxsize=512)
def _cached_title_pack(key: tuple, *, _gen_fn):
    """Cached title pack generation - key is produced by _cache_key"""
    return _gen_fn()

def _normalize_to_list_of_str(variants) -> list[str]:
    """Normalize variants to list of strings, handling both str and dict formats"""
    out = []
    if not variants:
        return out
    for v in variants:
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, dict):
            # accept common keys
            s = v.get("title") or v.get("text") or v.get("name")
            if s:
                s = str(s).strip()
                if s:
                    out.append(s)
        else:
            s = str(v).strip()
            if s:
                out.append(s)
    # de-dup case-insensitively
    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def generate_titles_v2(text: str, platform: str) -> list[str]:
    """V2 title engine wrapper that always returns list[str]"""
    try:
        from services.title_engine_v2 import generate_titles_v2 as _generate_titles_v2_raw
        raw = _generate_titles_v2_raw(text=text, platform=platform)
    except Exception as e:
        logger.warning("TITLE_ENGINE_V2: failed %s, falling back to V1", e)
        return []
    titles = _normalize_to_list_of_str(raw)
    # hard caps (80 chars typical)
    cap = 80
    trimmed = [t[:cap].rstrip() for t in titles if t]
    return trimmed[:6]

# Platform-specific budgets
PLAT_BUDGETS = {
    "shorts":  {"overlay": 32, "short": 32, "mid": 60, "long": 100},
    "tiktok":  {"overlay": 32, "short": 32, "mid": 60, "long": 80},
    "reels":   {"overlay": 32, "short": 32, "mid": 60, "long": 90},
    "default": {"overlay": 32, "short": 32, "mid": 60, "long": 90},
}

# Filler/transition opener patterns to filter out
FILLER_PREFIXES = (
    "transitioning to", "first,", "second,", "third,", "in conclusion", "welcome back",
    "in this episode", "today we'll", "today we will", "we'll explore", "we will explore",
    "let's explore", "join us", "come along", "in this video", "in this clip",
    "everything you need to know", "ultimate guide", "at the end of the day",
    "an exploration of", "in an age where", "to begin,", "this episode will", 
    "today we discuss", "let's dive into", "today we're going to", 
    "in today's episode", "this week we", "in this segment"
)

# Expanded banlist for better quality
_BANNED_OPENERS = {
    "transitioning to", "first,", "second,", "third,", "in conclusion", "welcome back",
    "in this episode", "today we'll", "today we will", "we'll explore", "we will explore",
    "let's explore", "join us", "come along", "in this video", "in this clip",
    "everything you need to know", "ultimate guide", "at the end of the day",
    "an exploration of", "in an age where", "to begin,", "this episode will", 
    "today we discuss", "let's dive into", "today we're going to", 
    "in today's episode", "this week we", "in this segment"
}

def _is_filler_open(title: str) -> bool:
    """Check if title starts with filler/transition phrases"""
    t = (title or "").strip().lower()
    return any(t.startswith(b) for b in _BANNED_OPENERS)

def _force_keyword(title: str, keywords: list[str]) -> str:
    """Inject the strongest keyword into title if not already present"""
    if not keywords:
        return title
    
    # Check if any keyword is already in title
    title_lower = title.lower()
    if any(k.lower() in title_lower for k in keywords):
        return title
    
    # Inject the strongest keyword near the front
    k = keywords[0]
    # Try to insert after first word
    words = title.split()
    if len(words) > 1:
        words.insert(1, k)
        return " ".join(words)
    else:
        return f"{title} {k}"

def _classify_style(title: str) -> str:
    """Classify title style for better organization and A/B testing"""
    t = title.strip()
    low = t.lower()

    if re.match(r"^\d+\s", t) or re.search(r"\b(top|ways|tips|mistakes|rules)\b", low):
        return "list"
    if re.search(r"\b(my|our)\b\s+(take|mistake|story|lesson)", low):
        return "personal"
    if re.search(r"\b(how to|how we|how i)\b", low):
        return "how_to"
    if low.startswith(("why ", "what ", "when ", "where ", "how ")):
        return "question"
    if re.search(r"\b(vs|versus)\b", low) and not re.search(r"\b(myth|truth)\b", low):
        return "x_vs_y"
    if re.search(r"\b(myth|truth)\b", low):
        return "contrarian"
    return "hook_short" if len(t) <= 32 else "general"

def _title_case(s: str) -> str:
    """Minimal title case: capitalize major words, preserve ALLCAPS acronyms"""
    SMALL = {"a","an","and","or","the","to","of","in","on","for","with","at","by","from","is","are","was","were","be","been","being"}
    words = s.split()
    out = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) > 1:
            out.append(w)  # keep acronyms like AI, GPU
        else:
            lw = w.lower()
            if i > 0 and i < len(words)-1 and lw in SMALL:
                out.append(lw)
            else:
                out.append(lw.capitalize())
    return " ".join(out)

def _polish_title(title: str, style: str) -> str:
    """Polish title with proper punctuation, case, and formatting"""
    t = (title or "").strip()
    # Remove trailing ellipses/multiple punctuation
    t = re.sub(r"[\.…]+$", "", t)
    t = re.sub(r"\s+([:!?])", r"\1", t)
    # Allow at most one colon
    parts = t.split(":")
    if len(parts) > 2:
        t = parts[0] + ": " + " ".join(p.strip() for p in parts[1:])
    # Case strategy
    if len(t) <= 32 or style in {"hook_short","question"}:
        # sentence case for very short hooks
        t = t[0:1].upper() + t[1:]
    else:
        t = _title_case(t)
    return t.strip()

def _hashtags_from_keywords(keywords: list[str], limit_total_chars: int = 45, max_tags: int = 3) -> list[str]:
    """Generate hashtags from keywords with character budget"""
    tags = []
    total = 0
    for k in keywords:
        # Clean the keyword and create hashtag
        clean_k = re.sub(r"[^A-Za-z0-9]+", "", k)
        if len(clean_k) < 2:  # Skip very short keywords
            continue
        
        # Preserve acronyms (all caps) or title case
        if k.isupper() and len(k) > 1:
            tag = "#" + clean_k
        else:
            tag = "#" + clean_k.title()
        
        # Check if we can add this tag
        if len(tags) < max_tags:
            new_total = total + len(tag) + (1 if total else 0)
            if new_total <= limit_total_chars:
                tags.append(tag)
                total = new_total
    return tags

def _title_engine_v2(text: str, want: int = 6, seed_sentence: str | None = None, payoff_sentence: str | None = None) -> list[str]:
    """Title engine v2 facade with heuristic fallback"""
    try:
        # Try to use existing v2 engine
        return generate_titles_v2(text, "default")
    except Exception as e:
        logger.warning("TITLE_ENGINE_V2: failed %s, using heuristic fallback", e)
        return _heuristic_title_generator(text, want)

def _heuristic_title_generator(text: str, want: int = 6) -> list[str]:
    """Simple heuristic title generator as fallback"""
    if not text or not text.strip():
        return ["Untitled Content"]
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ["Untitled Content"]
    
    titles = []
    
    # Use first sentence as base
    first_sentence = sentences[0]
    if len(first_sentence) > 10:
        # Try different angles
        titles.append(first_sentence)
        
        # Add "How to" version
        if not first_sentence.lower().startswith(("how to", "how do", "what is", "why")):
            titles.append(f"How to {first_sentence.lower()}")
        
        # Add "Why" version
        if not first_sentence.lower().startswith(("why", "what", "how")):
            titles.append(f"Why {first_sentence.lower()}")
    
    # Use other sentences if we need more
    for sentence in sentences[1:]:
        if len(sentence) > 10 and len(titles) < want:
            titles.append(sentence)
    
    # Fallback if we still don't have enough
    while len(titles) < want:
        titles.append(f"Key Insight #{len(titles) + 1}")
    
    return titles[:want]

def _generate_title_pack_uncached(clip_id: str, platform: str, text: str, episode_text: str | None = None, keywords: list[str] = None, seed_sentence: str | None = None, payoff_sentence: str | None = None) -> dict:
    """
    Generate a complete title pack with labeled variants, overlay, and metadata.
    Returns dict with variants, overlay, engine, generated_at, meta fields.
    """
    from .keyword_extraction import extract_salient_keywords
    
    # Extract keywords if not provided
    if keywords is None:
        keywords = extract_salient_keywords(text, limit=5) or extract_salient_keywords(episode_text or "", limit=5) or []
    
    budgets = PLAT_BUDGETS.get(platform, PLAT_BUDGETS["default"])
    
    # Generate raw titles using v2 engine
    raw_titles = _title_engine_v2(text, want=6, seed_sentence=seed_sentence, payoff_sentence=payoff_sentence)
    
    # Post-process: filter filler, normalize, inject keywords, classify style, polish, enforce budgets
    def norm_one(s: str, limit: int) -> str:
        s = s.strip().rstrip(". ").replace("..", ".")
        if _is_filler_open(s):
            return ""  # Skip filler titles
        s = _force_keyword(s, keywords)
        # Enforce character limit AFTER keyword injection
        s = s[:limit]
        return s
    
    overlay = ""
    variants = []
    seen = set()
    
    for s in raw_titles:
        overlay_c = norm_one(s, budgets["overlay"])
        short_c   = norm_one(s, budgets["short"])
        mid_c     = norm_one(s, budgets["mid"])
        long_c    = norm_one(s, budgets["long"])
        
        # Create variants with different styles
        for t, base_style, limit in [
            (overlay_c, "hook_short", budgets["overlay"]),
            (short_c,   "hook_mid",  budgets["short"]),
            (mid_c,     "how_to",    budgets["mid"]),
            (long_c,    "detailed",  budgets["long"])
        ]:
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            
            # Classify style and polish the title
            style = _classify_style(t)
            polished = _polish_title(t, style)
            
            variants.append({"title": polished, "style": style, "length": len(polished)})
        
        # Set overlay to first good short title
        if not overlay and overlay_c:
            overlay = overlay_c
        
        if len(variants) >= 5:
            break
    
    # Fallback overlay if none found
    if not overlay and variants:
        overlay = variants[0]["title"][:budgets["overlay"]]
    
    # Generate hashtags from keywords
    hashtags = _hashtags_from_keywords(keywords)
    
    # Use a fixed timestamp for caching consistency
    now = datetime.now(timezone.utc).isoformat()
    
    meta = {
        "keywords": keywords,
        "analytics": {
            "impressions": 0,
            "clicks": 0,
            "ctr": 0.0,
            "last_updated": now,
        }
    }
    
    if hashtags:
        meta["hashtags"] = hashtags
    
    return {
        "variants": variants[:5],
        "overlay": overlay,
        "engine": "v2",
        "generated_at": now,
        "version": 1,
        "meta": meta
    }

def generate_title_pack_v2(clip_id: str, platform: str, text: str, episode_text: str | None = None, seed_sentence: str | None = None, payoff_sentence: str | None = None) -> dict:
    """
    Generate a complete title pack with LRU caching for performance (v2 API).
    Returns dict with variants, overlay, engine, generated_at, meta fields.
    """
    from .keyword_extraction import extract_salient_keywords
    
    # Extract keywords for cache key (consistent with what's used in generation)
    keywords = extract_salient_keywords(text, limit=5) or extract_salient_keywords(episode_text or "", limit=5) or []
    
    # Generate cache key
    key = _cache_key(clip_id, platform, text, keywords)
    
    # Define the generation function with consistent keywords
    def _gen():
        return _generate_title_pack_uncached(clip_id, platform, text, episode_text, keywords, seed_sentence, payoff_sentence)
    
    # Return cached result
    return _cached_title_pack(key, _gen_fn=_gen)

def generate_title_pack(text: str, platform: str, episode_text: str | None = None, seed_sentence: str | None = None, payoff_sentence: str | None = None) -> dict:
    """
    Backward-compatible wrapper for existing callers (v1 API).
    Old callers use (text, platform). We map to v2 with an empty clip_id.
    Callers that have a real clip_id should prefer generate_title_pack_v2 for better caching/analytics.
    """
    clip_id = ""  # unknown here; callers like TitlesService don't pass it
    return generate_title_pack_v2(clip_id=clip_id, platform=platform, text=text, episode_text=episode_text, seed_sentence=seed_sentence, payoff_sentence=payoff_sentence)

# Generic scaffold patterns to penalize
GENERIC_SCAFFOLD_RE = re.compile(
    r"\b(what you need to know|everything you need to know|the truth about|all about|ultimate guide|complete guide)\b",
    re.I
)

def _soft_clean(s: str) -> str:
    """Clean title text: normalize whitespace, remove double punctuation, trim trailing spam"""
    s = re.sub(r"\s+", " ", s).strip()
    # remove double punctuation like "!!" or "??"
    s = re.sub(r"([!?.,:;])\1+", r"\1", s)
    # remove trailing punctuation spam
    s = re.sub(r"[!?.,:;]+$", "", s)
    return s

def _score_social(title: str, platform: str) -> float:
    """Score title for social media appeal: hook words + specificity + length sweet spot"""
    t = title.lower()
    score = 0.0
    
    # Hook words bonus
    if re.search(r"\b(why|how|stop|the secret|truth|mistake|don[']?t|you|this)\b", t):
        score += 0.6
    
    # Numbers / quantified claims bonus
    if re.search(r"(^|\s)(\d+[%$kKmMbB]?|#\d+)\b", title):
        score += 0.1
    
    # Named entities bonus (two proper-case tokens)
    if re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", title):
        score += 0.06
    
    # Length sweet spot
    n_words = len(title.split())
    if 5 <= n_words <= 12:
        score += 0.2
    
    # Penalize generic scaffolds so they don't dominate
    if GENERIC_SCAFFOLD_RE.search(title):
        score -= 0.25  # solid penalty for generic patterns
    
    # Tiny penalty for too generic
    if len(set(w for w in re.findall(r"[A-Za-z]+", t) if w not in {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"})) < 3:
        score -= 0.3
    
    return round(score, 3)

def _coherence(title: str, clip_text: str) -> float:
    """Score how well the title matches the clip content"""
    t = set(re.findall(r"[a-z0-9]+", title.lower()))
    c = set(re.findall(r"[a-z0-9]+", clip_text[:200].lower()))
    if not t or not c: return 0.0
    inter = len(t & c)
    return inter / max(1, min(len(t), len(c)))

def _dedup_ci(titles: list[str]) -> list[str]:
    """Case-insensitive deduplication of titles"""
    seen = set()
    out = []
    for tt in titles:
        k = re.sub(r"[\s\W_]+", "", tt).lower()
        if k not in seen:
            seen.add(k)
            out.append(tt)
    return out

def _extract_text(obj: Any) -> str:
    """Extract text from various object types (dict, list, string, etc.)"""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Prioritize exact transcript over other text fields
        if isinstance(obj.get("transcript"), str) and obj["transcript"].strip():
            return obj["transcript"]
        if "transcript" in obj and isinstance(obj["transcript"], dict):
            t = obj["transcript"].get("text")
            if isinstance(t, str) and t.strip():
                return t
        if isinstance(obj.get("segment_text"), str) and obj["segment_text"].strip():
            return obj["segment_text"]
        if isinstance(obj.get("text"), str):
            return obj["text"]
        # try words → join tokens
        if isinstance(obj.get("words"), list):
            toks = []
            for w in obj["words"]:
                if isinstance(w, dict):
                    toks.append(w.get("word") or w.get("w") or "")
                else:
                    toks.append(str(w))
            return " ".join(t for t in toks if t)
        # segments fallback
        if isinstance(obj.get("segments"), list):
            return " ".join((s.get("text") or "") for s in obj["segments"] if isinstance(s, dict))
        return str(obj)
    if isinstance(obj, list):
        parts = []
        for item in obj:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(obj)

def _looks_like_ad(text) -> bool:
    """Check if text looks like an advertisement - delegates to centralized detector"""
    from services.ads import looks_like_ad
    return looks_like_ad(text)

# Minimal English stopwords (expand if needed)
STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","from","as","at","by","is","are","was","were",
    "be","been","being","it","its","this","that","these","those","you","your","we","our","they","their","i",
    "he","she","him","her","his","hers","them","do","does","did","doing","have","has","had","having","not",
    "no","but","so","if","then","than","there","here","what","which","who","whom","into","over","under","about",
    # extra fillers that polluted titles
    "through","again","maybe","say","will","ill","youre","very","answer","two","words"
}

# Specific ban list for title "topic" words
BAN_TOPIC = {"this","that","those","there","within","and","because"}

_STOP = set("a an the and or for to in of with over under on at is are was were be been being it this that".split())

CONTRACTION_FIX = {
    r"\bweve\b": "we've",
    r"\btheyre\b": "they're", 
    r"\btheres\b": "there's",
    r"\bdoesnt\b": "doesn't",
    r"\bwont\b": "won't",
    r"\bcant\b": "can't",
    r"\bhavent\b": "haven't",
    r"\bhasnt\b": "hasn't",
    r"\bhadnt\b": "hadn't",
    r"\bwouldnt\b": "wouldn't",
    r"\bshouldnt\b": "shouldn't",
    r"\bcouldnt\b": "couldn't",
    r"\bthat that\b": "that",
    r"\bthe the\b": "the",
    r"\band and\b": "and",
}

def _normalize_contractions(s: str) -> str:
    s2 = s
    for pat, rep in CONTRACTION_FIX.items():
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    return s2

def _fallback_keywords(txt: str, n=3):
    if not txt:
        return []
    # crude nouny tokens: keep capitalized words & 4+ letter lowercase
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", txt)
    toks = [t for t in toks if t.lower() not in _STOP]
    # keep order, de-dup
    seen, out = set(), []
    for t in toks:
        k = t.lower()
        if k not in seen:
            seen.add(k); out.append(t)
        if len(out) >= n:
            break
    return out

def _deplaceholder(s: str, text: str) -> str:
    if "This Topic" not in s:
        return s
    kws = _fallback_keywords(text, n=3) or ["The Idea"]
    # simple replacements
    topic = " ".join(kws[:2]) if len(kws) >= 2 else kws[0]
    return s.replace("This Topic", topic)

# Patterns we never want in shorts titles
BANNED = re.compile(
    r"(how to(?!\s+\w)|in \d+\s+steps|secrets?|ultimate guide|tips & tricks|hack(s)?|unlock|master(?!\w)|click here)",
    re.I,
)

BANNED_PHRASES = {"Key Insight", "Key Takeaways", "Inside ", "What It Means", "What It Really Means", "Explained", "Understanding"}

BAN_PHRASE = re.compile(
    r"\b(appreciate|honored|thanks?|thank you|enjoyed|great to be here|see you|subscribe|like and subscribe)\b",
    re.I,
)

def _clean_text(t: str) -> str:
    t = (t or "").strip()
    # Keep hyphenated ("off-label") and apostrophes
    t = re.sub(r"[^A-Za-z0-9\s'\-\.,:;!?]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

# Domain anchors we care about; map variants -> canonical title token
ANCHORS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bdigital[-\s]?first\b", re.I), "Digital-First"),
    (re.compile(r"\brestructuring\b", re.I), "Restructuring"),
    (re.compile(r"\binvestment(s)?\b|\bfunding\b|\braise(d)?\b", re.I), "Investment"),
    (re.compile(r"\bpartnerships?\b", re.I), "Partnership"),
    (re.compile(r"\bacquisition(s)?\b|\bmerger(s)?\b", re.I), "Acquisition"),
    (re.compile(r"\bbankruptcy\b", re.I), "Bankruptcy"),
    (re.compile(r"\brevenue\b", re.I), "Revenue"),
    (re.compile(r"\bgrowth\b", re.I), "Growth"),
    (re.compile(r"\brefinanc(ing|e)\b|\bdebt\b", re.I), "Debt"),
    (re.compile(r"\bai\b", re.I), "AI"),
    # Sports/training domain
    (re.compile(r"\bpractice\b", re.I), "Practice"),
    (re.compile(r"\btraining\b", re.I), "Training"),
    (re.compile(r"\bdrills?\b", re.I), "Drills"),
    (re.compile(r"\bjuggl(e|ing)\b", re.I), "Juggling"),
    (re.compile(r"\bclub\b", re.I), "Club"),
    (re.compile(r"\bchelsea\b", re.I), "Chelsea"),
    (re.compile(r"\bajax\b|\biaxe\b", re.I), "Ajax"),
]

_MONEY = re.compile(r"(?:(?:\$?\s*\d+(?:\.\d+)?\s*(?:k|m|b))|(?:\$\s*\d[\d,]*\b)|(?:\d+\s*(?:million|billion)))", re.I)

def _fmt_money(s: str) -> str:
    s = s.lower().replace(",", "").replace("$", "").strip()
    if "million" in s:
        num = float(s.split()[0])
        return f"${num:.0f}M"
    if "billion" in s:
        num = float(s.split()[0])
        return f"${num:.0f}B"
    if s.endswith("k") or s.endswith("m") or s.endswith("b"):
        return f"${s.upper()}"
    # raw number: keep as-is with $ if it looked like money
    return f"${s}"

def _title_case_hyphen(s: str) -> str:
    parts = []
    for token in s.split():
        chunks = token.split("-")
        parts.append("-".join(c.capitalize() for c in chunks))
    return " ".join(parts)

def _mine_anchor_phrases(text: str) -> List[str]:
    """
    Extract compact, human-sounding phrases around domain anchors.
    Example: "Last year, Salem ... went through some major restructuring" -> "Salem Restructuring"
    """
    clean = _clean_text(text)
    # Also normalize contractions
    clean = _normalize_contractions(clean)
    # Tokens with original case for simple "proper noun" detection
    toks = re.findall(r"[A-Za-z][A-Za-z'-]*", clean)
    lowers = [t.lower() for t in toks]

    phrases: Counter = Counter()

    # Hyphenated terms are already good candidates (e.g., off-label)
    for h in re.findall(r"[A-Za-z]+-[A-Za-z]+", clean):
        # Filter out contractions in hyphenated terms
        if not any(contraction in h.lower() for contraction in ["they're", "we're", "you're", "it's", "that's", "there's", "here's", "doesn't", "won't", "can't"]):
            phrases[h.title()] += 2

    # Scan for each anchor and attach nearest proper noun to the left within a small window
    for rx, canon in ANCHORS:
        for m in rx.finditer(clean):
            # find token index closest to match start
            start_char = m.start()
            # map char offset to token index roughly by cumulative lengths
            acc = 0
            idx = 0
            for i, t in enumerate(toks):
                acc += len(t) + 1
                if acc >= start_char:
                    idx = i
                    break
            # search left for a proper-ish noun (capitalized and not a stopword)
            left = None
            for j in range(max(0, idx-6), idx):
                tj = toks[j]
                if tj[0].isupper() and tj.lower() not in STOP:
                    left = tj
            if left:
                phrase = f"{left} {canon}"
                phrases[phrase] += 4  # strong preference for <ProperNoun + Anchor>
            else:
                phrases[canon] += 2
            # money-aware: "<Left> $XM Investment"
            if canon == "Investment":
                # look ±8 tokens for a money figure
                low = max(0, idx-8); hi = min(len(toks), idx+8)
                window_text = " ".join(toks[low:hi])
                mny = _MONEY.search(window_text)
                if mny:
                    amt = _fmt_money(mny.group(0))
                    if left:
                        phrases[f"{left} {amt} Investment"] += 6
                    else:
                        phrases[f"{amt} Investment"] += 3

    # Also consider frequent 2–3 word ngrams that aren't all stopwords
    words = [w for w in lowers if w not in STOP]
    for n in (3, 2):
        for i in range(0, max(0, len(words)-n+1)):
            gram = words[i:i+n]
            if any(len(w) >= 6 for w in gram):
                phrase = " ".join(gram).title()
                # drop junk like "Players Anomalies Need" (reduces "Players Anomalies Need" style outputs)
                if any(w in ("need","because","there","just","some","then","also") for w in gram):
                    continue
                # Filter out contractions and common words that aren't good topics
                if (not BANNED.search(phrase) and not all(w in STOP for w in gram) and not BAN_PHRASE.search(phrase) and
                    not any(w in ["they're", "we're", "you're", "it's", "that's", "there's", "here's", "doesn't", "won't", "can't"] for w in gram)):
                    phrases[phrase] += 1

    # Return top unique phrases in order
    out: List[str] = []
    for p, _ in phrases.most_common(10):
        if p not in out:
            out.append(p)
    return out[:6]

def _title_from_text(text: str) -> str:
    """Create title from text, avoiding filler and repetition"""
    if not text:
        return "Clip"

    # Clean and tokenize
    toks = [t.strip(".,!?\"'():;").lower() for t in text.split()]
    toks = [t for t in toks if t and t not in STOP]

    # Collapse duplicates like "think think"
    dedup = []
    for t in toks:
        if not dedup or dedup[-1] != t:
            dedup.append(t)

    # Take top salient tokens
    key = dedup[:6]
    if not key:
        return "Clip"

    # Prefer descriptive, no "What It Means"
    return " ".join(w.capitalize() for w in key)

def _extract_noun_phrases(text) -> list:
    """Extract top TF-IDF noun chunks for better fallback titles"""
    # Normalize input to string
    raw_text = _extract_text(text)
    if not raw_text:
        return []
    
    # Simple noun phrase extraction (enhanced version)
    words = [w.strip(".,!?\"'():;").lower() for w in raw_text.split()]
    words = [w for w in words if w and w not in STOP and len(w) > 2]
    
    # Count frequency for TF-IDF-like scoring
    word_counts = Counter(words)
    total_words = len(words)
    
    # Score words by frequency and length
    scored_phrases = []
    for word, count in word_counts.items():
        if count >= 2:  # Must appear at least twice
            score = (count / total_words) * len(word)  # frequency * length
            scored_phrases.append((word, score))
    
    # Return top 3 noun phrases
    scored_phrases.sort(key=lambda x: x[1], reverse=True)
    return [phrase for phrase, _ in scored_phrases[:3]]

def _title_fallback(text, platform: str) -> str:
    """Content-aware title fallback with rotation to avoid repetition"""
    # Normalize input to string
    raw_text = _extract_text(text)
    if not raw_text:
        return "The most overlooked thing about this topic"
    
    # Extract content elements for pattern matching
    noun_phrases = _extract_noun_phrases(raw_text)
    
    # Better topic extraction with RAKE-style heuristic
    if noun_phrases:
        # Look for meaningful noun phrases (avoid common words)
        meaningful_phrases = []
        for phrase in noun_phrases[:3]:
            if len(phrase.split()) >= 2 and not any(word in phrase.lower() for word in ['thing', 'topic', 'this', 'that', 'what', 'how']):
                meaningful_phrases.append(phrase)
        
        if meaningful_phrases:
            topic = " ".join(meaningful_phrases[:2]).capitalize()
        else:
            # Fallback: use first noun phrase even if generic
            topic = noun_phrases[0].capitalize()
    else:
        # Last resort: extract from filename or use generic
        topic = "This Topic"
    
    # Content-aware pattern matching
    t = raw_text.lower()
    
    # Pattern 1: "don't/stop/avoid/mistake" → "Stop {mistake}. Do {better} instead."
    if any(word in t for word in ["don't", "stop", "avoid", "mistake", "wrong", "bad"]):
        mistake = topic if topic != "This Topic" else "This Common Mistake"
        better = "a Better Approach" if "approach" not in t else "the Right Way"
        return f"Stop {mistake}. Do {better} instead."
    
    # Pattern 2: claim + "because/so/which means" → "{Counterintuitive claim}. Here's why."
    if any(word in t for word in ["because", "so", "which means", "therefore", "as a result"]):
        claim = topic if topic != "This Topic" else "This Counterintuitive Truth"
        return f"{claim}. Here's why."
    
    # Pattern 3: instructional verbs ("how/steps/tips") → "How to {outcome} without {pain}"
    if any(word in t for word in ["how", "steps", "tips", "guide", "learn", "master"]):
        outcome = topic if topic != "This Topic" else "Success"
        pain = "the Common Pitfalls" if "pitfall" not in t else "Making Mistakes"
        return f"How to {outcome} without {pain}"
    
    # Pattern 4: default → "The most overlooked {X} about {topic}"
    # Rotate through variations to avoid repetition
    variations = [
        f"The most overlooked {topic} about {topic}",
        f"Why {topic} actually works",
        f"The {topic} secret nobody talks about",
        f"What {topic} really means"
    ]
    
    # Simple rotation based on text hash
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return variations[hash_val % len(variations)]

def _sanitize_title(title: str) -> str:
    """Always-on title sanitizer to remove boilerplate and generic content"""
    if not title:
        return "The most overlooked thing about this topic"
    
    # Check for banned boilerplate patterns
    boilerplate = re.compile(r'\b(what it means|explained|key takeaways|the truth about|everything you need to know|what it really means|key insight)\b', re.I)
    if boilerplate.search(title):
        # Use value-forward fallback
        return _title_fallback(title, "tiktok")
    
    # Check for too generic content
    if len(title.split()) < 3 or title.lower() in ["clip", "moment", "insight", "key insight"]:
        return _title_fallback(title, "tiktok")
    
    return title

def _extract_keywords(text: str, k: int = 6) -> List[str]:
    """
    Lightweight keyword extractor geared for clip transcripts.
    Prefers proper nouns and short noun-ish phrases. Avoids pronouns and glue words.
    """
    if not text:
        return []
    t = re.sub(r"\s+", " ", text.strip())
    tokens = re.findall(r"[A-Za-z0-9$][A-Za-z0-9\-\.''']*", t)

    # Candidate words with simple scoring
    scores = {}
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in STOP or len(low) < 3:
            continue
        score = 1.0
        if tok[:1].isupper() and i not in (0, 1):  # proper-ish noun mid-sentence
            score += 1.0
        if re.match(r"^\$?\d", tok):               # numbers/tickers
            score += 0.6
        scores[low] = max(scores.get(low, 0), score)

    # Simple phrase pass (bigrams/trigrams) that avoid STOP on the ends
    phrases = {}
    L = [w for w in tokens]
    for n in (3, 2):
        for i in range(len(L) - n + 1):
            phrase = " ".join(L[i:i+n])
            pl = phrase.lower()
            parts = pl.split()
            if any(p in STOP for p in (parts[0], parts[-1])):
                continue
            if any(len(p) < 3 for p in parts):
                continue
            # bump score if most parts have capital letters
            cap_bonus = sum(1 for p in L[i:i+n] if p[:1].isupper())
            phrases[pl] = phrases.get(pl, 0) + n + 0.3 * cap_bonus

    # Merge & pick top-k
    merged = {**scores, **phrases}
    top = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[: max(k, 6)]
    out = []
    seen = set()
    for term, _ in top:
        if term not in seen:
            seen.add(term)
            out.append(_title_case(term))
        if len(out) >= k:
            break
    return out

# Removed duplicate _title_case function - using the one defined earlier

def _variants_from_keywords(keys: List[str], platform: str) -> List[str]:
    # assemble 6 concise variants; prefer 4–8 words; title-case; avoid banned phrases
    # Use safer topic selection that avoids BAN_TOPIC words
    topic = next((kw for kw in keys if kw.lower() not in BAN_TOPIC), None)
    if not topic:
        return []
    main = topic.replace("-", " ").strip()
    main_tc = _title_case_hyphen(main)
    alts = []
    if platform == "shorts":
        alts = [
            f"The Truth About {main_tc}",
            f"Why {main_tc} Actually Works",
            f"What Everyone Misses About {main_tc}",
            f"{main_tc}: The Real Story",
            f"{main_tc} — Fast Facts",
            f"{main_tc} In Plain English",
        ]
    else:
        alts = [
            f"The Counterintuitive Truth About {main_tc}",
            f"Why {main_tc} Matters More Than You Think",
            f"The {main_tc} Method That Gets Results",
            f"What {main_tc} Experts Don't Tell You",
            f"How {main_tc} Really Works",
            f"Understanding {main_tc}",
        ]
    # filter banned & overly long, dedupe
    out = []
    seen = set()
    for t in alts:
        t = re.sub(r"\s+", " ", t).strip()
        if BANNED.search(t):
            continue
        if 4 <= len(t.split()) <= 10 and len(t) <= 80:
            if t.lower() not in seen:
                out.append(t)
                seen.add(t.lower())
    return out

def _safe_fallback(clip_title: str, platform: str) -> List[str]:
    base = clip_title.strip()[:60] if clip_title else "This Moment"
    base = _title_case(base)
    tail = "The Real Story" if platform == "shorts" else "What You Need to Know"
    return [f"{base}: {tail}"]

# ---------- Title caching & deduplication ----------
TITLE_GEN_VERSION = "v3"
_title_cache = {}  # Simple in-memory cache
CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds
_current_episode = None

def maybe_reset_cache(episode_id: str):
    """Reset cache when processing a new episode to prevent bleed"""
    global _current_episode, _title_cache
    if _current_episode != episode_id:
        _title_cache.clear()
        _current_episode = episode_id

def _norm(s: str) -> str:
    """Normalize text for consistent hashing"""
    return " ".join(s.lower().split())

def _title_cache_key(*, episode_id: str, clip_id: str, platform: str, text: str) -> str:
    """Generate collision-resistant cache key with all identifiers"""
    h = hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:12]
    return f"{TITLE_GEN_VERSION}:{episode_id}:{clip_id}:{platform}:{h}"

def _get_cached_titles(cache_key: str) -> Optional[List[Dict]]:
    """Get titles from cache if not expired"""
    if cache_key in _title_cache:
        cached_data, timestamp = _title_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
        else:
            # Remove expired entry
            del _title_cache[cache_key]
    return None

def _cache_titles(cache_key: str, titles: List[Dict]) -> None:
    """Cache titles with timestamp"""
    _title_cache[cache_key] = (titles, time.time())

def dedup_titles(titles: List[Dict]) -> List[Dict]:
    """Remove duplicate titles within a single clip"""
    seen = set()
    out = []
    for t in titles:
        k = t["title"].strip().lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out

# ---------- Platform limits & normalization ----------
# Using PLAT_LIMITS from config.settings to avoid configuration drift

def normalize_platform(p: Optional[str]) -> str:
    if not p:
        return "default"
    p = p.lower()
    if p in {"tiktok", "tt", "tik_tok"}: return "tiktok"
    if p in {"reels", "instagram", "ig", "ig_reels"}: return "reels"
    if p in {"shorts", "yt_shorts", "youtube_shorts"}: return "shorts"
    if p in {"youtube", "yt"}: return "youtube"
    return "default"

# ---------- Text normalization and cleaning ----------
def normalize_text(text: str) -> str:
    """Normalize text for better title generation"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common conversational fillers
    fillers = [
        r'\b(um|uh|er|ah|like|you know|I mean|so|well|right|okay|ok)\b',
        r'\b(thank you|thanks|sure|yeah|yep|nope|no problem)\b',
        r'\b(actually|basically|literally|obviously|clearly|definitely)\b'
    ]
    
    for pattern in fillers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up punctuation
    text = re.sub(r'[^\w\s\-.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Improved stop words list (includes previously problematic words)
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "for", "on", "in", "at", "with",
    "like", "you", "i", "we", "they", "he", "she", "it", "that", "this", "these", "those",
    "just", "really", "gonna", "got", "kind", "sort", "thing", "things", "stuff",
    # previously junky offenders:
    "need", "team", "sample",
    # additional common words
    "is", "are", "was", "were", "be", "as", "your", "our", "me", "my", "them", "their", 
    "his", "her", "us", "do", "did", "does", "have", "has", "had", "if", "can", "could", 
    "should", "would", "will", "may", "might", "must", "shall"
}

def key_terms(text: str, max_terms: int = 4) -> List[str]:
    """Extract key terms from text using improved heuristics"""
    if not text:
        return []
    
    # Normalize text first
    text = normalize_text(text)
    # Also normalize contractions
    text = _normalize_contractions(text)
    
    # Extract words (letters, hyphens, apostrophes)
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", text)]
    
    # Filter out stop words and short words
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    if not words:
        return []
    
    # Calculate frequency and position
    freq, order = {}, {}
    for i, w in enumerate(words):
        freq[w] = freq.get(w, 0) + 1
        order.setdefault(w, i)
    
    # Rank by frequency (descending), then by position (ascending)
    ranked = sorted(freq, key=lambda w: (-freq[w], order[w]))
    
    return ranked[:max_terms]

def clean_for_keywords(text: str) -> str:
    """Clean text for keyword extraction using improved heuristics"""
    if not text:
        return ""
    
    # Normalize first
    text = normalize_text(text)
    
    # Use the improved key_terms function
    terms = key_terms(text, max_terms=10)
    
    return ' '.join(terms)

# ---------- Relaxed title validation ----------
def is_valid_title(title: str) -> bool:
    """Check if title is valid with relaxed rules"""
    if not title or len(title.strip()) < 10:
        return False
    
    # Check for obvious nonsense patterns
    nonsense_patterns = [
        r'^\s*$',  # Empty or whitespace only
        r'^[^a-zA-Z]*$',  # No letters
        r'^(the|a|an)\s+(the|a|an)\s+',  # Double articles
        r'\b(these|this|that|more|less|first|second|third)\s+(these|this|that|more|less|first|second|third)\b',  # Repetitive words
    ]
    
    for pattern in nonsense_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return False
    
    # Check for reasonable word count
    words = title.split()
    if len(words) < 3 or len(words) > 15:
        return False
    
    return True

def fix_title(title: str) -> str:
    """Fix common title issues"""
    if not title:
        return "The Strategy That Works"
    
    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title.strip())
    
    # Fix common capitalization issues
    title = title.title()
    
    # Fix common article issues
    title = re.sub(r'\b(The|A|An)\s+(The|A|An)\b', r'\1', title)
    
    # Ensure it starts with a capital letter
    if title and not title[0].isupper():
        title = title[0].upper() + title[1:]
    
    return title

# ---------- Content Analysis Functions ----------
def _analyze_content_insights(text: str) -> dict:
    """Extract specific insights, emotions, and key concepts from content"""
    insights = {
        'emotion': 'neutral',  # excited, concerned, confident, etc.
        'urgency': 'low',      # high, medium, low
        'specifics': [],       # numbers, names, concrete details
        'contrast': [],        # before/after, right/wrong, etc.
        'actionable': []       # steps, tips, advice
    }
    
    text_lower = text.lower()
    
    # Detect emotional tone
    if any(word in text_lower for word in ['amazing', 'incredible', 'shocking', 'surprising', 'breakthrough', 'revolutionary']):
        insights['emotion'] = 'excited'
    elif any(word in text_lower for word in ['concerned', 'worried', 'problem', 'issue', 'failing', 'struggling']):
        insights['emotion'] = 'concerned'
    elif any(word in text_lower for word in ['confident', 'proven', 'successful', 'effective', 'works']):
        insights['emotion'] = 'confident'
    
    # Detect urgency
    if any(word in text_lower for word in ['urgent', 'immediately', 'right now', 'critical', 'emergency', 'asap']):
        insights['urgency'] = 'high'
    elif any(word in text_lower for word in ['soon', 'quickly', 'fast', 'rapid', 'swift']):
        insights['urgency'] = 'medium'
    
    # Extract specific numbers, percentages, timeframes
    specifics = re.findall(r'\b(\d+%?|\$\d+[kmb]?|\d+\s*(?:years?|months?|days?|hours?|minutes?))\b', text_lower)
    insights['specifics'] = specifics[:3]  # Limit to top 3
    
    # Detect contrast patterns
    contrast_words = ['vs', 'versus', 'instead', 'rather', 'but', 'however', 'although', 'while']
    if any(word in text_lower for word in contrast_words):
        insights['contrast'] = ['contrast_detected']
    
    # Detect actionable content
    action_words = ['how to', 'steps', 'method', 'process', 'technique', 'approach', 'strategy']
    if any(phrase in text_lower for phrase in action_words):
        insights['actionable'] = ['actionable_detected']
    
    return insights

def _extract_meaningful_keywords(text: str, max_terms: int = 5) -> List[str]:
    """Extract keywords that actually matter for titles"""
    if not text:
        return []
    
    # Remove filler and normalize
    clean_text = normalize_text(text)
    keywords = []
    
    # 1. Numbers and specific values (highest priority)
    numbers = re.findall(r'\b\d+[%$kmb]?\b', clean_text)
    keywords.extend(numbers[:2])
    
    # 2. Action words and outcomes (prioritize these)
    action_words = re.findall(r'\b(?:achieve|create|build|solve|improve|increase|reduce|eliminate|master|learn|develop|know|want|do|think|feel|believe|understand|realize|discover|creating|building|solving|improving|learning|developing|converting|selling|marketing|growing|scaling)\w*\b', clean_text.lower())
    keywords.extend([w.title() for w in action_words[:3]])
    
    # 3. Domain-specific terms
    domain_terms = re.findall(r'\b(?:strategy|method|approach|system|process|framework|model|technique|principle|school|education|career|life|identity|self|purpose|passion|dream|goal|future)\b', clean_text.lower())
    keywords.extend([w.title() for w in domain_terms[:2]])
    
    # 4. Important concepts (longer words that aren't stop words) - prioritize these
    words = re.findall(r'\b[A-Za-z]{4,}\b', clean_text.lower())
    important_words = [w for w in words if w not in STOP_WORDS and len(w) > 4]
    # Filter out generic words that aren't meaningful for titles
    generic_words = {'does', 'again', 'honestly', 'really', 'actually', 'basically', 'literally', 'obviously', 'clearly', 'definitely', 'probably', 'maybe', 'sometimes', 'always', 'never', 'often', 'usually', 'normally', 'typically', 'generally', 'mostly', 'mainly', 'primarily', 'especially', 'particularly', 'specifically', 'exactly', 'precisely', 'absolutely', 'completely', 'totally', 'entirely', 'perfectly', 'exactly', 'precisely', 'absolutely', 'completely', 'totally', 'entirely', 'perfectly'}
    meaningful_words = [w for w in important_words if w not in generic_words]
    # Take the most frequent meaningful words
    word_counts = Counter(meaningful_words)
    top_words = [word.title() for word, count in word_counts.most_common(3)]
    keywords.extend(top_words)
    
    # 5. Proper nouns and entities (only if they're meaningful, not common words)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_text)
    # Filter out common words that happen to be capitalized
    common_caps = {'But', 'You', 'I', 'The', 'This', 'That', 'What', 'When', 'Where', 'Why', 'How', 'Always', 'Never', 'Sometimes', 'Really', 'Actually', 'Exactly', 'Definitely', 'Probably', 'Maybe', 'Just', 'Only', 'Even', 'Still', 'Already', 'Also', 'Too', 'Very', 'Really', 'Quite', 'Pretty', 'Rather', 'Fairly', 'Somewhat', 'Kind', 'Sort', 'Type', 'Way', 'Thing', 'Things', 'Stuff', 'Something', 'Anything', 'Nothing', 'Everything', 'Someone', 'Anyone', 'Noone', 'Everyone'}
    meaningful_entities = [e for e in entities if e not in common_caps and len(e) > 3]
    keywords.extend(meaningful_entities[:2])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw.lower() not in seen and len(kw) > 2:
            seen.add(kw.lower())
            unique_keywords.append(kw)
    
    return unique_keywords[:max_terms]

# ---------- Content-Aware Title Templates ----------
CONTENT_AWARE_TEMPLATES = {
    'excited': [
        "The {specific} That Changed Everything",
        "Why {topic} Is {specific} (And Why It Matters)",
        "The {topic} Breakthrough Nobody Saw Coming",
        "How {topic} Just Got {specific}",
        "The {topic} Revolution That's Here"
    ],
    'concerned': [
        "The {topic} Problem Everyone's Ignoring",
        "Why {topic} Is Failing (And How to Fix It)",
        "The Hidden Risk in {topic}",
        "What's Really Wrong With {topic}",
        "The {topic} Crisis Nobody Talks About"
    ],
    'confident': [
        "The {topic} Method That Actually Works",
        "Why {topic} Is the {specific} Solution",
        "The {topic} Approach That Gets Results",
        "How {topic} Delivers {specific}",
        "The {topic} Strategy That Never Fails"
    ],
    'actionable': [
        "The {number} {topic} Rules That Actually Work",
        "How to {action} in {timeframe}",
        "The {topic} Method That Saves {benefit}",
        "Step-by-Step: {topic} That Works",
        "The {topic} Process Everyone Should Know"
    ],
    'contrast': [
        "Why {old_way} Fails (And {new_way} Works)",
        "The {topic} Difference Between Success and Failure",
        "What {topic} Experts Do Differently",
        "The {topic} Truth vs. The {topic} Myth",
        "Why {topic} A Works But {topic} B Doesn't"
    ],
    'default': [
        "The {topic} Insight That Changes Everything",
        "Why {topic} Matters More Than You Think",
        "The {topic} Approach That Actually Works",
        "What {topic} Really Means",
        "The {topic} Method That Gets Results"
    ]
}

# ---------- Legacy Title Templates (kept for fallback) ----------
TITLE_TEMPLATES = [
    # Decision-focused
    "Ask This Before You Decide",
    "Before You Decide, Do This",
    "The Question That Changes Everything",
    
    # Problem-solving
    "The Hidden Problem Most People Miss",
    "Why {topic} Fails (And How to Fix It)",
    "The {topic} Mistake Everyone Makes",
    
    # Strategy-focused
    "The {topic} Strategy That Actually Works",
    "How to Master {topic} in 5 Steps",
    "The Secret to {topic} Success",
    
    # Leadership-focused
    "What Great Leaders Do Differently",
    "The Leadership Lesson Nobody Talks About",
    "How to Lead When Everything's Uncertain",
    
    # Team-focused
    "The Team Dynamic That Changes Everything",
    "Why Your Team Struggles (And How to Fix It)",
    "The Feedback Method That Actually Works",
    
    # CFD-first fallbacks (no generic spam)
    "Why {topic} Actually Works",
    "The {topic} Method That Saves Time", 
    "What Everyone Misses About {topic}",
]

def extract_topic(text: str) -> str:
    """Extract the main topic from text using improved heuristics"""
    if not text:
        return "This Topic"
    
    # Normalize contractions first
    normalized_text = _normalize_contractions(text)
    
    # Get key terms using improved extraction
    terms = key_terms(normalized_text, max_terms=6)
    
    if not terms:
        return "This Topic"
    
    # Filter out contractions and common words that aren't good topics
    filtered_terms = []
    for term in terms:
        term_lower = term.lower()
        if (term_lower not in ["they're", "we're", "you're", "it's", "that's", "there's", "here's", "doesn't", "won't", "can't"] and 
            term_lower not in _STOP and len(term) > 2):
            filtered_terms.append(term)
    
    if not filtered_terms:
        return "This Topic"
    
    # Look for domain-specific terms first
    domain_terms = [
        'off-label', 'prescriptions', 'medications', 'healthcare', 'medical', 'doctors', 'patients',
        'strategy', 'leadership', 'team', 'feedback', 'decision', 'problem', 'solution',
        'management', 'communication', 'culture', 'growth', 'innovation', 'change',
        'planning', 'execution', 'performance', 'collaboration', 'trust', 'vision',
        'practice', 'training', 'coaching', 'development', 'improvement', 'skill'
    ]
    
    # Find the most relevant domain term
    for term in domain_terms:
        if term in [t.lower() for t in filtered_terms]:
            return term.title()
    
    # Use the first filtered term as topic
    return filtered_terms[0].title()

def _generate_dynamic_titles(text: str, platform: str) -> List[str]:
    """Generate titles based on actual content analysis"""
    if not text:
        return ["Quick Tip", "Coach's Insight", "One Thing Most Players Miss"]
    
    insights = _analyze_content_insights(text)
    keywords = _extract_meaningful_keywords(text)
    
    titles = []
    
    # Use the most specific keyword as the main topic
    main_topic = keywords[0] if keywords else "This Strategy"
    
    # Get template category based on insights
    template_category = 'default'
    if insights['actionable']:
        template_category = 'actionable'
    elif insights['contrast']:
        template_category = 'contrast'
    elif insights['emotion'] != 'neutral':
        template_category = insights['emotion']
    
    # Generate titles using content-aware templates
    templates = CONTENT_AWARE_TEMPLATES.get(template_category, CONTENT_AWARE_TEMPLATES['default'])
    
    for template in templates[:3]:  # Use top 3 templates
        try:
            # Replace placeholders with actual content
            title = template.format(
                topic=main_topic,
                specific=insights['specifics'][0] if insights['specifics'] else "Game-Changing",
                number=insights['specifics'][0] if insights['specifics'] and insights['specifics'][0].isdigit() else "3",
                action=keywords[1] if len(keywords) > 1 else "Succeed",
                timeframe=insights['specifics'][0] if insights['specifics'] else "30 Days",
                benefit="Time" if 'time' in text.lower() else "Results",
                old_way=keywords[0] if keywords else "Old Method",
                new_way=keywords[1] if len(keywords) > 1 else "New Method"
            )
            titles.append(title)
        except (KeyError, IndexError):
            # If template formatting fails, use a simple fallback
            titles.append(f"The {main_topic} Approach That Works")
    
    # Add question-based titles if content has questions
    if '?' in text:
        titles.append(f"What {main_topic} Really Means")
        titles.append(f"Why {main_topic} Matters More Than You Think")
    
    # Add specific number-based titles if we have specifics
    if insights['specifics']:
        specific = insights['specifics'][0]
        titles.append(f"The {specific} {main_topic} That Actually Works")
    
    # Remove duplicates and limit
    seen = set()
    unique_titles = []
    for title in titles:
        if title.lower() not in seen:
            seen.add(title.lower())
            unique_titles.append(title)
    
    return unique_titles[:6]

def make_titles(text: str) -> List[str]:
    """Generate titles using improved heuristics with real text"""
    if not text:
        return ["Quick Tip", "Coach's Insight", "One Thing Most Players Miss"]
    
    # Use the new dynamic title generation
    dynamic_titles = _generate_dynamic_titles(text, "default")
    
    # Fallback to old method if dynamic generation fails
    if not dynamic_titles:
        # Normalize contractions first
        normalized_text = _normalize_contractions(text)
        terms = key_terms(normalized_text, max_terms=4)
        options = []
        
        if terms:
            head = " ".join(t.capitalize() for t in terms[:2])
            options += [
                f"Stop Ignoring {head}",
                f"{head}: The Mistake Everyone Makes",
                f"Do This to Improve {terms[0].capitalize()}",
                f"Why {terms[0].capitalize()} Matters More Than You Think",
            ]
        
        # Add a direct, cleaned sentence if present
        sentences = text.strip().split(".")
        if sentences:
            s = sentences[0].strip()[:80]
            if len(s) > 12:
                options.append(s.strip() + "…")
        
        # Dedupe & cap to 6
        seen, deduped = set(), []
        for t in options:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        
        return deduped[:6]
    
    return dynamic_titles

# Ad detection markers (moved to _looks_like_ad function)

def generate_titles(
    text,
    *,
    platform: Optional[str] = None,
    n: int = 4,
    avoid_titles: Optional[Iterable[str]] = None,
    episode_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    seed_sentence: Optional[str] = None,
    payoff_sentence: Optional[str] = None,
    episode_vocab: Optional[dict] = None,
) -> List[Dict[str, object]]:
    """
    Deterministic, topic-aware titles from the actual clip text.
    Avoids generic/banned clickbait; returns 6 or a safe fallback.
    """
    
    # Normalize input to string early
    raw = _extract_text(text)
    if not raw.strip():
        # upstream bug or empty transcript; fail gracefully instead of 500
        return []
    
    # Check for V2 title engine feature flag (default to true)
    if TITLE_ENGINE_V2:
        try:
            from services.title_engine_v2 import generate_titles_v2 as _generate_titles_v2_raw
            clip_attrs = {
                "payoff_ok": True,  # Default to True, can be overridden
                "entity": None,
                "domain_topic": None,
                "keyword": None,
            }
            # Add any additional attrs from episode_vocab if available
            if episode_vocab:
                clip_attrs.update(episode_vocab)
            
            v2_results = generate_titles_v2(text, platform)
            
            if v2_results:
                logger.info(f"TITLE_ENGINE_V2: generated {len(v2_results)} titles")
                # Apply enhanced scoring, cleaning, and deduplication
                platform = normalize_platform(platform)
                limit = PLAT_LIMITS.get(platform, PLAT_LIMITS["default"])
                cleaned = [_soft_clean(x["title"] if isinstance(x, dict) else str(x)) for x in v2_results]
                cleaned = [t[:limit].rstrip(" .,!?;:") for t in cleaned]  # hard clip to limit
                # Rank by new quality scoring + coherence with clip content
                ranked = sorted(
                    cleaned,
                    key=lambda t: 0.7*_score_title_quality(t, text, platform) + 0.3*_coherence(t, text),
                    reverse=True
                )
                ranked = _dedup_ci(ranked)
                return [{"title": t} for t in ranked[:6]]
            else:
                logger.info("TITLE_ENGINE_V2: no results, falling back to V1")
        except Exception as e:
            logger.warning(f"TITLE_ENGINE_V2: failed with {e}, falling back to V1")
    
    # hard-drop obvious ads from titling
    if _looks_like_ad(raw):
        return []  # no titles for ads; upstream should already de-prefer these
    
    # Normalize inputs
    clean_text = normalize_text(raw)
    platform = normalize_platform(platform)
    avoid_set = set(avoid_titles or [])
    
    # Use seed/payoff sentences if provided (Phase 3 integration)
    hook_text = seed_sentence or clean_text.split('.')[0][:120] if clean_text else ""
    payoff_text = payoff_sentence or clean_text[-200:] if clean_text else ""
    
    # Reset cache on new episode
    if episode_id:
        maybe_reset_cache(episode_id)
    
    # Check cache
    if episode_id and clip_id and platform:
        cache_key = _title_cache_key(episode_id=episode_id, clip_id=clip_id, platform=platform, text=clean_text)
        cached_titles = _get_cached_titles(cache_key)
        if cached_titles:
            # Filter out avoided titles
            filtered_titles = [t for t in cached_titles if t["title"] not in avoid_set]
            if filtered_titles:
                return filtered_titles[:n]
    
    # Use dynamic title generation as the PRIMARY method
    dynamic_titles = _generate_dynamic_titles(clean_text, platform) or []
    
    # NEW: Integrate seed/payoff sentences for enhanced title generation (Phase 2 & 3)
    hook = (seed_sentence or "").strip()
    payoff = (payoff_sentence or "").strip()
    
    if hook and payoff:
        # Create hook→payoff pattern titles
        dynamic_titles.insert(0, f"{hook} → {payoff}")
        dynamic_titles.insert(1, f"Why {hook.rstrip('?')}? {payoff}")
        dynamic_titles.insert(2, f"{hook} (and {payoff})")
    elif hook:
        # Use seed sentence as primary hook
        dynamic_titles.insert(0, hook[:80])
        dynamic_titles.insert(1, f"Why {hook.rstrip('?')}?")
    elif payoff:
        # Use payoff sentence for conclusion-focused titles
        dynamic_titles.insert(0, f"The Key Insight: {payoff[:60]}")
        dynamic_titles.insert(1, f"Here's What Matters: {payoff[:60]}")
    
    if dynamic_titles:
        variants = dynamic_titles
        # Extract keywords for logging
        keys = _extract_meaningful_keywords(clean_text, max_terms=6)
    else:
        # Fallback to improved keyword-based method
        keys = _extract_meaningful_keywords(clean_text, max_terms=6)
        variants = _variants_from_keywords(keys, platform=platform)
        
        if not variants:
            # Check if we need fallback (banned boilerplate or < 3 meaningful tokens)
            needs_fallback = False
            if clean_text:
                # Check for banned boilerplate
                boilerplate = re.compile(r'\b(what it means|explained|key takeaways|the truth about|everything you need to know)\b', re.I)
                if boilerplate.search(clean_text):
                    needs_fallback = True
                
                # Check for < 3 meaningful tokens
                meaningful_tokens = [w for w in clean_text.split() if w.lower() not in STOP and len(w) > 2]
                if len(meaningful_tokens) < 3:
                    needs_fallback = True
            
            if needs_fallback and clean_text:
                base_title = _title_fallback(clean_text, platform)
                variants = [
                    base_title,
                    f"{base_title}: Key Takeaway",
                    f"Why {base_title}",
                ]
            elif clean_text:
                # Use improved title function
                base_title = _title_from_text(clean_text)
                variants = [
                    base_title,
                    f"{base_title}: Key Takeaway",
                    f"Why {base_title}",
                ]
            else:
                # Safe fallback
                title_hint = clean_text[:60] if clean_text else "This Moment"
                variants = _safe_fallback(title_hint, platform)
        
        # title-case lightly
        variants = [v[:1].upper() + v[1:] for v in variants if len(v) > 0]
        variants = variants[:5]
    
    logger.info("TITLES: k=%s variants=%s", keys[:4], variants[:3])
    
    # Convert to expected format with title sanitization
    result = []
    for i, title in enumerate(variants[:n]):
        if title not in avoid_set:
            # Always-on title sanitizer
            sanitized_title = _sanitize_title(title)
            # Replace "This Topic" placeholders with topicful text
            sanitized_title = _deplaceholder(sanitized_title, clean_text)
            # also collapse duplicated patterns
            sanitized_title = sanitized_title.replace(": What It Means", "").replace(": What It Really Means", "").replace(": Explained", "").replace(": Understanding", "")
            # Normalize contractions and fix grammar issues
            sanitized_title = _normalize_contractions(sanitized_title)
            # Fix common grammar issues
            sanitized_title = re.sub(r'\bthat that\b', 'that', sanitized_title, flags=re.IGNORECASE)
            sanitized_title = re.sub(r'\bthe the\b', 'the', sanitized_title, flags=re.IGNORECASE)
            sanitized_title = re.sub(r'\band and\b', 'and', sanitized_title, flags=re.IGNORECASE)
            
            # Additional grammar fixes for Phase 2/3
            sanitized_title = re.sub(r"\bHow to\s+(\w+?)(ing)\b", r"How to \1", sanitized_title, flags=re.I)   # Understanding -> Understand
            sanitized_title = re.sub(r"\bWhy\s+doesnt\b", "Why Doesn't", sanitized_title, flags=re.I)
            sanitized_title = re.sub(r"\bThe\s+3\s+\b\bRules\b", "The 3 Rules", sanitized_title)              # remove accidental double spaces
            sanitized_title = re.sub(r"\s{2,}", " ", sanitized_title).strip()
            # Remove dangling "How to  in 30 Days" gaps
            sanitized_title = re.sub(r"How to\s+in\s+(\d+\s*Days)", r"How to in \1", sanitized_title)  # if still broken, at least not empty slot
            # Check for banned phrases and rewrite if needed
            for banned in BANNED_PHRASES:
                if banned.lower() in sanitized_title.lower():
                    sanitized_title = sanitized_title.replace(banned, "").strip()
                    if not sanitized_title:
                        sanitized_title = "Important Insight"
            sanitized = sanitized_title != title
            
            # Use new quality scoring
            quality_score = _score_title_quality(sanitized_title, clean_text, platform)
            
            result.append({
                "title": sanitized_title,
                "score": quality_score,
                "reasons": ["Generated from text analysis"]
            })
            
            if sanitized:
                logger.info("TITLE_SANITIZED=True original='%s' final='%s'", title, sanitized_title)
    
    # Cache the result
    if episode_id and clip_id and platform:
        cache_key = _title_cache_key(episode_id=episode_id, clip_id=clip_id, platform=platform, text=clean_text)
        _cache_titles(cache_key, result)
    
    # 🔧 GRAMMAR GUARD: Fix common grammar issues
    if isinstance(result, list):
        result = [{"title": _grammar_guard(t.get("title", t) if isinstance(t, dict) else str(t))} if isinstance(t, dict) else _grammar_guard(str(t)) for t in result]
    
    return result

def _grammar_guard(t: str) -> str:
    """Fix common grammar issues in titles without blocking the pipeline"""
    t = t.replace("How to Moving", "How to Move")
    t = t.replace("How to Information", "How to Get Information")
    t = re.sub(r"\bWhy (\w+) Matters\b", r"Why \1 Matters", t)  # keeps common pattern ok
    return t

def _score_title_quality(title: str, text: str, platform: str) -> float:
    """Score title based on engagement potential and content relevance"""
    score = 0.0
    
    # Content relevance (most important - 40% of score)
    title_words = set(re.findall(r'\w+', title.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))
    overlap = len(title_words & text_words)
    if text_words:
        relevance_score = min(0.4, overlap / len(text_words) * 2)
        score += relevance_score
    
    # Engagement factors (30% of score)
    if re.search(r'\b(why|how|what|when|where)\b', title.lower()):
        score += 0.15  # Questions are engaging
    
    if re.search(r'\b\d+[%$kmb]?\b', title):
        score += 0.1  # Specific numbers
    
    if re.search(r'\b(you|your)\b', title.lower()):
        score += 0.05  # Second person
    
    # Length optimization (20% of score)
    length = len(title)
    platform_limit = PLAT_LIMITS.get(platform, PLAT_LIMITS["default"])
    if 20 <= length <= platform_limit:
        score += 0.1  # Good length
    elif length < 20:
        score += 0.05  # Too short but acceptable
    else:
        overage = length - platform_limit
        score -= min(0.1, overage * 0.01)  # Penalty for too long
    
    # Avoid generic phrases (10% penalty)
    generic_penalty = 0.0
    if re.search(r'\b(truth|secret|hidden|amazing|incredible)\b', title.lower()):
        generic_penalty += 0.05
    if re.search(r'\b(everyone|nobody|everybody)\b', title.lower()):
        generic_penalty += 0.03
    if re.search(r'\b(ultimate|complete|everything you need)\b', title.lower()):
        generic_penalty += 0.05
    
    score -= generic_penalty
    
    return max(0.0, min(1.0, score))

def score_title(title: str, platform: str, topic: str, text: str) -> float:
    """Score a title based on various factors (legacy function)"""
    # Use the new quality scoring as primary
    quality_score = _score_title_quality(title, text, platform)
    
    # Add some legacy factors for backward compatibility
    score = quality_score
    
    # Topic relevance bonus
    if topic.lower() in title.lower():
        score += 0.1
    
    # Question titles bonus
    if title.endswith('?'):
        score += 0.05
    
    return round(min(1.0, score), 2)

def get_score_reasons(title: str, platform: str, topic: str) -> List[str]:
    """Get reasons for the score"""
    reasons = []
    
    length = len(title)
    platform_limit = PLAT_LIMITS.get(platform, PLAT_LIMITS["default"])
    
    if length <= platform_limit:
        reasons.append("good-length")
    
    if topic.lower() in title.lower():
        reasons.append("topic-relevant")
    
    if title.endswith('?'):
        reasons.append("question")
    
    if re.search(r'\b\d+%?\b', title):
        reasons.append("has-number")
    
    if re.search(r'\b(you|your|you\'re)\b', title, re.IGNORECASE):
        reasons.append("second-person")
    
    return reasons