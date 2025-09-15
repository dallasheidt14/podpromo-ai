# backend/services/title_service.py
# Improved title generation with normalization, relaxed bans, dedup, and fallbacks

from __future__ import annotations
import re, math, itertools, hashlib, time
from collections import Counter
from typing import List, Dict, Iterable, Tuple, Optional, Set
import logging

logger = logging.getLogger("titles_service")

# Minimal English stopwords (expand if needed)
STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","from","as","at","by","is","are","was","were",
    "be","been","being","it","its","this","that","these","those","you","your","we","our","they","their","i",
    "he","she","him","her","his","hers","them","do","does","did","doing","have","has","had","having","not",
    "no","but","so","if","then","than","there","here","what","which","who","whom","into","over","under","about",
    # extra fillers that polluted titles
    "through","again","maybe","say","will","ill","youre","very","answer","two","words"
}

_STOP = set("a an the and or for to in of with over under on at is are was were be been being it this that".split())

CONTRACTION_FIX = {
    r"\bweve\b": "we've",
    r"\btheyre\b": "they're",
    r"\btheres\b": "there's",
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

BANNED_PHRASES = {"Key Insight", "Key Takeaways", "Inside ", "What It Means"}

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
    # Tokens with original case for simple "proper noun" detection
    toks = re.findall(r"[A-Za-z][A-Za-z'-]*", clean)
    lowers = [t.lower() for t in toks]

    phrases: Counter = Counter()

    # Hyphenated terms are already good candidates (e.g., off-label)
    for h in re.findall(r"[A-Za-z]+-[A-Za-z]+", clean):
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
                if not BANNED.search(phrase) and not all(w in STOP for w in gram) and not BAN_PHRASE.search(phrase):
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

def _extract_noun_phrases(text: str) -> list:
    """Extract top TF-IDF noun chunks for better fallback titles"""
    if not text:
        return []
    
    # Simple noun phrase extraction (enhanced version)
    words = [w.strip(".,!?\"'():;").lower() for w in text.split()]
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

def _title_fallback(text: str, platform: str) -> str:
    """Content-aware title fallback with rotation to avoid repetition"""
    if not text:
        return "The most overlooked thing about this topic"
    
    # Extract content elements for pattern matching
    noun_phrases = _extract_noun_phrases(text)
    
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
    t = text.lower()
    
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
    Phrase-first extraction (anchors + hyphens + short ngrams).
    Produces human-friendly phrases like "Salem Restructuring", "Digital-First Company".
    """
    phrases = _mine_anchor_phrases(text)
    if phrases:
        # normalize hyphen case (Digital-First), filter banter
        cleaned = []
        for p in phrases:
            if BAN_PHRASE.search(p):
                continue
            cleaned.append(_title_case_hyphen(p))
        return cleaned[:k]
    # fallback to unigram frequency if anchors fail
    clean = _clean_text(text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", clean)
    freq = Counter(w.lower() for w in words if w.lower() not in STOP)
    out = []
    for w, _ in freq.most_common(20):
        t = w.title()
        if BAN_PHRASE.search(t):
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out

def _title_case(s: str) -> str:
    return " ".join([w.capitalize() if w.lower() not in STOP else w.lower() for w in s.split()])

def _variants_from_keywords(keys: List[str], platform: str) -> List[str]:
    # assemble 6 concise variants; prefer 4–8 words; title-case; avoid banned phrases
    main = " ".join([k.replace("-", " ") for k in keys[:1]]).strip()  # use top phrase only
    if not main:
        return []
    main_tc = _title_case_hyphen(main)
    alts = []
    if platform == "shorts":
        alts = [
            f"{main_tc}: What It Really Means",
            f"{main_tc} Explained",
            f"Why {main_tc} Matters",
            f"{main_tc}: The Real Story",
            f"{main_tc} — Fast Facts",
            f"{main_tc} In Plain English",
        ]
    else:
        alts = [
            f"{main_tc}: What It Means",
            f"{main_tc} Explained",
            f"Inside {main_tc}",
            f"Why {main_tc} Matters",
            f"{main_tc}: Key Takeaways",
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
    tail = "Explained" if platform == "shorts" else "What It Means"
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
PLAT_LIMITS = {
    "tiktok": 80,
    "reels": 80,
    "shorts": 80,
    "youtube": 100,
    "default": 90,
}

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

# ---------- Title generation templates ----------
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
    
    # Generic fallbacks
    "The Strategy That Changes Everything",
    "What Most People Get Wrong",
    "The Method That Actually Works",
]

def extract_topic(text: str) -> str:
    """Extract the main topic from text using improved heuristics"""
    if not text:
        return "Strategy"
    
    # Get key terms using improved extraction
    terms = key_terms(text, max_terms=6)
    
    if not terms:
        return "Strategy"
    
    # Look for business/leadership terms
    business_terms = [
        'strategy', 'leadership', 'team', 'feedback', 'decision', 'problem', 'solution',
        'management', 'communication', 'culture', 'growth', 'innovation', 'change',
        'planning', 'execution', 'performance', 'collaboration', 'trust', 'vision',
        'practice', 'training', 'coaching', 'development', 'improvement', 'skill'
    ]
    
    # Find the most relevant business term
    for term in business_terms:
        if term in [t.lower() for t in terms]:
            return term.title()
    
    # Use the first key term as topic
    return terms[0].title()

def make_titles(text: str) -> List[str]:
    """Generate titles using improved heuristics with real text"""
    if not text:
        return ["Quick Tip", "Coach's Insight", "One Thing Most Players Miss"]
    
    terms = key_terms(text, max_terms=4)
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

# Ad detection markers
AD_MARKERS = {"sponsored", "brought to you by", "promo code", "use code", "limited time",
              "shop now", "visit", "link in bio", "terms apply", "free trial", "subscribe",
              "Wayfair", "NordVPN", "Squarespace", "Raid Shadow Legends"}

def _looks_like_ad(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in AD_MARKERS)

def generate_titles(
    text: str,
    *,
    platform: Optional[str] = None,
    n: int = 4,
    avoid_titles: Optional[Iterable[str]] = None,
    episode_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    episode_vocab: Optional[dict] = None,
) -> List[Dict[str, object]]:
    """
    Deterministic, topic-aware titles from the actual clip text.
    Avoids generic/banned clickbait; returns 6 or a safe fallback.
    """
    
    # hard-drop obvious ads from titling
    if _looks_like_ad(text):
        return []  # no titles for ads; upstream should already de-prefer these
    
    # Normalize inputs
    clean_text = normalize_text(text)
    platform = normalize_platform(platform)
    avoid_set = set(avoid_titles or [])
    
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
    
    # Extract keywords from actual text
    keys = _extract_keywords(clean_text, k=6)
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
            sanitized_title = sanitized_title.replace(": What It Means", "").replace(": What It Really Means", "")
            # Normalize contractions
            sanitized_title = _normalize_contractions(sanitized_title)
            # Check for banned phrases and rewrite if needed
            for banned in BANNED_PHRASES:
                if banned.lower() in sanitized_title.lower():
                    sanitized_title = sanitized_title.replace(banned, "").strip()
                    if not sanitized_title:
                        sanitized_title = "Important Insight"
            sanitized = sanitized_title != title
            
            result.append({
                "title": sanitized_title,
                "score": 1.0 - (i * 0.1),  # Decreasing score
                "reasons": ["Generated from text analysis"]
            })
            
            if sanitized:
                logger.info("TITLE_SANITIZED=True original='%s' final='%s'", title, sanitized_title)
    
    # Cache the result
    if episode_id and clip_id and platform:
        cache_key = _title_cache_key(episode_id=episode_id, clip_id=clip_id, platform=platform, text=clean_text)
        _cache_titles(cache_key, result)
    
    return result

def score_title(title: str, platform: str, topic: str, text: str) -> float:
    """Score a title based on various factors"""
    score = 1.0
    
    # Length scoring
    length = len(title)
    platform_limit = PLAT_LIMITS.get(platform, PLAT_LIMITS["default"])
    
    if length <= platform_limit:
        score += 0.2
    else:
        # Penalty for being too long
        overage = length - platform_limit
        score -= min(0.5, overage * 0.01)
    
    # Topic relevance
    if topic.lower() in title.lower():
        score += 0.3
    
    # Question titles
    if title.endswith('?'):
        score += 0.2
    
    # Number/percentage
    if re.search(r'\b\d+%?\b', title):
        score += 0.2
    
    # Second person
    if re.search(r'\b(you|your|you\'re)\b', title, re.IGNORECASE):
        score += 0.15
    
    # Action words
    action_words = ['do', 'stop', 'start', 'avoid', 'fix', 'master', 'learn', 'change']
    if any(word in title.lower() for word in action_words):
        score += 0.1
    
    return round(score, 2)

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