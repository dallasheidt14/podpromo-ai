# backend/services/title_service.py
# DEPRECATES old ad-hoc heuristics. Single source of truth for titles.

from __future__ import annotations
import re, math, itertools, hashlib, time
from collections import Counter
from typing import List, Dict, Iterable, Tuple, Optional, Set

# ---------- Title caching & deduplication ----------
TITLE_GEN_VERSION = "v2"
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

# ---------- Text utils ----------
from services.title_filters import clean_for_keywords, fix_or_reject_title, dedup_titles, get_anchor_set

def _gate_titles(cands, clip_text):
    """Apply filtering to all title candidates"""
    okd = []
    for raw in cands:
        ok, norm, reason = fix_or_reject_title(raw, clip_text)
        if ok:
            okd.append(_titlecase(norm))
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("TITLE_REJECT: reason=%s title=%r", reason, raw)
    return dedup_titles(okd)

SMALL_WORDS = {"a","an","the","and","or","but","for","nor","on","in","to","at","by","with","of"}

def _titlecase(s: str) -> str:
    """Apply proper title case to strings"""
    w = s.split()
    return " ".join((t if i and t in SMALL_WORDS else t.capitalize()) for i, t in enumerate(w))

# Bad anchors to avoid (stopwords + weak tokens)
BAD_ANCHORS = {
    "a","an","the","and","or","but","if","then","so","than","to","of","in","on","at","by","for","from","with","about",
    "as","into","onto","over","under","up","down","out","off","again","still","very",
    "i","i'm","you","you're","we","we're","they","they're","he","she","it","its","it's","that","this","these","those",
    "who","whom","whose","which","what","when","where","why","how",
    "be","am","is","are","was","were","been","being","do","does","did","doing","have","has","had","having",
    # common "bad anchors" seen in logs:
    "name","into","which","public","open","moved","basis","feel","site","been","mega",
    # additional weak tokens from recent logs:
    "new","sense","solar"
}

# Can't stand alone as anchors (fine in phrases, not as whole anchor)
CANT_STAND_ALONE = {
    "market","solar","sense","new","open","public","daily","basis","spanish"
}

def is_bad_anchor(tok: str) -> bool:
    """Check if a token is a bad anchor (stopword, too short, or weak)"""
    t = tok.strip(" -–—:,.").lower()
    return (not t.isalpha()) or (len(t) < 3) or (t in BAD_ANCHORS) or (t in CANT_STAND_ALONE)

def looks_like_name(kw: str) -> bool:
    """Check if a keyword looks like a proper name"""
    tokens = kw.strip().split()
    if len(tokens) >= 2 and any(t.lower() in NAME_LINKERS for t in tokens):
        return True
    cap_runs = sum(t[:1].isupper() for t in tokens)
    return cap_runs >= 2

# Domain-specific noun hints for better anchor selection
DOMAIN_NOUN_HINTS = {
    "ai","model","llm","prompt","agent","workflow","pipeline","initiative","roles",
    "voice","articles","blogs","brand","market","distribution","channel","message",
    "feedback","leadership","team","culture","accountability","meeting","strategy",
    "process","system","revenue","growth","audience","platform","search","algorithm"
}

# Track used anchors per episode to avoid repetition
USED_ANCHORS = set()

# Content word detection patterns
CONTENT_WORD_RE = re.compile(
    r"\b(ai|model|llm|agent|workflow|pipeline|initiative|role|strategy|plan|method|framework|team|market|brand|channel|"
    r"distribution|project|delivery|risk|budget|schedule|contract|finance|revenue|growth|customers?)\b",
    re.I
)

# Rich bigrams that read well as objects
NOUN_PHRASE_RE = re.compile(
    r"\b(public[-\s]private partnership|project delivery|risk management|project finance|market entry|talent strategy|"
    r"go[-\s]to[-\s]market|infrastructure project|transit project)\b",
    re.I
)

# Phrase normalizers - convert common patterns to safe domain anchors
PHRASE_NORMALIZERS = [
    # Markets
    (re.compile(r"\b(spanish|latam|us|uk|eu|china|india)\s+markets?\b", re.I), "Market Expansion"),
    (re.compile(r"\bexpand(ing|ed)?\s+into\s+new\s+markets?\b", re.I),         "Market Expansion"),
    (re.compile(r"\bgo[-\s]?to[-\s]?market\b", re.I),                          "Go-To-Market"),
    # Project-y
    (re.compile(r"\bproject\s+makes\s+sense\b", re.I),                          "Project"),
    (re.compile(r"\binfrastructure|transit\s+project\b", re.I),                 "Project"),
    # Generic idioms → Strategy
    (re.compile(r"\bmakes\s+sense\b", re.I),                                    "Strategy"),
]

# Name linkers for proper noun detection
NAME_LINKERS = {"del","de","van","von","da","di","la","le"}

# Content nouns that work well as anchors
CONTENT_NOUNS = {"strategy","framework","method","plan","pipeline","risk management",
                 "project","project delivery","project finance","role","team","hiring"}

def candidate_anchors(clip_text: str, episode_vocab: dict = None) -> List[str]:
    """Extract candidate anchor words from clip text, preferring domain terms and rare words"""
    toks = [t for t in clean_for_keywords(clip_text).split() if t.isalpha()]
    episode_vocab = episode_vocab or {}
    
    # crude noun-ish filter: length + not a stopword + not ending in ly/ing
    cand = [t for t in toks if (len(t) > 3 and t not in STOP and 
                               not t.endswith(("ly", "ing")))]
    
    # prefer domain hints and words rarer in the episode
    return sorted(set(cand), key=lambda t: (
        t in DOMAIN_NOUN_HINTS,
        -episode_vocab.get(t, 0),   # lower episode freq first
        -len(t)
    ), reverse=True)

def canonicalize(anchor: str) -> str | None:
    """Canonicalize anchor to safe domain terms"""
    a = anchor.strip(" -–—:,.").lower()
    if a in {"market"}:            return "Market Expansion"
    if a in {"sense","new"}:       return None
    if a in {"solar"}:             return None
    return anchor.title()

def choose_anchor(text: str, kw_candidates: list[str]) -> str:
    """Choose a safe anchor with normalization and guards"""
    s = text or ""

    # 0) Phrase normalizers
    for rx, canon in PHRASE_NORMALIZERS:
        if rx.search(s):
            return canon

    # 1) Prefer good phrases/nouns from text
    for n in CONTENT_NOUNS:
        if re.search(rf"\b{re.escape(n)}\b", s, re.I):
            return n.title()

    # 2) Mine KW list safely
    for kw in kw_candidates or []:
        if looks_like_name(kw):
            continue
        # head noun: take last *meaningful* token
        tail = kw.split()[-1].strip(" -–—:,.").lower()
        if tail in BAD_ANCHORS or tail in CANT_STAND_ALONE:
            # try a canonical mapping (e.g., market -> Market Expansion)
            mapped = canonicalize(tail)
            if mapped:
                return mapped
            continue
        c = canonicalize(tail)
        if c:
            return c

    # 3) Last-resort default
    return "Strategy"

def pick_anchor(clip_text: str, episode_vocab: dict = None) -> str:
    """Pick the best anchor word for this clip, avoiding repetition within episode"""
    for t in candidate_anchors(clip_text, episode_vocab):
        if t not in USED_ANCHORS and not is_bad_anchor(t):
            USED_ANCHORS.add(t)
            return t
    return "plan"  # safe fallback

def title_from_anchor(anchor: str) -> str:
    """Build a human title from a safe anchor"""
    a = anchor.title()
    if any(k in a.lower() for k in ["strategy","framework","method","plan","pipeline","risk","project"]):
        return f"What No One Tells You About {a}"
    return f"Do This Before Your Next {a}"

def finalize_title(clip_text: str, kw: list[str]) -> str:
    """Finalize title with comprehensive safety checks"""
    anchor = choose_anchor(clip_text, kw)
    title = title_from_anchor(anchor)
    m = BAD_TITLE_RE.search(title)
    if m and (m.group(1).lower() in BAD_ANCHORS | CANT_STAND_ALONE):
        return title_from_anchor("Strategy")
    return title

# Bad title pattern detection
BAD_TITLE_RE = re.compile(r"#\s*1\s*Reason\s+(\w+)\s+Fails", re.I)

def post_validate(title: str, clip_text: str, kw_candidates: list[str] = None) -> str:
    """Final guard: if a template accidentally produces nonsense, redo"""
    m = BAD_TITLE_RE.search(title)
    if m and is_bad_anchor(m.group(1)):
        anchor = choose_anchor(clip_text, kw_candidates) or "Strategy"
        return title_from_anchor(anchor)
    return title

def fallback_title_from_clip(clip_text: str, episode_vocab: dict = None, kw_candidates: list[str] = None) -> str:
    """Generate a fallback title using smart anchor selection"""
    return finalize_title(clip_text, kw_candidates or [])

STOP = {
    "the","a","an","and","or","but","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","as","that","this","it","you","your","yours","we","our","ours",
    "i","me","my","they","them","their","he","she","his","her","us","do","did","does","done",
    "have","has","had","if","so","just","very","really","can","could","should","would",
    # conversational fillers
    "thank","thanks","sure","about","today","now","really","just","maybe","kind","sort",
    "little","bit","like","well","ok","okay","yeah","yep","right","so","uh","um",
    # too-generic business nouns we don't want as the *object*
    "company","client","people","thing","things","work","time","job","team"
}

ROLE_WORDS = {
    "leader","leaders","manager","managers","exec","executive","executives",
    "founder","founders","creator","creators","coach","coaches","team","teams","sales","product"
}

CURIOSITY = {"mistake","secret","trap","myth","warning","why","before","avoid","fix","rule","truth"}
IMPERATIVE_STARTERS = {"stop","start","avoid","fix","remember","do","don't","never"}

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z'\-]+|\d+%?", text)]

def _is_too_short(text: str) -> bool:
    toks = [t for t in _tokens(text) if t.isalpha() and t not in STOP]
    return len(toks) < 6  # short/empty clip guard

def _ngrams(words: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(words)-n+1):
        yield tuple(words[i:i+n])

BAD_SINGLE = {
    "these","this","that","more","less","first","second","third","where","when","why","how","what","who","which",
    "and","or","the","a","an","to","of","in","on","for","with","your"
}
VALID_TOPIC_RE = re.compile(r"^[A-Za-z][A-Za-z\- ]{2,}$")

def _contextual_topic(text: str) -> str | None:
    """Extract high-confidence contextual topics from text"""
    # High-confidence multi-grams first
    m = re.search(r"\b(positive|corrective)\s+feedback\b", text, re.I)
    if m: return f"{m.group(1).title()} Feedback"
    if re.search(r"\bfeedback\s+sandwich\b", text, re.I):
        return "The Feedback Sandwich"
    if re.search(r"\b(one[-\s]?on[-\s]?ones?|1[:\- ]?1s?)\b", text, re.I):
        return "1:1s"
    if re.search(r"\bfeedback\b", text, re.I):
        return "Feedback"
    if re.search(r"\byour?\s+team\b", text, re.I):
        return "Your Team"
    return None

def _ngrams(tokens, n):
    """Generate n-grams from token list"""
    for i in range(len(tokens)-n+1):
        yield tokens[i:i+n]

def _candidate_phrases(text: str, max_k: int = 12) -> list[str]:
    """Extract candidate phrases using improved scoring with anchor detection"""
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text)
    anchors = get_anchor_set()
    
    def score_ngram(ng):
        phrase = " ".join(ng)
        low = phrase.lower()
        
        # Check for stopwords
        if any(w.lower() in STOP for w in ng): return 0.0
        
        # Check for hard-banned patterns (pronouns/questions at edges)
        if len(ng) == 1:
            if ng[0].lower() in {"i","i'm","you","you're","we","we're","they","they're","he","she","it","this","that","these","those","why","what","how","when"}: return 0.0
        else:
            head, tail = ng[0].lower(), ng[-1].lower()
            if head in {"i","i'm","you","you're","we","we're","they","they're","he","she","it","this","that","these","those","why","what","how","when"}: return 0.0
            if tail in {"i","i'm","you","you're","we","we're","they","they're","he","she","it","this","that","these","those","why","what","how","when","of","to","for","with","on","in","at","by","about"}: return 0.0
        
        # Base score by length (prefer 2-3 grams)
        L = len(ng)
        base = {1: 0.6, 2: 1.6, 3: 1.8}.get(L, 0.0)
        
        # Anchor bonus if phrase ends with a useful nouny term
        end = ng[-1].lower()
        bonus = 0.5 if end in anchors else 0.0
        
        # Specificity bonus for caps/hyphens
        spec = 0.2 if any("-" in t or re.match(r"[A-Z]", t) for t in ng) else 0.0
        
        # Discourage 1-gram generic nouns
        if L == 1 and len(ng[0]) < 5: base -= 0.4
        
        return base + bonus + spec
    
    cands = []
    for n in (3,2,1):
        for i in range(len(toks)-n+1):
            ng = toks[i:i+n]
            s = score_ngram(ng)
            if s > 0:
                cands.append((" ".join(ng), s))
    
    # de-dup on lowercase, keep highest score
    best = {}
    for p, s in cands:
        k = p.lower()
        if s > best.get(k, -1): best[k] = s
    
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [p for p,_ in ranked[:max_k]]

BAD_HEADS = {"thank","sure","about","client","company","vertical","complement","eyes"}

def _pick_object_phrase(clean: str) -> str | None:
    """Pick the best object phrase for title templates"""
    anchors = get_anchor_set()
    for ph in _candidate_phrases(clean):
        # final guard: don't allow single-word or vague tails unless anchored
        tail = ph.split()[-1].lower()
        if (ph.count(" ") == 0) and (tail not in anchors):  # e.g., "company"
            continue
        return ph
    return None

def _clean_slot(raw: str) -> str:
    """Clean and sanitize slot to avoid junky words"""
    s = raw.strip(" ,.!?:;\"'").lower()
    if not s or s in BAD_SINGLE: 
        return ""
    if not VALID_TOPIC_RE.match(s):
        return ""
    # Title-case but keep common acronyms simple
    s = " ".join(w.capitalize() if len(w) > 1 else w.upper() for w in s.split())
    # Common fixes
    s = re.sub(r"\bTeam\b$", "Your Team", s)  # 'team' → 'Your Team'
    s = re.sub(r"\bGiving\b$", "Feedback", s) # 'giving' (alone) → 'Feedback'
    return s

def _is_decision_context(text: str) -> bool:
    """Check if text contains decision/choice language"""
    return bool(re.search(r"\b(decide|decision|choose|pick|select|before you)\b", text, re.I))

def _best_phrases(text: str, max_phrases: int = 6) -> List[str]:
    # extract meaningful phrases from text
    words = [w for w in _tokens(text) if w.isalpha() and w not in STOP]
    
    # Look for meaningful 2-3 word phrases first
    good_phrases = []
    
    # Find noun phrases and meaningful combinations
    meaningful_words = [w for w in words if len(w) > 3 and w not in STOP]
    
    # Look for common business/leadership terms
    business_terms = ["problem", "problems", "challenge", "challenges", "solution", "solutions", 
                     "decision", "decisions", "strategy", "strategies", "team", "teams", 
                     "leader", "leaders", "organization", "front", "line", "effort", "efforts"]
    
    # Prioritize business terms
    for term in business_terms:
        if term in words:
            good_phrases.append(term)
    
    # Add other meaningful words
    for word in meaningful_words[:5]:
        if word not in good_phrases:
            good_phrases.append(word)
    
    # Create 2-word combinations from good phrases
    for i in range(len(good_phrases)-1):
        combo = f"{good_phrases[i]} {good_phrases[i+1]}"
        if len(combo) < 25:  # reasonable length
            good_phrases.append(combo)
    
    # Fallback to simple extraction if nothing good found
    if not good_phrases:
        counts = Counter(words)
        good_phrases = [w for w, c in counts.most_common(5) if len(w) > 3]
    
    # Clean and dedupe, filtering out WH-words
    dedup, seen = [], set()
    for p in good_phrases[:max_phrases]:
        cleaned = _clean_slot(p)
        if cleaned:
            key = re.sub(r"[^a-z0-9]+","",cleaned.lower())
            if key and key not in seen and len(cleaned) > 2:
                seen.add(key)
                dedup.append(cleaned)
    
    return dedup or ["the real problem","the big mistake","status quo"]

def _find_roles(text: str) -> List[str]:
    toks = set(_tokens(text))
    roles = [r for r in ROLE_WORDS if r in toks]
    # pluralize if needed
    fixed = []
    for r in roles[:3]:
        if not r.endswith("s"):
            if r == "sales": fixed.append(r)
            else: fixed.append(r + "s")
        else:
            fixed.append(r)
    return fixed or ["leaders"]

def _find_ing(text: str) -> List[str]:
    ings = [t for t in _tokens(text) if t.endswith("ing") and t not in STOP and t.isalpha()]
    # prefer decision verbs etc.
    P = ("deciding","hiring","firing","planning","scaling","selling","launching","building","meeting","budgeting")
    ings = sorted(set(ings), key=lambda w: (w in P, len(w)), reverse=True)
    return (ings or ["deciding","hiring","planning"])[:3]

def _find_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d{1,3}%\b|\b\d{1,4}\b", text)

def _dedup_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _sentence_case(s: str) -> str:
    s = s.strip()
    if not s: return s
    return s[0].upper() + s[1:]

def _final_fixups(title: str) -> str:
    """Apply final cleanup to prevent junky artifacts"""
    t = re.sub(r"\bYour\s+(These|This|That|More|Less|First|Second)\b", r"\1", title, flags=re.I)
    t = re.sub(r"\bon\s+(where|when|why|how|what|who|which)\b$", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip(" -:,.")
    return t

def _clarity_penalty(title: str) -> float:
    """Penalize titles with junky words"""
    if re.search(r"\b(These|This|That|More|Less|First|Second|Where)\b", title, re.I):
        return 0.10
    return 0.0

def _looks_nonsense(s: str) -> bool:
    """Check if title looks like nonsense due to bad phrase extraction"""
    # Use the new filter system for comprehensive nonsense detection
    ok, _, reason = fix_or_reject_title(s)
    return not ok

# ---------- Candidate builders ----------
def _build_candidates(text: str, obj_phrase: str = None) -> List[str]:
    phrases = _best_phrases(text)
    roles = _find_roles(text)
    ings = _find_ing(text)
    nums = _find_numbers(text)
    percent = next((n for n in nums if n.endswith("%")), None)
    top = phrases[0] if phrases else "Problem"
    second = phrases[1] if len(phrases) > 1 else top

    cand: List[str] = []

    # Content-aware families first
    if re.search(r"\bfeedback\s+sandwich\b", text, re.I):
        cand += [
            "Stop Using the Feedback Sandwich",
            "The Feedback Sandwich Doesn't Work—Here's Why",
            "Why the Feedback Sandwich Fails Your Team",
        ]

    if re.search(r"\bfeedback\b", text, re.I):
        cand += [
            f"The #1 Reason { _sentence_case(top) } Fails",
            f"Stop Guessing—Fix { _sentence_case(top) } This Way",
            f"How to Give { _sentence_case(top) } That Lands",
            f"Your Team Needs Clear, Fast { _sentence_case(top) }",
        ]

    # Use object phrase if available, otherwise fall back to topic
    obj = obj_phrase if obj_phrase else top

    # Gate the "decide" family
    if _is_decision_context(text):
        if obj_phrase:
            cand += [
                f"Ask This Before You Decide on { _sentence_case(obj) }",
                f"Before You Decide, Do This",
            ]
        else:
            cand += [
                "Ask This Before You Decide",
                "Before You Decide, Do This",
            ]

    # Object-based templates (only if we have a good object phrase)
    if obj_phrase:
        cand += [
            f"The Hidden Trap in { _sentence_case(obj) }",
            f"Most { _sentence_case(roles[0]) } Miss This About { _sentence_case(obj) }",
            f"Don't Make This { _sentence_case(obj) } Mistake",
            f"The #1 Reason { _sentence_case(obj) } Fails",
            f"Are You Solving the Wrong { _sentence_case(obj) }?",
            f"What If { _sentence_case(obj) } Isn't the Real Problem?",
            f"You're Closer Than You Think—Fix { _sentence_case(obj) }",
            f"The Shortcut Everyone Ignores on { _sentence_case(obj) }",
        ]
    else:
        # Fallback to object-less templates
        cand += [
            "What No One Tells You About This",
            "The Hidden Trap Most Leaders Miss",
            "Don't Make This Common Mistake",
            "The #1 Reason Most Strategies Fail",
        ]

    # Number/percent hooks
    if percent:
        cand += [
            f"{percent} of { _sentence_case(roles[0]) } Get { _sentence_case(obj) } Wrong",
            f"Why {percent} Keep Missing { _sentence_case(obj) }",
        ]

    # "Before you …" (only if decision context)
    if _is_decision_context(text):
        cand += [
            f"Before You Start { _sentence_case(ings[0]) }, Do This",
            f"Before Your Next { _sentence_case(ings[0]) }, Do This",
        ]

    # Role-targeted
    cand += [
        f"{ _sentence_case(roles[0]) }: Do This Before Your Next Decision",
        f"{ _sentence_case(roles[0]) }: The Simple Fix for { _sentence_case(obj) }",
    ]

    # Variety with second phrase
    if len(phrases) > 1:
        cand += [
            f"Break the { _sentence_case(second) } Loop—Here's How",
            f"The Playbook to Beat { _sentence_case(second) }",
        ]

    # Normalize whitespace and dedupe
    seen, out = set(), []
    for t in cand:
        t = re.sub(r"\s+", " ", t).strip()
        k = _dedup_key(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out

# ---------- Scoring ----------
def _score(title: str, *, platform: str, topic_hint: str, text: str, avoid: Set[str]) -> Tuple[float, List[str]]:
    reasons = []
    lim = PLAT_LIMITS.get(platform, PLAT_LIMITS["default"])
    L = len(title)

    # base
    score = 1.0

    # features
    if re.search(r"\b\d+%?\b", title): score += 0.35; reasons.append("number/percent")
    if title.endswith("?"): score += 0.20; reasons.append("question")
    if any(title.lower().startswith(w+" ") for w in IMPERATIVE_STARTERS): score += 0.15; reasons.append("imperative")
    if any(w in title.lower() for w in ("you ","your ","you—","you're","you've","you'll")):
        score += 0.25; reasons.append("second-person")
    if any(w in title.lower() for w in CURIOSITY):
        score += 0.25; reasons.append("curiosity")

    # topic relevance
    if re.search(rf"\b{re.escape(topic_hint)}\b", title, flags=re.I):
        score += 0.20; reasons.append("topic-match")

    # platform length penalty (soft)
    if L > lim:
        over = L - lim
        score -= min(0.60, 0.015 * over); reasons.append(f"length-{L}/{lim}")

    # short is usually good; too short can be bland
    if L < 24:
        score -= 0.10; reasons.append("very-short")
    elif L <= 64:
        score += 0.10; reasons.append("tight-length")

    # duplicate penalty vs avoid set
    key = _dedup_key(title)
    if key in avoid:
        score -= 2.5; reasons.append("duplicate-episode")

    return (round(score, 3), reasons)

# ---------- Public API ----------
def build_episode_vocab(episode_text: str) -> dict:
    """Build vocabulary frequency map for an episode to help with anchor selection"""
    toks = clean_for_keywords(episode_text).split()
    vocab = {}
    for t in toks:
        if t.isalpha() and len(t) > 3:
            vocab[t] = vocab.get(t, 0) + 1
    return vocab

def generate_titles(
    text: str,
    *,
    platform: Optional[str] = None,
    n: int = 6,
    avoid_titles: Optional[Iterable[str]] = None,
    episode_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    episode_vocab: Optional[dict] = None,
) -> List[Dict[str, object]]:
    # Clean text before processing to remove conversational fillers
    clean = clean_for_keywords(text)
    text = _clean(text)
    plat = normalize_platform(platform)
    avoid_keys = {_dedup_key(t) for t in (avoid_titles or []) if t}
    
    # Reset cache on new episode to prevent bleed
    if episode_id:
        maybe_reset_cache(episode_id)
    
    # Check cache if we have all required parameters (use cleaned text for cache key)
    if episode_id and clip_id and platform:
        cache_key = _title_cache_key(episode_id=episode_id, clip_id=clip_id, platform=plat, text=clean)
        cached_titles = _get_cached_titles(cache_key)
        if cached_titles:
            # Apply avoid_titles filter to cached results
            filtered_titles = [t for t in cached_titles if _dedup_key(t["title"]) not in avoid_keys]
            if filtered_titles:
                # Debug: Log cache hit to verify it's working
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"TITLE_CACHE_HIT: episode={episode_id}, clip={clip_id}, text_len={len(clean)}")
                return filtered_titles[:n]
    
    if not text or _is_too_short(text):
        # Generate contextual fallbacks based on available text
        fallbacks = []
        
        # Try to extract meaningful words from the text
        if text and len(text.strip()) > 10:
            words = re.findall(r'\b\w+\b', text.lower())
            if words:
                # Use the most common meaningful words
                word_counts = Counter(words)
                common_words = [w for w, c in word_counts.most_common(3) if len(w) > 3 and w not in {'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'where', 'much', 'some', 'very', 'when', 'here', 'just', 'into', 'than', 'only', 'over', 'think', 'know', 'take', 'come', 'made', 'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull'}]
                
                if common_words:
                    # Create contextual fallbacks using the actual content
                    if len(common_words) >= 2:
                        fallbacks.append(f"The {common_words[0].title()} {common_words[1].title()} Strategy")
                        fallbacks.append(f"Why {common_words[0].title()} Matters More Than You Think")
                    if len(common_words) >= 1:
                        fallbacks.append(f"The {common_words[0].title()} Method")
                        fallbacks.append(f"Master {common_words[0].title()} in Minutes")
        
        # If no contextual fallbacks, use generic ones
        if not fallbacks:
            fallbacks = [
                "The Strategy That Changes Everything",
                "What Most People Get Wrong",
                "The Method That Actually Works",
            ]
        
        out = []
        for t in fallbacks:
            sc, rs = _score(t, platform=plat, topic_hint="strategy", text=text, avoid=avoid_keys)
            out.append({"title": t, "score": sc, "reasons": rs})
        return sorted(out, key=lambda x: x["score"], reverse=True)[:max(1, n)]

    # Use new phrase picker on cleaned text
    obj_phrase = _pick_object_phrase(clean)
    
    # Log keyword extraction for debugging
    import logging
    logger = logging.getLogger(__name__)
    if obj_phrase:
        candidates = _candidate_phrases(clean, max_k=5)
        logger.info(f"TITLE_KW: top={candidates[:3]} chosen='{obj_phrase}'")
    else:
        logger.info(f"TITLE_KW_SKIP: no good phrases found in cleaned text")
    
    # Use contextual topic mapping first, then object phrase, then clean slot
    topic_hint = _contextual_topic(clean) or (obj_phrase if obj_phrase else None) or "Feedback"
    cands = _build_candidates(text, obj_phrase)

    # score + rank + ensure uniqueness
    scored = []
    for t in cands:
        sc, rs = _score(t, platform=plat, topic_hint=topic_hint, text=text, avoid=avoid_keys)
        scored.append({"title": t, "score": sc, "reasons": rs})

    # sort by score, then prefer diversity by different starters
    scored.sort(key=lambda x: x["score"], reverse=True)

    picked, starters = [], set()
    for item in scored:
        start = item["title"].split(" ", 1)[0].lower()
        if start in starters and len(picked) < n//2:
            # defer duplicates starters early to diversify hooks
            continue
        picked.append(item)
        starters.add(start)
        if len(picked) >= n:
            break

    # if we over-diversified and came short, top up
    if len(picked) < n:
        for item in scored:
            if item not in picked:
                picked.append(item)
                if len(picked) >= n:
                    break

    # clamp reasons length & round scores
    for p in picked:
        p["score"] = float(p["score"])
        p["reasons"] = p.get("reasons", [])[:4]
        
        # Apply final fixups to prevent junky artifacts
        p["title"] = _final_fixups(p["title"])

    # Apply deduplication to final results
    seen_titles = set()
    deduped_picked = []
    for v in picked:
        title_key = v["title"].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            deduped_picked.append(v)
    picked = deduped_picked
    
    # Apply comprehensive gating to all titles
    raw_titles = [v["title"] for v in picked]
    gated_titles = _gate_titles(raw_titles, clean)
    
    # Rebuild picked list with gated titles, preserving scores and reasons
    picked = [v for v in picked if v["title"] in gated_titles]
    
    # Fallback if all titles were filtered out
    if not picked:
        # Get keyword candidates for better anchor selection
        kw_candidates = _candidate_phrases(clean, max_k=5)
        
        # Generate fallback using smart anchor selection
        fb_title = fallback_title_from_clip(clean, episode_vocab, kw_candidates)
        final_fallbacks = _gate_titles([fb_title], clean)
        if not final_fallbacks:
            final_fallbacks = _gate_titles(["Do This Before Your Next Decision"], clean)
        picked = [{"title": final_fallbacks[0], "score": 0.52, "reasons": ["fallback_generic"]}]
    
    # Pick the best variant using clarity penalty
    if picked:
        best = max(picked, key=lambda v: v["score"] - _clarity_penalty(v["title"]))
        # Move best to front
        picked.remove(best)
        picked.insert(0, best)
    
    # Cache results if we have all required parameters (use cleaned text for cache key)
    if episode_id and clip_id and platform:
        cache_key = _title_cache_key(episode_id=episode_id, clip_id=clip_id, platform=plat, text=clean)
        _cache_titles(cache_key, picked)
        # Debug: Log cache miss to verify fresh generation
        import logging
        logger = logging.getLogger(__name__)
        text_sha = hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:12]
        logger.info(f"TITLE_GEN: episode={episode_id}, clip={clip_id}, platform={plat}, text_sha={text_sha} -> '{picked[0]['title'] if picked else 'NONE'}'")

    return picked
