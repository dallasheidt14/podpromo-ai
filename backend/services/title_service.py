# backend/services/title_service.py
# Improved title generation with normalization, relaxed bans, dedup, and fallbacks

from __future__ import annotations
import re, math, itertools, hashlib, time
from collections import Counter
from typing import List, Dict, Iterable, Tuple, Optional, Set

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
    """Generate titles with improved normalization and fallbacks"""
    
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
    
    # Extract topic
    topic = extract_topic(clean_text)
    
    # Generate candidate titles using improved heuristics
    candidates = []
    
    # Use improved title generation with real text
    improved_titles = make_titles(clean_text)
    for title in improved_titles:
        if is_valid_title(title) and title not in avoid_set:
            candidates.append(title)
    
    # Use templates with topic substitution as fallback
    for template in TITLE_TEMPLATES:
        if '{topic}' in template:
            title = template.format(topic=topic)
        else:
            title = template
        
        if is_valid_title(title) and title not in avoid_set and title not in candidates:
            candidates.append(title)
    
    # Add some topic-specific variations
    if topic != "Strategy":
        topic_variations = [
            f"The {topic} Method",
            f"Master {topic} in Minutes",
            f"Why {topic} Matters More Than You Think",
            f"The {topic} Approach That Works",
        ]
        for title in topic_variations:
            if is_valid_title(title) and title not in avoid_set and title not in candidates:
                candidates.append(title)
    
    # Score and rank candidates
    scored_candidates = []
    for title in candidates:
        score = score_title(title, platform, topic, clean_text)
        scored_candidates.append({
            "title": fix_title(title),
            "score": score,
            "reasons": get_score_reasons(title, platform, topic)
        })
    
    # Sort by score and deduplicate
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    deduped = dedup_titles(scored_candidates)
    
    # Ensure we have at least 3-4 titles
    if len(deduped) < 3:
        # Add fallback titles
        fallbacks = [
            "The Strategy That Changes Everything",
            "What Most People Get Wrong", 
            "The Method That Actually Works",
            "Do This Before Your Next Decision"
        ]
        
        for fallback in fallbacks:
            if fallback not in [t["title"] for t in deduped]:
                deduped.append({
                    "title": fallback,
                    "score": 0.5,
                    "reasons": ["fallback"]
                })
                if len(deduped) >= 4:
                    break
    
    # Cache results
    if episode_id and clip_id and platform:
        cache_key = _title_cache_key(episode_id=episode_id, clip_id=clip_id, platform=platform, text=clean_text)
        _cache_titles(cache_key, deduped)
    
    return deduped[:n]

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