# backend/services/title_engine_v2.py
# CFD-First Title Engine with Archetype Classification and Template Registry

from __future__ import annotations
import re, math, itertools, hashlib, time
from collections import Counter
from typing import List, Dict, Iterable, Tuple, Optional, Set
import logging

logger = logging.getLogger("title_engine_v2")

# ============================================================================
# ARCHETYPE CLASSIFICATION SYSTEM
# ============================================================================

ARCHETYPES = [
    "Confession/Vulnerability", "Hot Take/Contradiction", "Reveal/Twist", "Insider/Secret",
    "Shock/Outrage", "Relatability/Humor", "Advice/Playbook", "Origin/Backstory",
    "Mistake/Failure/Lesson", "Debate/Argument", "Stats/Proof/Receipts", "Mystery/Cliffhanger"
]

EMOTIONS = ["Curiosity", "Awe", "Anger", "Fear", "Relief", "Humor", "Nostalgia", "FOMO", "Empathy"]

# Cue patterns for archetype detection
CUES = {
    "Confession/Vulnerability": {
        "lex_strong": [
            r"\b(honestly|to be honest|truth is)\b",
            r"\b(i'?ve never told (anyone|this))\b",
            r"\b(i('?m| am) (ashamed|embarrassed|scared|afraid))\b",
            r"\b(i regret(ed)?|i wish i (knew|had)|vulnerable)\b",
        ],
        "lex_weak": [
            r"\b(i'?m not proud|this is hard to say|real talk)\b",
        ],
        "struct": [
            r"^\s*i\b",  # first-person open
        ],
        "disc": [
            r"\b(i did .+? (and|but) (here'?s|this is) (what i learned|the lesson))\b",
        ],
    },

    "Hot Take/Contradiction": {
        "lex_strong": [
            r"\bunpopular opinion\b",
            r"\b(actually|in reality|the truth)\b",
            r"\b(everyone('?s| is) (wrong|lying)|that('?s| is) a (myth|lie))\b",
            r"\b(stop (doing|believing) this|don'?t do this)\b",
        ],
        "lex_weak": [
            r"\b(no one talks about|nobody talks about|people won'?t like this)\b",
        ],
        "struct": [
            r"\b(but|however|instead|except)\b",
        ],
        "disc": [
            r"\b(people think .+? but .+?)\b",
        ],
    },

    "Reveal/Twist": {
        "lex_strong": [
            r"\bturns out\b", r"\bwhat (shocked|surprised) me\b",
            r"\b(then|until) everything changed\b", r"\bi realized\b",
        ],
        "lex_weak": [
            r"\bthe twist\b", r"\bplot twist\b",
        ],
        "struct": [
            r"\b(then|until|after|at first|suddenly)\b",
        ],
        "disc": [
            r"\b(setup .+? → .+? reversal)\b",  # conceptual; you can detect with two-phase markers
        ],
    },

    "Insider/Secret": {
        "lex_strong": [
            r"\bbehind the scenes\b", r"\binsider\b", r"\b(off[- ]camera|off the record)\b",
            r"\bplaybook\b", r"\bhow it (really|actually) works\b", r"\bthe rule every\b",
        ],
        "lex_weak": [
            r"\b(nobody tells you|no one tells you|here'?s what they don'?t say)\b",
        ],
        "struct": [],
        "disc": [],
    },

    "Shock/Outrage": {
        "lex_strong": [
            r"\b(this should be illegal)\b", r"\b(can('?t)? believe|unbelievable)\b",
            r"\b(sc(am|andal)|exposed|they lied|fraud)\b",
        ],
        "lex_weak": [r"!{1,}", r"\b(literally|insane|outrageous|wild)\b"],
        "struct": [r"!{1,}"],
        "disc": [],
    },

    "Relatability/Humor": {
        "lex_strong": [
            r"\b(we (all|both))\b", r"\btell me you .* without telling me\b",
            r"\bwhy are we like this\b",
            r"(😂|🤣|lol|lmao)",
        ],
        "lex_weak": [
            r"\btoo real\b", r"\bthat person\b", r"\bcringe\b", r"\blow[- ]key\b",
        ],
        "struct": [r"\bpresent tense everyday scenario\b"],  # treat as weak
        "disc": [],
    },

    "Advice/Playbook": {
        "lex_strong": [
            r"\b(here('?s| is) how)\b", r"\b(do this|try this)\b",
            r"\b(step[- ]?by[- ]?step|framework|rule|hack|tip|playbook)\b",
            r"\b(\d+\s*(steps|rules|tips|things))\b",
        ],
        "lex_weak": [r"\bguide|primer|checklist\b"],
        "struct": [r"\b(first|then|next|finally)\b"],
        "disc": [r"\b(problem .+? → .+? method .+? → .+? result)\b"],
    },

    "Origin/Backstory": {
        "lex_strong": [
            r"\b(back when|how i (started|got)|day one|first (job|client))\b",
            r"\bin \d{4}\b",
        ],
        "lex_weak": [r"\b(origin story|backstory)\b"],
        "struct": [r"\bpast tense narrative\b"],  # treat as weak
        "disc": [],
    },

    "Mistake/Failure/Lesson": {
        "lex_strong": [
            r"\b(biggest mistake|i messed up|i failed|cost me|never again)\b",
            r"\b(lesson learned|what i learned)\b",
        ],
        "lex_weak": [r"\bshouldn'?t have\b", r"\bthat was dumb\b"],
        "struct": [r"\b(because|so|therefore|ended up)\b"],
        "disc": [],
    },

    "Debate/Argument": {
        "lex_strong": [
            r"\b(i disagree|you('?re| are) wrong|prove it|that('?s| is) not true)\b",
        ],
        "lex_weak": [r"\bchange my mind\b", r"\blet'?s debate\b"],
        "struct": [r"\b\?\s*$"],  # question ending
        "disc": [r"\b(claim .+? ↔ .+? counterclaim)\b"],
    },

    "Stats/Proof/Receipts": {
        "lex_strong": [
            r"\b\d+(\.\d+)?%\b", r"\b1 in \d+\b", r"\bone in (four|five|six|seven|eight|nine|ten|\d+)\b", r"\$\d[\d,]*",
            r"\b(study (shows?|found)|data|according to|source|evidence|proof|receipts?)\b",
        ],
        "lex_weak": [r"\b(by the numbers|on paper)\b"],
        "struct": [r"\b(number .+? → interpretation)\b"],
        "disc": [],
    },

    "Mystery/Cliffhanger": {
        "lex_strong": [
            r"\b(you won('?t)? believe|guess what happened|wait for it|the ending)\b",
            r"\bi can('?t)? say (who|which)\b",
        ],
        "lex_weak": [r"\.\.\.$"],
        "struct": [r"\.\.\.$"],
        "disc": [],
    },
}

# Scoring weights
CUE_WEIGHTS = {
    "lex_strong": 18,
    "lex_weak": 10,
    "struct": 12,
    "disc": 14,
}

BONUSES = {
    "early_position": 8,  # hit in first 8 tokens
    "contrast": 6,  # but/however/instead/except
    "named_entity": 6,  # for Insider/Stats
    "intensifier_shock": 5,  # ! for Shock
    "intensifier_humor": 3,  # ! for Humor
}

NEGATION_PENALTY = -8

# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

TEMPLATES = {
    "Curiosity": [
        "{Stat}: What Everyone Misses About {Topic}",
        "The Counterintuitive Truth About {Topic}",
        "{Topic}: More Common Than You Think",
        "We Tried {Topic} — Here's What Surprised Us",
        "{Topic} Actually Works — Here's Why",
    ],
    "Fear": [
        "Stop {Topic} Mistakes — It's Costing You",
        "The Hidden Risk in {Topic}",
        "Are You Doing {Topic} Wrong?",
        "Don't Fall for This {Topic} Myth",
        "The #1 Reason {Topic} Fails",
    ],
    "Desire": [
        "How to Win With {Topic}",
        "Get Results Faster: {Topic} Playbook",
        "{Topic}: The Simple Way to Get Results",
        "Turn {Problem} Into {Outcome} With {Topic}",
        "{Topic} That Saves You Time & Money",
    ],
    "Confession/Vulnerability": [
        "The Mistake That Changed Everything",
        "What Nobody Warns You About {Topic}",
        "I Wish I Knew This Earlier",
    ],
    "Hot Take/Contradiction": [
        "Unpopular Opinion: {Punchy Claim}",
        "Why Everyone Is Wrong About {Topic}",
        "Actually, {Topic} Doesn't Work Like That",
    ],
    "Reveal/Twist": [
        "It Wasn't {X} After All",
        "Turns Out {Topic} Wasn't the Problem",
        "Here's What Really Happened Next",
    ],
    "Insider/Secret": [
        "The Rule Every {Group} Quietly Follows",
        "Behind the Scenes: How {Topic} Really Works",
        "What They Don't Tell You About {Topic}",
    ],
    "Shock/Outrage": [
        "They Actually Did This?",
        "This Should Be Illegal",
        "How Are They Getting Away With This?",
    ],
    "Relatability/Humor": [
        "We've All Been That Person",
        "Tell Me You're {X} Without Telling Me",
        "Why Is This So True?",
    ],
    "Stats/Proof": [
        "The Data on {Topic} Will Surprise You",
        "{Stat} That Changes How You See {Topic}",
    ],
}

# Archetype to emotion mapping
ARCHETYPE_TRIGGERS = {
    "Hot Take/Contradiction": ["Curiosity", "Anger"],
    "Reveal/Twist": ["Curiosity", "Awe"],
    "Mistake/Failure/Lesson": ["Empathy", "Curiosity"],
    "Advice/Playbook": ["Desire", "Relief"],
    "Shock/Outrage": ["Anger", "Fear"],
    "Stats/Proof/Receipts": ["Curiosity", "Awe"],
    "Confession/Vulnerability": ["Empathy", "Curiosity"],
    "Insider/Secret": ["Curiosity", "Awe"],
    "Relatability/Humor": ["Humor", "Empathy"],
    "Origin/Backstory": ["Nostalgia", "Curiosity"],
    "Debate/Argument": ["Anger", "Curiosity"],
    "Mystery/Cliffhanger": ["Curiosity", "Fear"],
}

# ============================================================================
# PERSONA WEIGHTING SYSTEM
# ============================================================================

PERSONA_WEIGHTS = {
    "Empathy Seekers": {
        "Confession/Vulnerability": 0.9,
        "Mistake/Failure/Lesson": 0.8,
        "Relatability/Humor": 0.6
    },
    "Drama Lovers": {
        "Hot Take/Contradiction": 0.9,
        "Shock/Outrage": 0.8,
        "Reveal/Twist": 0.7
    },
    "Logic Nerds": {
        "Stats/Proof/Receipts": 0.9,
        "Advice/Playbook": 0.8,
        "Hot Take/Contradiction": 0.7
    },
}

# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "as", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those", "you", "your", "we", "our", "they", "their", "i",
    "he", "she", "him", "her", "his", "hers", "them", "do", "does", "did", "doing", "have", "has", "had", "having", "not",
    "no", "but", "so", "if", "then", "than", "there", "here", "what", "which", "who", "whom", "into", "over", "under", "about",
    "through", "again", "maybe", "say", "will", "ill", "youre", "very", "answer", "two", "words"
}

def extract_stat(text: str) -> Optional[str]:
    """Extract statistics from text"""
    patterns = [
        r"\b(1 in \d+|one in (four|five|six|seven|eight|nine|ten|\d+))\b",
        r"\b\d{1,3}%",
        r"\$\d[\d,]*\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(0)
    return None

def cheap_topic(text: str, clip_attrs: Dict = None) -> str:
    """Extract topic using tiered approach"""
    # 1) prefer stable domain phrases if you already compute them
    if clip_attrs:
        for k in ("entity", "domain_topic", "keyword"):
            v = (clip_attrs.get(k) or "").strip()
            if v:
                return v
    
    # 2) look for meaningful noun phrases first (2-3 words)
    compound_patterns = [
        r"\b(off-label|machine learning|artificial intelligence|health care|healthcare|data science)\b",
        r"\b(prescriptions?|medications?|treatments?|therapies?)\b",
        r"\b(doctors?|physicians?|patients?|professionals?)\b",
    ]
    
    for pattern in compound_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[0].title()
    
    # 3) capitalized sequences (but skip single words that are likely not topics)
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    # Filter out common single words that aren't topics
    filtered_matches = [m for m in matches if len(m.split()) > 1 or m.lower() not in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'the', 'this', 'that']]
    if filtered_matches:
        return max(filtered_matches, key=len)
    
    # 4) fallback: most frequent non-stopword 1-2 gram
    tokens = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()) if w not in STOPWORDS]
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    if grams:
        return Counter(grams).most_common(1)[0][0]
    
    return "this"

# ============================================================================
# ARCHETYPE CLASSIFICATION
# ============================================================================

def classify_transcript(text: str) -> Dict:
    """Classify transcript into archetypes and emotions"""
    text_lower = text.lower()
    tokens = text_lower.split()
    
    archetype_scores = {}
    
    for archetype, cue_sets in CUES.items():
        score = 0
        
        for cue_type, patterns in cue_sets.items():
            if not patterns:
                continue
                
            weight = CUE_WEIGHTS[cue_type]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                if matches:
                    for match in matches:
                        # Check for negation
                        if _is_negated(text_lower, match.start()):
                            score += NEGATION_PENALTY
                        else:
                            score += weight
                            
                            # Early position bonus
                            if match.start() < len(" ".join(tokens[:8])):
                                score += BONUSES["early_position"]
                            
                            # Contrast bonus for Hot Take/Reveal
                            if archetype in ["Hot Take/Contradiction", "Reveal/Twist"]:
                                if re.search(r"\b(but|however|instead|except)\b", text_lower):
                                    score += BONUSES["contrast"]
                            
                            # Named entity bonus for Insider/Stats
                            if archetype in ["Insider/Secret", "Stats/Proof/Receipts"]:
                                if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
                                    score += BONUSES["named_entity"]
                            
                            # Intensifier bonuses
                            if archetype == "Shock/Outrage" and "!" in match.group():
                                score += BONUSES["intensifier_shock"]
                            elif archetype == "Relatability/Humor" and "!" in match.group():
                                score += BONUSES["intensifier_humor"]
        
        archetype_scores[archetype] = score
    
    # Select primary and secondary
    if not archetype_scores or max(archetype_scores.values()) < 35:
        return {
            "primary_archetype": None,
            "secondary_archetype": None,
            "emotions": [],
            "confidence": 0.0,
            "cue_features": []
        }
    
    sorted_scores = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
    primary_archetype = sorted_scores[0][0]
    primary_score = sorted_scores[0][1]
    
    secondary_archetype = None
    if len(sorted_scores) > 1 and sorted_scores[1][1] >= primary_score - 10 and sorted_scores[1][1] >= 30:
        secondary_archetype = sorted_scores[1][0]
    
    # Determine emotions
    emotions = []
    for arch in [primary_archetype, secondary_archetype]:
        if arch and arch in ARCHETYPE_TRIGGERS:
            for emotion in ARCHETYPE_TRIGGERS[arch]:
                if emotion not in emotions:
                    emotions.append(emotion)
    
    return {
        "primary_archetype": primary_archetype,
        "secondary_archetype": secondary_archetype,
        "emotions": emotions,
        "confidence": min(1.0, primary_score / 50.0),
        "cue_features": [f"{arch}:{score}" for arch, score in sorted_scores[:3]]
    }

def _is_negated(text: str, pos: int) -> bool:
    """Check if a match position is negated"""
    # Look for negation words before the match
    before_text = text[:pos]
    negation_patterns = [
        r"\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b",
        r"\b(doesn'?t|don'?t|didn'?t|won'?t|can'?t|couldn'?t|shouldn'?t|wouldn'?t)\b",
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, before_text[-50:]):  # Check last 50 chars
            return True
    return False

# ============================================================================
# TEMPLATE SELECTION AND REALIZATION
# ============================================================================

def choose_triggers(primary_arch: str, secondary_arch: str, emotions: List[str]) -> List[str]:
    """Choose emotion triggers based on archetypes"""
    picks = emotions[:]
    
    for arch in [primary_arch, secondary_arch]:
        if arch and arch in ARCHETYPE_TRIGGERS:
            for trig in ARCHETYPE_TRIGGERS[arch]:
                if trig not in picks:
                    picks.append(trig)
    
    return picks[:3]  # Keep it tight

def realize(template: str, clip: Dict, topic: str = None) -> Optional[str]:
    """Realize template with slot filling"""
    topic = topic or cheap_topic(clip.get("text", ""), clip.get("attrs", {}))
    
    # Basic replacements
    result = template.replace("{Topic}", topic.title())
    result = result.replace("{X}", topic.title())
    result = result.replace("{Group}", "Professional")
    result = result.replace("{Problem}", "This Problem")
    result = result.replace("{Outcome}", "Success")
    result = result.replace("{Punchy Claim}", topic.title())
    
    # Stat replacement
    stat = extract_stat(clip.get("text", ""))
    if "{Stat}" in template:
        if stat:
            result = result.replace("{Stat}", stat.title())
        else:
            return None  # Skip stat templates when no stat present
    
    return result

# ============================================================================
# SCORING AND QUALITY GATES
# ============================================================================

BANNED_PHRASES = {"Key Insight", "Key Takeaways", "Inside ", "What It Means"}

def score_title(title: str, trigger: str, classification: Dict, platform: str = "default") -> float:
    """Score title based on multiple factors"""
    title_lower = title.lower()
    
    # Length scoring
    length = len(title)
    if platform in ["tiktok", "reels", "shorts"]:
        if 38 <= length <= 60:
            clarity = 1.0
        elif length <= 72:
            clarity = 0.7
        else:
            clarity = 0.3
    else:  # YouTube long
        if length <= 80:
            clarity = 1.0
        else:
            clarity = 0.3
    
    # Benefit scoring
    benefit_patterns = r"\b(how to|get|save|fix|avoid|faster|simple|win)\b"
    benefit = 1.0 if re.search(benefit_patterns, title_lower) else 0.4
    
    # Novelty scoring
    novelty_patterns = r"\b(1 in \d+|\d+%)\b"
    novelty = 1.0 if re.search(novelty_patterns, title_lower) or "counterintuitive" in title_lower else 0.5
    
    # Keyword scoring
    kw_patterns = ["off-label", "risk", "data", "doctor", "evidence"]
    kw = 1.0 if any(k in title_lower for k in kw_patterns) else 0.5
    
    # Intent fit scoring
    intent_fit = 1.0
    primary_arch = classification.get("primary_archetype")
    if primary_arch == "Advice/Playbook" and "how to" not in title_lower:
        intent_fit = 0.8
    
    # Penalties
    penalties = 0.0
    
    # Banned phrases
    if any(banned.lower() in title_lower for banned in BANNED_PHRASES):
        penalties += 1.2
    
    # Excessive punctuation
    punct_count = title.count(":") + title.count("—") + title.count("-") + title.count("?") + title.count("!")
    if punct_count > 1:
        penalties += 0.3
    
    # Duplicate words
    if re.search(r"\b(\w+)\s+\1\b", title_lower):
        penalties += 0.2
    
    # All caps words
    if re.search(r"\b[A-Z]{4,}\b", title):
        penalties += 0.1
    
    # Clickbait without payoff
    if not classification.get("payoff_ok", True):
        clickbait_patterns = ["secret", "revealed", "exposed"]
        if any(pattern in title_lower for pattern in clickbait_patterns):
            penalties += 0.15
    
    # Calculate final score
    score = (30 * clarity + 25 * intent_fit + 20 * benefit + 15 * kw + 10 * novelty - 20 * penalties)
    
    return round(max(0, score), 1)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_titles_v2(
    text: str,
    *,
    platform: Optional[str] = None,
    n: int = 3,
    avoid_titles: Optional[Iterable[str]] = None,
    episode_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    clip_attrs: Optional[Dict] = None,
    persona: Optional[str] = None,
) -> List[str]:
    """
    Generate CFD-first titles using archetype classification and template registry
    """
    if not text or len(text.strip()) < 3:
        return _generate_fallback_titles(text or "test", platform or "default", n, set(avoid_titles or []))
    
    # Normalize inputs
    platform = (platform or "default").lower()
    avoid_set = set(avoid_titles or [])
    clip_attrs = clip_attrs or {}
    
    # Classify the transcript
    classification = classify_transcript(text)
    
    # If confidence is too low, fall back to simple approach
    if classification["confidence"] < 0.35:
        logger.debug("Low classification confidence, using fallback")
        return _generate_fallback_titles(text, platform, n, avoid_set)
    
    # Choose emotion triggers
    triggers = choose_triggers(
        classification["primary_archetype"],
        classification["secondary_archetype"],
        classification["emotions"]
    )
    
    # Apply persona weighting if specified
    if persona and persona in PERSONA_WEIGHTS:
        weights = PERSONA_WEIGHTS[persona]
        # Adjust trigger order based on persona preferences
        # This is a simplified implementation
        pass
    
    # Generate candidates
    candidates = []
    clip_data = {
        "text": text,
        "attrs": clip_attrs,
        "payoff_ok": clip_attrs.get("payoff_ok", True)
    }
    
    for trigger in triggers:
        # Get templates for this trigger
        templates = TEMPLATES.get(trigger, [])
        
        # Also check archetype-specific templates
        for arch in [classification["primary_archetype"], classification["secondary_archetype"]]:
            if arch and arch in TEMPLATES:
                templates.extend(TEMPLATES[arch])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_templates = []
        for t in templates:
            if t not in seen:
                seen.add(t)
                unique_templates.append(t)
        
        # Generate titles from templates
        for template in unique_templates[:2]:  # Max 2 per trigger
            realized = realize(template, clip_data)
            if not realized or any(avoided.lower() in realized.lower() for avoided in avoid_set):
                continue
            
            score = score_title(realized, trigger, classification, platform)
            candidates.append({
                "title": realized,
                "score": score,
                "trigger": trigger,
                "archetype": classification["primary_archetype"]
            })
    
    # Sort by score and return top N
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Remove duplicates
    seen_titles = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate["title"] not in seen_titles:
            seen_titles.add(candidate["title"])
            unique_candidates.append(candidate)
    
    return [c["title"] for c in unique_candidates[:n]]

def _generate_fallback_titles(text: str, platform: str, n: int, avoid_set: Set[str]) -> List[str]:
    """Fallback title generation for low-confidence cases"""
    topic = cheap_topic(text)
    
    fallback_templates = [
        f"What You Need to Know About {topic.title()}",
        f"The Truth About {topic.title()}",
        f"Why {topic.title()} Matters",
        "Important Insight",
        "Key Information",
        "What You Should Know",
    ]
    
    candidates = []
    for i, template in enumerate(fallback_templates[:n]):
        if not any(avoided.lower() in template.lower() for avoided in avoid_set):
            candidates.append({
                "title": template,
                "score": 50.0 - (i * 5),
                "trigger": "Fallback",
                "archetype": "Generic"
            })
    
    return [c["title"] for c in candidates]

# ============================================================================
# TEST CASES
# ============================================================================

TESTS = [
    ("Hot Take/Contradiction", "Everyone says it works, but actually it doesn't."),
    ("Reveal/Twist", "I thought X was the fix. Then everything changed."),
    ("Stats/Proof/Receipts", "One in four prescriptions is off-label. The data shows..."),
    ("Confession/Vulnerability", "Honestly, I'm embarrassed to admit this."),
    ("Advice/Playbook", "Here's how to fix it in three steps."),
    ("Shock/Outrage", "They lied to us. This should be illegal!"),
    ("Relatability/Humor", "Tell me you're a parent without telling me you're a parent 😂"),
    ("Origin/Backstory", "Back when I started in 2014..."),
    ("Mistake/Failure/Lesson", "Biggest mistake of my career. Never again."),
    ("Debate/Argument", "You're wrong. Prove it."),
    ("Mystery/Cliffhanger", "Wait for it... you won't believe the ending."),
]

def run_tests():
    """Run test cases to validate classification"""
    results = []
    
    for expected_archetype, text in TESTS:
        classification = classify_transcript(text)
        actual_archetype = classification["primary_archetype"]
        
        results.append({
            "expected": expected_archetype,
            "actual": actual_archetype,
            "correct": actual_archetype == expected_archetype,
            "confidence": classification["confidence"],
            "emotions": classification["emotions"]
        })
    
    return results

if __name__ == "__main__":
    # Run tests
    test_results = run_tests()
    for result in test_results:
        status = "✅" if result["correct"] else "❌"
        print(f"{status} {result['expected']} -> {result['actual']} (conf: {result['confidence']:.2f})")
