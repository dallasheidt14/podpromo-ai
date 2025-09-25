# backend/services/keyword_extraction.py
# Simple keyword extraction with stopwords + proper-noun preference

import re
from typing import List, Set
from collections import Counter

# Common English stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
    'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'would', 'you', 'your', 'this', 'these',
    'they', 'them', 'their', 'there', 'then', 'than', 'but', 'or', 'so', 'if', 'when', 'where', 'why',
    'how', 'what', 'who', 'which', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall',
    'have', 'had', 'has', 'having', 'been', 'being', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting',
    'go', 'went', 'going', 'come', 'came', 'coming', 'see', 'saw', 'seeing', 'know', 'knew', 'knowing',
    'think', 'thought', 'thinking', 'take', 'took', 'taking', 'make', 'made', 'making', 'give', 'gave',
    'giving', 'say', 'said', 'saying', 'tell', 'told', 'telling', 'use', 'used', 'using', 'find', 'found',
    'finding', 'work', 'worked', 'working', 'call', 'called', 'calling', 'try', 'tried', 'trying',
    'ask', 'asked', 'asking', 'need', 'needed', 'needing', 'feel', 'felt', 'feeling', 'become', 'became',
    'becoming', 'leave', 'left', 'leaving', 'put', 'putting', 'mean', 'meant', 'meaning', 'keep', 'kept',
    'keeping', 'let', 'letting', 'begin', 'began', 'beginning', 'seem', 'seemed', 'seeming', 'help',
    'helped', 'helping', 'talk', 'talked', 'talking', 'turn', 'turned', 'turning', 'start', 'started',
    'starting', 'show', 'showed', 'showing', 'hear', 'heard', 'hearing', 'play', 'played', 'playing',
    'run', 'ran', 'running', 'move', 'moved', 'moving', 'live', 'lived', 'living', 'believe', 'believed',
    'believing', 'hold', 'held', 'holding', 'bring', 'brought', 'bringing', 'happen', 'happened',
    'happening', 'write', 'wrote', 'writing', 'sit', 'sat', 'sitting', 'stand', 'stood', 'standing',
    'lose', 'lost', 'losing', 'pay', 'paid', 'paying', 'meet', 'met', 'meeting', 'include', 'included',
    'including', 'continue', 'continued', 'continuing', 'set', 'setting', 'learn', 'learned', 'learning',
    'change', 'changed', 'changing', 'lead', 'led', 'leading', 'understand', 'understood', 'understanding',
    'watch', 'watched', 'watching', 'follow', 'followed', 'following', 'stop', 'stopped', 'stopping',
    'create', 'created', 'creating', 'speak', 'spoke', 'speaking', 'read', 'reading', 'allow', 'allowed',
    'allowing', 'add', 'added', 'adding', 'spend', 'spent', 'spending', 'grow', 'grew', 'growing',
    'open', 'opened', 'opening', 'walk', 'walked', 'walking', 'win', 'won', 'winning', 'offer', 'offered',
    'offering', 'remember', 'remembered', 'remembering', 'love', 'loved', 'loving', 'consider', 'considered',
    'considering', 'appear', 'appeared', 'appearing', 'buy', 'bought', 'buying', 'wait', 'waited', 'waiting',
    'serve', 'served', 'serving', 'die', 'died', 'dying', 'send', 'sent', 'sending', 'expect', 'expected',
    'expecting', 'build', 'built', 'building', 'stay', 'stayed', 'staying', 'fall', 'fell', 'falling',
    'cut', 'cutting', 'reach', 'reached', 'reaching', 'kill', 'killed', 'killing', 'remain', 'remained',
    'remaining', 'suggest', 'suggested', 'suggesting', 'raise', 'raised', 'raising', 'pass', 'passed',
    'passing', 'sell', 'sold', 'selling', 'require', 'required', 'requiring', 'report', 'reported',
    'reporting', 'decide', 'decided', 'deciding', 'pull', 'pulled', 'pulling'
}

def extract_salient_keywords(text: str, limit: int = 5) -> List[str]:
    """
    Extract salient keywords from text using simple heuristics.
    Prioritizes proper nouns, capitalized words, and numbers.
    """
    if not text or not text.strip():
        return []
    
    # Tokenize and clean
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stopwords and very short words
    filtered_words = [
        word for word in words 
        if len(word) > 2 and word not in STOPWORDS
    ]
    
    if not filtered_words:
        return []
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Find proper nouns and capitalized words in original text
    proper_nouns = set()
    capitalized_words = set()
    numbers = set()
    
    # Look for proper nouns (capitalized words that aren't at sentence start)
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        words_in_sentence = re.findall(r'\b\w+\b', sentence.strip())
        if words_in_sentence:
            # Skip first word (likely capitalized due to sentence start)
            for word in words_in_sentence[1:]:
                if word[0].isupper() and len(word) > 2:
                    proper_nouns.add(word.lower())
    
    # Find all capitalized words
    for word in re.findall(r'\b[A-Z][a-z]+\b', text):
        if len(word) > 2:
            capitalized_words.add(word.lower())
    
    # Find numbers
    for match in re.findall(r'\b\d+\b', text):
        numbers.add(match)
    
    # Score words based on frequency and type
    scored_words = []
    for word, count in word_counts.items():
        score = count
        
        # Boost proper nouns and capitalized words
        if word in proper_nouns:
            score += 10
        elif word in capitalized_words:
            score += 5
        
        # Boost numbers
        if word in numbers:
            score += 3
        
        # Boost longer words (more specific)
        score += len(word) * 0.1
        
        scored_words.append((word, score))
    
    # Sort by score (descending) and return top keywords
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    # Return unique keywords up to limit
    keywords = []
    seen = set()
    for word, _ in scored_words:
        if word not in seen:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= limit:
                break
    
    return keywords
