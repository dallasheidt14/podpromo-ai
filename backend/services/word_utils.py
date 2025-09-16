"""
Word token normalization utilities to handle mixed schemas.
Ensures consistent word token format across the entire pipeline.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def normalize_word_token(tok: dict) -> dict:
    """
    Normalize word token to unified schema: {'word': str, 'start': float, 'end': float}
    
    Handles both schemas:
    - {'w': 'word', 's': 12.34, 'e': 12.87}
    - {'word': 'word', 'start': 12.34, 'end': 12.87}
    """
    if not isinstance(tok, dict):
        return {'word': '', 'start': None, 'end': None}
    
    # Accepts {'w','s','e'} or {'word','start','end'}
    return {
        "word": tok.get("word") or tok.get("w") or "",
        "start": tok.get("start", tok.get("s")),
        "end": tok.get("end", tok.get("e")),
    }

def normalize_word_tokens(tokens: List[dict]) -> List[dict]:
    """Normalize a list of word tokens and filter out invalid ones."""
    if not tokens:
        return []
    
    normalized = []
    for tok in tokens:
        norm_tok = normalize_word_token(tok)
        # Only keep tokens with valid timing data
        if norm_tok['start'] is not None and norm_tok['end'] is not None:
            normalized.append(norm_tok)
    
    return normalized

def slice_transcript(words: List[dict], t0: float, t1: float) -> List[dict]:
    """
    Slice transcript using word-level timestamps.
    
    Args:
        words: List of word tokens (any schema)
        t0: Start time in seconds
        t1: End time in seconds
    
    Returns:
        List of normalized word tokens within time range
    """
    if not words:
        return []
    
    # Normalize all tokens
    w = normalize_word_tokens(words)
    if not w:
        return []
    
    # Filter words within time range
    filtered = []
    for word in w:
        start_time = word.get('start')
        end_time = word.get('end')
        if start_time is not None and end_time is not None:
            if start_time >= t0 - 0.05 and end_time <= t1 + 0.05:
                filtered.append(word)
            elif start_time < t1 and end_time > t0:  # overlap fallback
                filtered.append(word)
    
    return filtered

def fallback_sentence_or_chars(episode_text: str, t0: float, t1: float, 
                              sent_map: Optional[List[dict]] = None) -> str:
    """
    Fallback transcript slicing using sentence boundaries or character windows.
    
    Args:
        episode_text: Full episode transcript
        t0: Start time in seconds
        t1: End time in seconds
        sent_map: Optional sentence boundary map
    
    Returns:
        Fallback transcript text (bounded to ~800 chars)
    """
    if not episode_text:
        return ""
    
    # Try EOS/sentence map first
    if sent_map:
        # Find nearest sentence span
        for sent in sent_map:
            sent_start = sent.get('start', 0)
            sent_end = sent.get('end', 0)
            if sent_start <= t0 and sent_end >= t1:
                return sent.get('text', '')[:800].strip()
    
    # Character window fallback (bounded)
    # Approximate character range based on time ratio
    total_duration = len(episode_text) / 200  # rough chars per second
    if total_duration > 0:
        char_start = int((t0 / total_duration) * len(episode_text))
        char_end = int((t1 / total_duration) * len(episode_text))
        
        # Add padding and bounds
        start_idx = max(0, char_start - 120)
        end_idx = min(len(episode_text), char_end + 120)
        
        return episode_text[start_idx:end_idx][:800].strip()
    
    return ""

def validate_word_tokens(tokens: List[dict]) -> bool:
    """Validate that word tokens have the required schema."""
    if not tokens:
        return False
    
    for tok in tokens:
        if not isinstance(tok, dict):
            return False
        if 'w' not in tok or 's' not in tok or 'e' not in tok:
            return False
        if tok['s'] is None or tok['e'] is None:
            return False
    
    return True

def get_transcript_with_fallback(words: List[dict], episode_text: str, 
                                t0: float, t1: float, 
                                sent_map: Optional[List[dict]] = None) -> str:
    """
    Get transcript text with comprehensive fallback strategy.
    
    Returns:
        Transcript text, never empty string unless all fallbacks fail
    """
    # Primary: word-level slice
    text = slice_transcript(words, t0, t1)
    if text:
        return text
    
    # Fallback: sentence or character-based
    text = fallback_sentence_or_chars(episode_text, t0, t1, sent_map)
    if text:
        logger.warning(f"TRANSCRIPT_FALLBACK: used sentence/char slice for [{t0:.1f}, {t1:.1f}]")
        return text
    
    # Last resort: empty string (log once)
    logger.warning(f"TRANSCRIPT_EMPTY: no text found for [{t0:.1f}, {t1:.1f}]")
    return ""
