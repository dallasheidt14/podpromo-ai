"""
Unicode text normalization and EOS detection utilities.
Handles international text and smart punctuation correctly.
"""

import unicodedata
import re
import logging

logger = logging.getLogger(__name__)

# CJK (Chinese, Japanese, Korean) sentence-ending punctuation
CJK_EOS = "。？！"

# Common smart punctuation and symbols
ELLIPSIS = "\u2026"  # Horizontal ellipsis
SMART_QUOTES = {
    "'": "'",  # Right single quotation mark
    "'": "'",  # Left single quotation mark  
    """: '"',  # Left double quotation mark
    """: '"',  # Right double quotation mark
}

def normalize_text(text: str) -> str:
    """
    Normalize Unicode text for consistent processing.
    
    Handles:
    - Unicode normalization (NFKC)
    - Smart quotes → ASCII quotes
    - Ellipsis → three dots
    - Other common smart punctuation
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization (NFKC handles most compatibility issues)
    normalized = unicodedata.normalize("NFKC", text)
    
    # Replace smart quotes with ASCII equivalents
    for smart, ascii in SMART_QUOTES.items():
        normalized = normalized.replace(smart, ascii)
    
    # Replace ellipsis with three dots
    normalized = normalized.replace(ELLIPSIS, "...")
    
    # Replace other common smart punctuation
    normalized = normalized.replace("–", "-")  # En dash
    normalized = normalized.replace("—", "-")  # Em dash
    
    return normalized

def is_eos_char(char: str) -> bool:
    """
    Check if a character is a sentence-ending punctuation mark.
    
    Supports:
    - ASCII punctuation: . ? !
    - CJK punctuation: 。 ？ ！
    
    Args:
        char: Single character to check
        
    Returns:
        True if character ends sentences
    """
    if not char:
        return False
    
    # ASCII sentence endings
    if char in ".?!":
        return True
    
    # CJK sentence endings
    if char in CJK_EOS:
        return True
    
    return False

def extract_eos_positions(text: str) -> list[int]:
    """
    Extract positions of sentence-ending punctuation in text.
    
    Args:
        text: Input text
        
    Returns:
        List of character positions where sentences end
    """
    eos_positions = []
    
    for i, char in enumerate(text):
        if is_eos_char(char):
            eos_positions.append(i)
    
    return eos_positions

def normalize_segment_text(segment: dict) -> dict:
    """
    Normalize text in a segment dictionary.
    
    Args:
        segment: Segment dictionary with 'text' field
        
    Returns:
        Updated segment with normalized text
    """
    if "text" in segment and segment["text"]:
        original_text = segment["text"]
        normalized_text = normalize_text(original_text)
        
        if original_text != normalized_text:
            logger.debug(f"TEXT_NORMALIZE: '{original_text[:50]}...' → '{normalized_text[:50]}...'")
            segment["text"] = normalized_text
    
    return segment

def normalize_all_segments(segments: list) -> list:
    """
    Normalize text in all segments.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        List of segments with normalized text
    """
    normalized_segments = []
    
    for segment in segments:
        try:
            normalized_segment = normalize_segment_text(segment.copy())
            normalized_segments.append(normalized_segment)
        except Exception as e:
            logger.warning(f"TEXT_NORMALIZE_ERROR: Failed to normalize segment: {e}")
            # Keep original segment if normalization fails
            normalized_segments.append(segment)
    
    return normalized_segments
