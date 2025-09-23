"""
Utility functions for cross-cutting concerns and data normalization.
"""

from typing import Dict, List, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def normalize_asr_result(res: Union[Tuple, Dict, List]) -> Tuple[List, Dict, List]:
    """
    Normalize ASR result to consistent (segments, info, words) tuple format.
    
    Supports:
    - (segments, info, words) tuple
    - (segments, info) tuple  
    - dict with "segments", "info", "words" keys
    """
    if isinstance(res, dict):
        segments = res.get("segments", [])
        info = res.get("info") or {}
        words = res.get("words") or []
        return segments, info, words
    
    if isinstance(res, (list, tuple)):
        if len(res) == 3:
            segments, info, words = res
        elif len(res) == 2:
            segments, info = res
            words = []
        else:
            raise TypeError(f"Unexpected ASR tuple length: {len(res)}")
        return segments or [], (info or {}), (words or [])
    
    raise TypeError(f"Unexpected ASR result type: {type(res)}")


def _normalize_schema(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize word timestamps to consistent {start, end, text} format.
    Bulletproof version that handles all edge cases.
    """
    out = []
    if not words:
        return out
    
    for w in words:
        if not isinstance(w, dict):
            continue
            
        # Handle {t, d, w} format
        if "w" in w and "t" in w and "d" in w:
            start = float(w.get("t", 0.0))
            end = start + float(w.get("d", 0.0))
            text = str(w.get("w", "")).strip()
        # Handle {start, end, text} format
        elif "start" in w and "end" in w and "text" in w:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            text = str(w.get("text", "")).strip()
        # Handle {start, end, word} format (alternative field name)
        elif "start" in w and "end" in w and "word" in w:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            text = str(w.get("word", "")).strip()
        else:
            continue
            
        if text and end > start:
            out.append({"start": start, "end": end, "text": text})
    
    return out


def _synthesize_words_from_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    """
    Very simple word synthesizer from segments.
    Good enough as a safety net when word timestamps are missing.
    """
    out = []
    if not segments:
        return out
        
    for seg in segments:
        # Handle different segment formats
        if hasattr(seg, 'start') and hasattr(seg, 'end') and hasattr(seg, 'text'):
            s = float(seg.start)
            e = float(seg.end)
            txt = str(seg.text or "").strip()
        elif isinstance(seg, dict):
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            txt = str(seg.get("text", "")).strip()
        else:
            continue
            
        if not txt or e <= s:
            continue
            
        tokens = [t for t in txt.split() if t]
        if not tokens:
            continue
            
        step = (e - s) / max(1, len(tokens))
        for i, t in enumerate(tokens):
            start = s + i * step
            end = s + (i + 1) * step
            out.append({"start": start, "end": end, "text": t})
    
    return out


def _extract_words_safe(segments: List[Any], words: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Extract word timestamps safely with synthesis fallback.
    Never returns None - always returns a list (empty if no words).
    
    Args:
        segments: List of transcription segments
        words: Optional existing words to normalize first
    
    Returns:
        List of normalized word dicts with {start, end, text} format
    """
    # 1) prefer provided words (normalize schema)
    norm = _normalize_schema(words) if words else []
    if norm:
        logger.debug(f"Using normalized words: {len(norm)} words")
        return norm
    
    # 2) synthesize from segments
    synthesized = _synthesize_words_from_segments(segments)
    logger.debug(f"Synthesized {len(synthesized)} words from {len(segments)} segments")
    return synthesized
