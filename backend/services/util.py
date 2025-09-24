"""
Utility functions for cross-cutting concerns and data normalization.
"""

from typing import Dict, List, Any, Tuple, Union
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """
    Dict that also supports attribute access: d.key == d['key'].
    Safe for legacy code paths that expect seg.start / seg.text / word.prob
    while preserving normal dict semantics for callers that use ['key'].
    """
    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        # keep behavior consistent with dict: set as key
        self[name] = value


def coerce_word_schema(w) -> dict:
    """Coerce word objects to canonical schema with word/start/end/prob fields."""
    # Helper to get attribute or dict key with fallbacks
    def g(attr, *alts):
        if hasattr(w, attr):
            return getattr(w, attr)
        if isinstance(w, dict):
            for key in [attr] + list(alts):
                if key in w:
                    return w[key]
        return None

    txt = g("word") or g("text") or g("token") or ""
    start = g("start") or 0.0
    end = g("end") or 0.0
    prob = g("prob") or g("probability") or 1.0

    # normalize types
    try:
        start = float(start)
    except (ValueError, TypeError):
        start = 0.0
    try:
        end = float(end)
    except (ValueError, TypeError):
        end = start
    try:
        prob = float(prob)
    except (ValueError, TypeError):
        prob = 1.0

    return {"word": str(txt), "start": start, "end": end, "prob": prob}


def to_attrdict_list(items: List[Dict[str, Any]]) -> List[AttrDict]:
    """Convert list of dicts to AttrDict objects for attribute access compatibility."""
    out: List[AttrDict] = []
    for s in items:
        if isinstance(s, AttrDict):
            out.append(s)
            continue
        if not isinstance(s, dict):
            # leave non-dicts untouched (already an object from some path)
            out.append(s)  # type: ignore
            continue
        sd = AttrDict(s)
        # normalize nested words if present
        wlist = sd.get("words")
        if isinstance(wlist, list):
            sd["words"] = [AttrDict(w) if isinstance(w, dict) else w for w in wlist]
        out.append(sd)
    return out


def _coerce_word(w: Any) -> Dict[str, Any]:
    """Accept FW Word object or dict variants and return {start,end,text}."""
    if isinstance(w, dict):
        start = w.get("start", w.get("t", 0.0)) or 0.0
        end = w.get("end", w.get("d", start)) or start
        text = w.get("text", w.get("word", w.get("w", ""))) or ""
        return {"start": float(start), "end": float(end), "text": str(text)}
    start = getattr(w, "start", getattr(w, "t", 0.0)) or 0.0
    end = getattr(w, "end", getattr(w, "d", start)) or start
    text = getattr(w, "word", getattr(w, "text", "")) or ""
    return {"start": float(start), "end": float(end), "text": str(text)}


def _coerce_segment(seg: Any) -> Dict[str, Any]:
    """Accept FW Segment object or dict and return normalized dict with words list coerced."""
    if isinstance(seg, dict):
        start = seg.get("start", 0.0) or 0.0
        end = seg.get("end", start) or start
        text = seg.get("text", "") or ""
        words = seg.get("words", None)
    else:
        start = getattr(seg, "start", 0.0) or 0.0
        end = getattr(seg, "end", start) or start
        text = getattr(seg, "text", "") or ""
        words = getattr(seg, "words", None)

    words_list: List[Dict[str, Any]] = []
    if words is not None:
        try:
            words_list = [_coerce_word(w) for w in words]
        except TypeError:
            # Single word object
            words_list = [_coerce_word(words)]

    return {"start": float(start), "end": float(end), "text": str(text), "words": words_list}


def normalize_asr_result(res: Union[Tuple, Dict, List]) -> Tuple[List, Dict, List]:
    """
    Normalize ASR result to consistent (segments, info, words) tuple format.
    
    Supports:
    - (segments, info, words) tuple
    - (segments, info) tuple  
    - dict with "segments", "info", "words" keys
    - Single Segment object (wraps to list)
    """
    segs, info, words = [], {}, []
    
    if isinstance(res, tuple):
        # (segments, info) or (segments, info, words)
        parts = list(res)
        if len(parts) >= 1: 
            segs = parts[0]
        if len(parts) >= 2: 
            info = parts[1] or {}
        if len(parts) >= 3: 
            words = parts[2] or []
    elif isinstance(res, dict):
        segs = res.get("segments", [])
        info = res.get("info", {}) or {}
        words = res.get("words", []) or []
    else:
        segs = res
    
    # If a single Segment object sneaks in, wrap it
    if hasattr(segs, "start") and hasattr(segs, "end"):
        segs = [segs]
    
    # Convert Segment/Word objects to plain dicts and aggregate words
    out: List[Dict[str, Any]] = []
    for s in segs or []:
        out.append(_coerce_segment(s))

    # Prefer provided words (normalize), else aggregate from segments
    words_out: List[Dict[str, Any]] = []
    if words:
        # Normalize incoming word schema to {start,end,text}
        words_out = _normalize_schema(words)
    if not words_out:
        for seg in out:
            seg_words = seg.get("words") or []
            for w in seg_words:
                if isinstance(w, dict):
                    words_out.append(_coerce_word(w))
                else:
                    words_out.append(_coerce_word(w))

    return out, info, words_out


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
