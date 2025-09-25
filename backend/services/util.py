"""
Utility functions for cross-cutting concerns and data normalization.
"""

from typing import Dict, List, Any, Tuple, Union
import logging
import json
import os
import tempfile
from pathlib import Path
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


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Atomically write JSON data to a file using temp file + rename.
    Ensures atomic writes on Windows and prevents partial/corrupted files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory for atomic rename
    with tempfile.NamedTemporaryFile(
        mode='w', 
        delete=False, 
        dir=path.parent, 
        suffix='.tmp',
        encoding='utf-8'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        json.dump(data, tmp_file, ensure_ascii=False, separators=(',', ':'))
        tmp_file.flush()
        os.fsync(tmp_file.fileno())  # Force write to disk
    
    try:
        # Atomic rename (works on same filesystem)
        os.replace(tmp_path, path)
    except Exception as e:
        # Clean up temp file on failure
        try:
            tmp_path.unlink(missing_ok=True)
        except:
            pass
        raise e


def detect_sentence_endings_from_words(words: List[Dict]) -> List[float]:
    """
    Return list of times (float seconds) marking likely sentence ends.
    Enhanced version that works with the existing EOS system.
    """
    eos = []
    n = len(words)
    
    for i, w in enumerate(words):
        text = (w.get("word") or w.get("text") or "").strip()
        end = float(w.get("end") or 0.0)

        if not text:
            continue

        # Strong punctuation
        if text.endswith((".", "!", "?")):
            nxt = words[i + 1] if i + 1 < n else None
            nxt_tok = (nxt.get("word") or nxt.get("text") or "").strip() if nxt else ""
            if (not nxt) or (nxt_tok and nxt_tok[0].isupper()):
                eos.append(end)
                continue

        # Weak punctuation + pause
        if text.endswith((",", ";", ":")) and i + 1 < n:
            nxt = words[i + 1]
            gap = float(nxt.get("start") or 0.0) - end
            if gap > 0.5:
                eos.append(end)

    return eos


def unify_eos_markers(existing_eos: List[float], eos_from_words: List[float], *, tol: float = 0.25) -> List[float]:
    """
    Merge and de-dup EOS markers; prefer words-based within ±tol seconds.
    """
    merged = []
    for t in sorted(existing_eos + eos_from_words):
        if not merged or abs(t - merged[-1]) > tol:
            merged.append(t)
        else:
            # Collision: prefer the words-based timestamp if it's in this neighborhood
            # Heuristic: if any words-based marker falls within tol, use the one
            # closest to a punctuation end; otherwise keep earliest.
            pass  # optional: keep simple, current 'closest-to-words' logic below
    
    # Simple preference: snap merged markers to nearest words-based marker if within tol
    if eos_from_words:
        snapped = []
        for t in merged:
            nearest = min(eos_from_words, key=lambda w: abs(w - t), default=None)
            snapped.append(nearest if nearest is not None and abs(nearest - t) <= tol else t)
        merged = snapped
    
    return merged


def extend_to_natural_end(clip: Dict, words: List[Dict], max_extend_sec: float = 3.0) -> Dict:
    """
    Extend clip['end'] to the next EOS within max_extend_sec, if any.
    """
    end = float(clip.get("end", 0.0))
    best = None
    n = len(words)
    
    # Find first word ending after current end
    for i, w in enumerate(words):
        w_end = float(w.get("end") or 0.0)
        if w_end <= end:
            continue
        if w_end - end > max_extend_sec:
            break
            
        tok = (w.get("word") or w.get("text") or "").strip()
        if tok.endswith((".", "!", "?")):
            nxt = words[i + 1] if i + 1 < n else None
            nxt_tok = (nxt.get("word") or nxt.get("text") or "").strip() if nxt else ""
            if (not nxt) or (nxt_tok and nxt_tok[0].isupper()):
                best = w_end
                break

    if best is not None and best > end:
        clip["end"] = best
        clip["extended"] = True
        clip["extension_delta"] = best - end
    
    return clip


def _coerce_words_list(words):
    """
    Return a list[dict] of words with keys ('t' or 'start', 'd' or 'end', 'w' or 'text', 'prob' optional).
    If words is invalid (None/str/other), return [] so callers can proceed safely.
    """
    if not isinstance(words, list):
        return []
    out = []
    for w in words:
        if not isinstance(w, dict):
            continue
        txt = (w.get("w") or w.get("word") or w.get("text") or "").strip()
        t0  = float(w.get("t", w.get("start", 0.0)) or 0.0)
        d   = float(w.get("d", (w.get("end", t0) - t0)) or 0.0)
        prob = float(w.get("prob", w.get("p", w.get("confidence", 0.7)))) or 0.7
        out.append({"text": txt, "start": t0, "dur": d, "end": t0 + max(0.0, d), "prob": prob})
    return out

def _nearest_eos_after(end_t: float, eos_markers: list[float] | None) -> float | None:
    if not eos_markers:
        return None
    after = [t for t in eos_markers if t >= end_t]
    return (min(after) - end_t) if after else None

def calculate_finish_confidence(
    clip: dict,
    words: list | None = None,
    eos_markers: list[float] | None = None,
) -> float:
    """
    Returns [0..1] confidence a clip is a finished thought.
    Robust to bad 'words' inputs (None/str/etc).
    """
    words_coerced = _coerce_words_list(words)
    
    text = (clip.get("text") or "").strip()
    if not text:
        return 0.0
    
    confidence = 0.0
    
    # Strong punctuation (high confidence)
    if text.endswith((".", "!", "?")):
        confidence += 0.6
        
        # Check for proper sentence structure (capitalization after period)
        if words_coerced:
            clip_end = float(clip.get("end", 0.0))
            for w in words_coerced:
                w_start = float(w.get("start", 0.0))
                if w_start > clip_end + 0.1:  # Next word after clip
                    w_text = (w.get("text") or "").strip()
                    if w_text and w_text[0].isupper():
                        confidence += 0.2
                    break
    
    # Discourse closers (medium-high confidence)
    discourse_closers = [
        "and that's why", "so we", "that's why", "which is why", 
        "this is why", "that's it", "that's right", "all set", "we're done"
    ]
    text_lower = text.lower()
    if any(closer in text_lower for closer in discourse_closers):
        confidence += 0.4
    
    # Pause after end (if words available)
    if words_coerced:
        clip_end = float(clip.get("end", 0.0))
        for i, w in enumerate(words_coerced):
            w_start = float(w.get("start", 0.0))
            if w_start > clip_end:
                gap = w_start - clip_end
                if gap > 0.5:  # Significant pause
                    confidence += 0.2
                break
    
    # Weak punctuation (low confidence)
    if text.endswith((",", ";", ":")):
        confidence += 0.1
    
    # EOS proximity boost (if eos_markers provided)
    end_t = float(clip.get("end", 0.0))
    nearest = _nearest_eos_after(end_t, eos_markers)
    if nearest is not None and 0.0 <= nearest <= 1.2:
        # Gentle, capped boost (up to +0.10) for very close EOS
        confidence = min(1.0, confidence + (0.10 * (1.2 - nearest) / 1.2))
    
    return min(1.0, confidence)


def finish_threshold_for(genre: str, indicators: Dict) -> float:
    """
    Calculate adaptive finish threshold based on genre and content indicators.
    """
    t = 0.60  # Base threshold
    
    if indicators.get("conversational_ratio", 0) > 0.7:
        t -= 0.15
    if genre in {"educational", "technical"}:
        t += 0.10
    if indicators.get("interview_format"):
        t -= 0.10
    
    return max(0.30, min(0.80, t))
