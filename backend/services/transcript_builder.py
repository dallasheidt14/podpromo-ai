"""
Exact transcript building for clips - matches audio window precisely
"""
import re
import logging

logger = logging.getLogger(__name__)

def _concat_segment_text(source, start_s: float, end_s: float) -> str:
    """
    Build transcript by concatenating segment texts that overlap [start, end].
    Fallback when word timestamps are not available.
    """
    segments = getattr(source, "segments", None) or []
    if not segments:
        return ""
    
    # Find segments that overlap with the clip window
    overlapping_segments = []
    for seg in segments:
        seg_start = getattr(seg, "start", 0.0)
        seg_end = getattr(seg, "end", 0.0)
        # Check for overlap
        if seg_end > start_s and seg_start < end_s:
            overlapping_segments.append(seg)
    
    # Concatenate text from overlapping segments
    texts = []
    for seg in overlapping_segments:
        text = getattr(seg, "text", "") or ""
        if text.strip():
            texts.append(text.strip())
    
    return " ".join(texts)

def slice_items_by_time(items, start: float, end: float, get_start, get_end, pad_s: float = 0.25, eps: float = 0.01):
    """
    Return items that OVERLAP the [start-pad, end+pad] window.
    Overlap check fixes lost head/tail words that straddle boundaries.
    """
    s, e = max(0.0, float(start) - pad_s), float(end) + pad_s
    out = []
    for it in items:
        ws, we = float(get_start(it)), float(get_end(it))
        if we > s - eps and ws < e + eps:  # overlap, not containment
            out.append(it)
    return out

def slice_words_by_time(words, start: float, end: float, pad_s: float = 0.25, eps: float = 1e-3):
    """Slice words using overlap logic with proper boundary handling"""
    s_p = start - pad_s
    e_p = end + pad_s
    return [w for w in words
            if (w.get("start", 0.0) < e_p - eps) and (w.get("end", 0.0) > s_p + eps)]

def slice_segments_by_time(segments, start: float, end: float, pad_s: float = 0.25):
    """Slice segments using overlap logic"""
    return slice_items_by_time(segments, start, end, lambda s: s["start"], lambda s: s["end"], pad_s)

def build_words_for_clip(words, clip_start, clip_end):
    """Build words for clip with hard clamping to prevent overshoot"""
    eps = 1e-3
    out = []
    for w in words:
        w_t0 = float(w.get("t", w.get("start", 0.0)))
        w_d  = float(w.get("d", w.get("end", 0.0) - w_t0))
        w_t1 = w_t0 + max(0.0, w_d)

        t0 = max(clip_start, w_t0)
        t1 = min(clip_end,   w_t1)
        if t1 - t0 <= eps:
            continue

        out.append({
            "w": w.get("w", w.get("text", w.get("word", ""))),
            "t": round(t0 - clip_start, 2),               # relative time
            "d": round(max(0.0, t1 - t0), 2)
        })

    # make sure last token never spills past clip length
    if out:
        L = round(clip_end - clip_start, 2)
        last = out[-1]
        if last["t"] + last["d"] > L + eps:
            last["d"] = round(max(0.0, L - last["t"]), 2)
    return out

def build_clip_transcript_exact(source, start_s: float, end_s: float, pad_s: float = 0.25) -> tuple[str, str, dict]:
    """
    Build transcript from word timestamps inside the exact clip window.
    source can be episode object or list[dict] of words.
    Returns (text, source_type, metadata)
    """
    # Handle None inputs
    if start_s is None or end_s is None:
        return "", "none", {"start": start_s, "end": end_s, "word_count": 0, "coverage_s": 0.0}
    
    # Get words from source (episode or list)
    words = getattr(source, "words", None) or source or []
    
    # Graceful fallback if no words available
    if not words:
        logger.warning("CLIP_TRANSCRIPT: episode.words missing; using segment-text fallback")
        # Build transcript by concatenating segment texts that overlap [start, end]
        text = _concat_segment_text(source, start_s, end_s)
        return text, "segment_fallback", {"start": start_s, "end": end_s, "word_count": 0, "coverage_s": 0.0}
    
    # Normalize word schema to handle different formats
    norm = []
    for w in words:
        if not isinstance(w, dict):
            continue
            
        # Try different text field names
        text = w.get("text") or w.get("word") or w.get("token")
        start = w.get("start") or w.get("ts")
        end = w.get("end") or w.get("te")
        
        if text is not None and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            norm.append({
                "start": float(start), 
                "end": float(end), 
                "text": str(text).strip()
            })
    
    if not norm:
        logger.warning(f"CLIP_TRANSCRIPT: episode.words missing, no valid words found for window [{start_s:.2f}, {end_s:.2f}]")
        return "", "none", {"start": start_s, "end": end_s, "word_count": 0, "coverage_s": 0.0}
    
    # Use overlap logic to get words that intersect the window
    overlapping_words = slice_words_by_time(norm, start_s, end_s, pad_s)
    
    if overlapping_words:
        # Extract text from overlapping words
        text = " ".join(w["text"] for w in overlapping_words).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        
        # Calculate coverage
        coverage_s = (overlapping_words[-1]["end"] - overlapping_words[0]["start"]) if overlapping_words else 0.0
        
        metadata = {
            "start": start_s, 
            "end": end_s, 
            "pad": pad_s,
            "word_count": len(overlapping_words),
            "coverage_s": coverage_s
        }
        
        logger.info(f"CLIP_TRANSCRIPT: src=word_slice ({start_s:.2f}→{end_s:.2f}) words={len(overlapping_words)} chars={len(text)} coverage={coverage_s:.1f}s/{end_s-start_s:.1f}s")
        return text, "word_slice", metadata
    
    # Fallback A: Try segment-level text
    logger.warning(f"CLIP_TRANSCRIPT: words=0 using segment_span fallback for window [{start_s:.2f}, {end_s:.2f}]")
    
    # Try to get segments from source
    segments = getattr(source, "segments", None) or []
    if segments:
        overlapping_segments = slice_segments_by_time(segments, start_s, end_s, pad_s)
        if overlapping_segments:
            text = " ".join(s.get("text", "") for s in overlapping_segments).strip()
            if text:
                metadata = {
                    "start": start_s, 
                    "end": end_s, 
                    "pad": pad_s,
                    "word_count": 0,
                    "coverage_s": 0.0
                }
                return text, "segment_span", metadata
    
    # Fallback B: Use candidate snippet as last resort
    candidate_text = getattr(source, "text", "") or ""
    metadata = {
        "start": start_s, 
        "end": end_s, 
        "pad": pad_s,
        "word_count": 0,
        "coverage_s": 0.0
    }
    return candidate_text, "candidate_fallback", metadata

def build_clip_transcript_for_clip(episode, clip):
    """Wrapper for clips that handles the signature confusion"""
    start = clip.get("start")
    end = clip.get("end")
    
    if start is None or end is None:
        logger.warning(f"TRANSCRIPT_WARN: clip {clip.get('id', 'unknown')} has invalid timing: start={start}, end={end}")
        return "", "none", {"start": start, "end": end, "word_count": 0, "coverage_s": 0.0}
    
    txt, src, meta = build_clip_transcript_exact(episode, float(start), float(end))
    return txt, src, meta

# Legacy function for backward compatibility
def build_clip_transcript(episode, clip_start: float, clip_end: float):
    """Legacy function - now just calls the exact builder"""
    text, source = build_clip_transcript_exact(episode, clip_start, clip_end)
    return {"text": text, "source": source}
