"""
Exact transcript building for clips - matches audio window precisely
"""
import re
import logging

logger = logging.getLogger(__name__)

def build_clip_transcript_exact(source, start_s: float, end_s: float, eps: float = 0.05) -> tuple[str, str]:
    """
    Build transcript from word timestamps inside the exact clip window.
    source can be episode object or list[dict] of words.
    Returns (text, source_type)
    """
    # Handle None inputs
    if start_s is None or end_s is None:
        return "", "none"
    
    # Get words from source (episode or list)
    words = getattr(source, "words", None) or source or []
    
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
        logger.debug(f"TRANSCRIPT_DEBUG: no valid words found for window [{start_s:.2f}, {end_s:.2f}]")
        return "", "none"
    
    # Include any word that overlaps [start_s, end_s] with epsilon
    out = []
    for w in norm:
        if w["end"] < start_s - eps:
            continue
        if w["start"] > end_s + eps:
            break
        out.append(w["text"])
    
    text = " ".join(out).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    
    source_type = "word_slice" if text else "none"
    logger.debug(f"TRANSCRIPT_DEBUG: found {len(out)} words, text_len={len(text)}")
    return text, source_type

def build_clip_transcript_for_clip(episode, clip):
    """Wrapper for clips that handles the signature confusion"""
    start = clip.get("start")
    end = clip.get("end")
    
    if start is None or end is None:
        logger.warning(f"TRANSCRIPT_WARN: clip {clip.get('id', 'unknown')} has invalid timing: start={start}, end={end}")
        return "", "none"
    
    txt, src = build_clip_transcript_exact(episode, float(start), float(end))
    return txt, src

# Legacy function for backward compatibility
def build_clip_transcript(episode, clip_start: float, clip_end: float):
    """Legacy function - now just calls the exact builder"""
    text, source = build_clip_transcript_exact(episode, clip_start, clip_end)
    return {"text": text, "source": source}
