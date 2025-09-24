# services/asr_helpers.py
from typing import Dict, Any, List

def sanitize_ct2_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize CTranslate2 kwargs to prevent type errors"""
    clean = {}
    for k, v in kwargs.items():
        if v is None: 
            continue
        if k in {"beam_size","num_hypotheses","no_repeat_ngram_size","max_length","max_initial_timestamp_index"}:
            clean[k] = int(v)
        elif k in {"patience","length_penalty","repetition_penalty","sampling_temperature"}:
            clean[k] = float(v)
        elif k in {"asynchronous","return_scores","return_logits_vocab","return_no_speech_prob","suppress_blank"}:
            clean[k] = bool(v)
        elif k == "suppress_tokens":
            clean[k] = [-1] if v == -1 else [int(x) for x in (v or [])]
        else:
            clean[k] = v
    return clean

def synthesize_words_from_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create word spans if word_timestamps weren't returned.
    We do a naive split, evenly spaced across the segment window.
    """
    out: List[Dict[str, Any]] = []
    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        if not text: 
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.01))
        toks = [t for t in text.split() if t]
        n = max(1, len(toks))
        dur = max(0.01, end - start)
        step = dur / n
        for i, tok in enumerate(toks):
            ws = start + i * step
            we = start + (i + 1) * step
            out.append({"start": ws, "end": we, "text": tok})
    return out

def ensure_words(segments: List[Dict[str, Any]], words: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Ensure words exist - synthesize if missing"""
    return words if (isinstance(words, list) and len(words) > 0) else synthesize_words_from_segments(segments)
