# services/transcript_utils.py
from bisect import bisect_left, bisect_right
import re
from typing import List, Dict, Tuple

SPACE_FIXES = [
    (re.compile(r"\s+([.,!?;:])"), r"\1"),         # no space before punctuation
    (re.compile(r"\s+'"), " '"),                   # keep space before apostrophes if tokenized oddly
    (re.compile(r"\s+"), " "),                     # collapse whitespace
]

STOPWORDS_DISFLUENCY = {"uh", "um"}  # optional: drop filler if desired

def _normalize_text(s: str) -> str:
    s = s.strip()
    for rgx, rep in SPACE_FIXES:
        s = rgx.sub(rep, s)
    return s.strip()

def words_between(
    words: List[Dict], 
    start_sec: float, 
    end_sec: float, 
    tol: float = 0.08,
    drop_fillers: bool = True
) -> List[Dict]:
    """
    words: list of {"word": str, "start": float, "end": float} in episode time
    Returns only the words fully inside [start_sec - tol, end_sec + tol]
    """
    # Fast index by start time
    starts = [w["start"] for w in words]
    i0 = max(0, bisect_left(starts, start_sec - tol))
    i1 = min(len(words), bisect_right(starts, end_sec + tol))
    out = []
    for w in words[i0:i1]:
        ws, we = float(w["start"]), float(w["end"])
        if ws >= start_sec - tol and we <= end_sec + tol:
            if drop_fillers and w["word"].lower() in STOPWORDS_DISFLUENCY:
                continue
            out.append({"word": w["word"], "start": ws, "end": we})
    return out

def words_to_text(words: List[Dict]) -> str:
    # Join tokens, then fix spacing around punctuation
    text = " ".join(w["word"] for w in words)
    return _normalize_text(text)

def words_to_captions(words: List[Dict], clip_start: float) -> List[Dict]:
    """
    Returns per-word captions with time relative to the clip start.
    """
    caps = []
    for w in words:
        t0 = max(0.0, float(w["start"]) - clip_start)
        t1 = max(t0, float(w["end"]) - clip_start)
        caps.append({"t": round(t0, 3), "d": round(t1 - t0, 3), "w": w["word"]})
    return caps

def captions_to_vtt(caps: List[Dict], line_chars: int = 38) -> str:
    """
    Simple word-wrapping to WebVTT cues. One line ~38 chars; merge words until limit or time gap.
    """
    def fmt(ts: float) -> str:
        # WebVTT uses HH:MM:SS.mmm
        h = int(ts // 3600); ts -= 3600*h
        m = int(ts // 60);   ts -= 60*m
        s = int(ts); ms = int(round((ts - s)*1000))
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    cues = []
    if not caps:
        return "WEBVTT\n\n"

    chunk = []
    chunk_start = caps[0]["t"]
    current_len = 0
    for c in caps:
        gap = 0 if not chunk else c["t"] - (chunk[-1]["t"] + chunk[-1]["d"])
        word = c["w"]
        # new cue if too long or a noticeable gap (>0.35s)
        if current_len + 1 + len(word) > line_chars or gap > 0.35:
            if chunk:
                start = chunk_start
                end = chunk[-1]["t"] + chunk[-1]["d"]
                text = _normalize_text(" ".join(x["w"] for x in chunk))
                cues.append((start, end, text))
            chunk = [c]; current_len = len(word); chunk_start = c["t"]
        else:
            chunk.append(c)
            current_len += (1 + len(word))

    if chunk:
        start = chunk_start
        end = chunk[-1]["t"] + chunk[-1]["d"]
        text = _normalize_text(" ".join(x["w"] for x in chunk))
        cues.append((start, end, text))

    lines = ["WEBVTT", ""]
    for i, (s, e, t) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt(s)} --> {fmt(e)}")
        lines.append(t)
        lines.append("")
    return "\n".join(lines)
