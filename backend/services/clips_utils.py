"""
Clips Utilities - Cloud→Disk Fallback System
Handles Supabase clips with disk-based fallback for local development.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import os
import json

from config.settings import UPLOAD_DIR
UPLOADS = Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))

def _read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _norm_score(obj: Dict[str, Any]) -> float:
    # prefer normalized 0..1 if present; fallback to v4 fields
    for k in ("final_score", "score", "composite", "hook_v5_cal", "viral_score"):
        if k in obj and isinstance(obj[k], (int, float)):
            v = float(obj[k])
            # handle 0..100 style fields
            if "viral_score_100" in obj and isinstance(obj["viral_score_100"], (int, float)):
                return max(0.0, min(1.0, float(obj["viral_score_100"]) / 100.0))
            if v > 1.0 and v <= 100.0:
                return max(0.0, min(1.0, v / 100.0))
            return max(0.0, min(1.0, v))
    # explicit 0..100
    if "viral_score_100" in obj:
        return max(0.0, min(1.0, float(obj["viral_score_100"]) / 100.0))
    return 0.0

def _to_clip_obj(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    start = raw.get("start_time") or raw.get("start") or raw.get("t_start") or 0.0
    end   = raw.get("end_time")   or raw.get("end")   or raw.get("t_end")   or 0.0
    dur   = raw.get("duration")   or (float(end) - float(start)) if end and start else raw.get("len") or 0.0
    text  = raw.get("text") or raw.get("transcript") or raw.get("content") or ""
    reason = raw.get("reason") or raw.get("reasoning") or raw.get("why") or ""
    features = raw.get("features") or raw
    # include ad flags if present for the frontend filter
    is_ad = bool(raw.get("is_advertisement") or raw.get("_ad_flag") or False)

    return {
        "id": str(raw.get("id") or f"clip_{idx+1}"),
        "title": raw.get("title") or None,
        "text": text,
        "score": _norm_score(raw),
        "start_time": float(start) if start is not None else 0.0,
        "end_time": float(end) if end is not None else float(start) + float(dur),
        "duration": float(dur) if dur is not None else max(0.0, float(end) - float(start)),
        "reason": reason,
        "features": features,
        "is_advertisement": is_ad,
    }

def _find_clip_arrays(blob: Any) -> List[Dict[str, Any]]:
    """
    Try common shapes we've seen in your logs:
    - {"clips":[...]}
    - {"candidates":[...]}
    - {"combined_segments":[...]}  (scored segments)
    - plain list
    """
    if isinstance(blob, list):
        return [x for x in blob if isinstance(x, dict)]

    if not isinstance(blob, dict):
        return []

    for key in ("clips", "candidates", "combined_segments", "segments"):
        if key in blob and isinstance(blob[key], list):
            return [x for x in blob[key] if isinstance(x, dict)]

    # Sometimes transcript JSON nests data:
    data = blob.get("data") if isinstance(blob, dict) else None
    if isinstance(data, dict):
        for key in ("clips", "candidates", "combined_segments", "segments"):
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]

    return []

def load_clips_from_disk(episode_id: str) -> List[Dict[str, Any]]:
    transcript = UPLOADS / "transcripts" / f"{episode_id}.json"
    if not transcript.exists() or transcript.stat().st_size < 10:
        return []
    blob = _read_json(transcript)
    raw_list = _find_clip_arrays(blob)
    clips = [_to_clip_obj(obj, i) for i, obj in enumerate(raw_list)]
    # sort by score desc as a sane server-side default
    clips.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return clips
