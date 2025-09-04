"""
Progress Utilities - Cloud→Local Fallback System
Handles Supabase progress tracking with disk-based fallback for local development.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import os
from typing import Optional, Dict, Any, Tuple

# ---------- Config ----------
UPLOADS = Path(os.getenv("UPLOADS_DIR", r"C:\Users\Dallas Heidt\Desktop\podpromo\backend\uploads"))
SUPABASE_URL = os.getenv("SUPABASE_URL") or ""
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "episodes")

# ---------- Supabase (optional) ----------
def _get_supabase_client():
    """
    Returns a Supabase client if SUPABASE_URL/KEY are set and library is installed; else None.
    """
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    try:
        from supabase import create_client, Client  # type: ignore
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

def supabase_fetch_progress(episode_id: str) -> Optional[Dict[str, Any]]:
    """
    Try to read progress from Supabase.
    Expecting a row with columns like: episode_id, status, percentage, stage, updated_at
    Returns a normalized dict or None.
    """
    sb = _get_supabase_client()
    if sb is None:
        return None
    try:
        res = (
            sb.table(SUPABASE_TABLE)
              .select("*")
              .eq("episode_id", episode_id)
              .limit(1)
              .maybe_single()
              .execute()
        )
        row = getattr(res, "data", None) or res  # supports both SDK response shapes
        if not row:
            return None

        # Normalize/defend against missing fields
        status = str(row.get("status") or "").lower() or "processing"
        percentage = int(row.get("percentage") or (100 if status == "completed" else 0))
        stage = str(row.get("stage") or status or "processing").lower()
        updated_at = row.get("updated_at") or datetime.now(timezone.utc).isoformat()

        return {
            "ok": True,
            "progress": {
                "percentage": percentage,
                "stage": stage,
                "message": f"Supabase: {stage}",
                "timestamp": updated_at,
            },
            "status": status,
            "_source": "supabase",
        }
    except Exception:
        return None

def supabase_upsert_progress(payload: Dict[str, Any]) -> None:
    """
    Best-effort upsert to keep cloud in sync when disk says 'completed'.
    Never raises to caller.
    """
    sb = _get_supabase_client()
    if sb is None:
        return
    try:
        row = {
            "episode_id": payload.get("episode_id"),
            "status": payload.get("status", "completed"),
            "percentage": payload.get("progress", {}).get("percentage", 100),
            "stage": payload.get("progress", {}).get("stage", "completed"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        sb.table(SUPABASE_TABLE).upsert(row, returning="minimal").execute()
    except Exception:
        pass

# ---------- Disk inference ----------
def _episode_paths(episode_id: str) -> Tuple[Path, Path]:
    audio = UPLOADS / f"{episode_id}.mp3"
    transcript = UPLOADS / "transcripts" / f"{episode_id}.json"
    return audio, transcript

def _safe_json_len(p: Path) -> int:
    try:
        if not p.exists():
            return 0
        if p.stat().st_size < 10:
            return 0
        # Optional: verify it's valid JSON
        with p.open("r", encoding="utf-8") as f:
            json.load(f)
        return 1
    except Exception:
        # If not valid JSON, treat as missing
        return 0

def disk_infer_progress(episode_id: str) -> Dict[str, Any]:
    """
    Inference order:
    - transcript exists → completed/100%
    - audio exists → queued/10%
    - nothing → unknown
    """
    audio, transcript = _episode_paths(episode_id)

    if _safe_json_len(transcript) > 0:
        return {
            "ok": True,
            "progress": {
                "percentage": 100,
                "stage": "completed",
                "message": "Completed (inferred from transcript on disk)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "status": "completed",
            "episode_id": episode_id,
            "_source": "disk",
        }

    if audio.exists() and audio.stat().st_size > 10:
        return {
            "ok": True,
            "progress": {
                "percentage": 10,
                "stage": "queued",
                "message": "Audio present; waiting/transcribing (inferred from disk)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "status": "queued",
            "episode_id": episode_id,
            "_source": "disk",
        }

    return {
        "ok": False,
        "progress": {
            "percentage": 0,
            "stage": "unknown",
            "message": "Episode not found on disk",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "status": "unknown",
        "episode_id": episode_id,
        "_source": "disk",
    }

# ---------- Normalizer to your frontend format ----------
def normalize_progress_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure we always return:
    { ok, progress: { percentage, stage, message, timestamp }, status }
    """
    ok = bool(p.get("ok", True))
    progress = p.get("progress") or {}
    status = (p.get("status") or progress.get("stage") or "processing").lower()
    # Guarantee required fields
    progress.setdefault("percentage", 0)
    progress.setdefault("stage", status)
    progress.setdefault("message", "")
    progress.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    return {
        "ok": ok,
        "progress": progress,
        "status": status,
    }
