# backend/services/preview_service.py
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
import hashlib
import os

# Get directories from environment or default
from config.settings import UPLOAD_DIR, OUTPUT_DIR
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", OUTPUT_DIR))
PREVIEWS_DIR = OUTPUTS_DIR / "previews"

# Ensure previews directory exists
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

def _hash(*parts: str) -> str:
    """Generate a stable hash for consistent filenames."""
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]

def build_preview_filename(episode_id: str, clip_id: str) -> str:
    """Generate a stable filename for caching and re-use."""
    return f"{_hash(episode_id, clip_id)}.m4a"

def ensure_preview(
    *,
    source_media: Path,
    episode_id: str,
    clip_id: str,
    start_sec: float,
    end_sec: float,
    max_preview_sec: float = 20.0,   # keep fast; adjust to taste
    pad_start_sec: float = 0.0,      # set to 0.5–1.0 if you want a tiny lead-in
    audio_bitrate: str = "96k",
    sample_rate: int = 44100,
) -> Optional[str]:
    """
    Extracts a short audio-only preview for the clip and returns a relative URL.
    Returns None on failure (non-fatal).
    """
    try:
        if not source_media.exists():
            return None
            
        start = max(0.0, start_sec - pad_start_sec)
        full = max(0.0, end_sec - start_sec)
        dur = min(max_preview_sec, full if full > 0 else max_preview_sec)

        out_name = build_preview_filename(episode_id, clip_id)
        out_path = PREVIEWS_DIR / out_name

        # Return existing preview if it exists and has content
        if out_path.exists() and out_path.stat().st_size > 0:
            return f"/clips/previews/{out_name}"

        # Generate new preview with ffmpeg
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output files
            "-ss", f"{start:.3f}",  # start time
            "-t", f"{dur:.3f}",     # duration
            "-i", str(source_media),  # input file
            "-vn",                  # no video
            "-acodec", "aac",       # m4a container
            "-b:a", audio_bitrate,  # audio bitrate
            "-ar", str(sample_rate), # sample rate
            "-ac", "2",             # stereo
            str(out_path),
        ]
        
        # Run ffmpeg command
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=30  # prevent hanging
        )
        
        if out_path.exists() and out_path.stat().st_size > 0:
            return f"/clips/previews/{out_name}"
        else:
            return None
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"[PREVIEW] Failed to generate preview for {episode_id}/{clip_id}: {e}")
        return None
    except Exception as e:
        print(f"[PREVIEW] Unexpected error generating preview: {e}")
        return None

def get_episode_media_path(episode_id: str) -> Optional[Path]:
    """Get the path to the episode's audio file."""
    # Try common audio extensions
    for ext in ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi']:
        path = UPLOADS_DIR / f"{episode_id}{ext}"
        if path.exists():
            return path
    return None
