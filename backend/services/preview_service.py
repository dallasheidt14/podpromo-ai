# backend/services/preview_service.py
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
import hashlib
import os
import logging
import json

# Get directories from environment or default
from config.settings import UPLOAD_DIR, OUTPUT_DIR
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", OUTPUT_DIR))
PREVIEWS_DIR = OUTPUTS_DIR / "previews"

# Ensure previews directory exists
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

DEFAULT_PAD_START_SEC = 0.08   # -80ms
DEFAULT_PAD_END_SEC   = 0.24   # +240ms

def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg/ffprobe failed: {proc.stderr.strip()}")
    return proc.stdout

def ffprobe_duration(path: str) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path
    ])
    data = json.loads(out)
    dur = float(data.get("format", {}).get("duration", 0.0))
    return max(dur, 0.0)

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
    pad_start_sec: float = None,     # -80ms default
    pad_end_sec: float = None,       # +240ms default
    audio_bitrate: str = "96k",
    sample_rate: int = 44100,
    **kwargs,  # swallow unknown future options to avoid breaking callers
) -> Optional[str]:
    """
    Create a trimmed audio preview with tiny safety pads and return relative URL.
    - Uses accurate trimming (-i then -ss/-to).
    - Pads are configurable; defaults are conservative for AAC encoder delay.
    """
    try:
        if not source_media.exists():
            return None

        # Allow legacy arg names (defensive)
        if "pad_start" in kwargs and pad_start_sec is None:
            pad_start_sec = kwargs["pad_start"]
        if "pad_end" in kwargs and pad_end_sec is None:
            pad_end_sec = kwargs["pad_end"]

        pad_start = DEFAULT_PAD_START_SEC if pad_start_sec is None else float(pad_start_sec)
        pad_end   = DEFAULT_PAD_END_SEC   if pad_end_sec   is None else float(pad_end_sec)

        # Compute padded boundaries with explicit duration math
        window_dur = float(end_sec) - float(start_sec)
        
        # Our intended file length including pads
        max_output_dur = window_dur + pad_start + pad_end
        
        # If caller gives max_preview_sec, treat it as the UNPADDED window target and add pads
        if max_preview_sec is not None:
            max_output_dur = float(max_preview_sec) + pad_start + pad_end
        
        cut_start = max(0.0, float(start_sec) - pad_start)
        cut_end   = min(float(end_sec) + pad_end, cut_start + max_output_dur)
        
        # Clamp end to prevent overrun (hard cap after padding)
        if max_preview_sec is not None:
            cut_end = min(cut_end, cut_start + max_preview_sec)

        out_name = build_preview_filename(episode_id, clip_id)
        out_path = PREVIEWS_DIR / out_name

        # Return existing preview if it exists and has content
        if out_path.exists() and out_path.stat().st_size > 0:
            return f"/clips/previews/{out_name}"

        # Accurate trim: input first, then -ss/-to
        cmd = [
            "ffmpeg", "-y",
            "-i", str(source_media),
            "-ss", f"{cut_start:.3f}",
            "-to", f"{cut_end:.3f}",
            "-map", "0:a:0",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", str(sample_rate),
            "-ac", "2",
            "-movflags", "+faststart",
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
            # Get actual duration from ffprobe
            try:
                dur = ffprobe_duration(str(out_path))
                
                # Helpful metrics for logs and UI guards
                intended_unpadded = window_dur
                overrun_ms = int(round((dur - intended_unpadded) * 1000))
                
                logger.info(
                    "PREVIEW: start=%.3f, end=%.3f, dur=%.3f, max_preview_sec=%s, overrun_ms=%d",
                    start_sec, end_sec, dur,
                    f"{max_preview_sec:.3f}" if max_preview_sec else "None",
                    overrun_ms
                )
            except Exception:
                dur = cut_end - cut_start
                intended_unpadded = window_dur
                overrun_ms = int(round((dur - intended_unpadded) * 1000))
                logger.info(f"PREVIEW: start={start_sec:.3f}, end={end_sec:.3f}, dur={dur:.3f} (estimated), overrun_ms={overrun_ms}")
            
            return f"/clips/previews/{out_name}"
        else:
            return None
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.error(f"Preview generation failed for {episode_id}/{clip_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating preview: {e}", exc_info=True)
        return None

def _hash_name(episode_id: str, clip_id: str, start: float, end: float) -> str:
    """Generate a hash-based filename for previews."""
    key = f"{episode_id}:{clip_id}:{round(start,2)}:{round(end,2)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

def _run_ffmpeg(argv: list[str]) -> None:
    """Run ffmpeg command with error handling."""
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[:400])

def ensure_preview(
    *,
    source_media: Path,
    episode_id: str,
    clip_id: str,
    start_sec: float,
    end_sec: float,
    max_preview_sec: float = 20.0,
    audio_bitrate: str = "128k",
    sample_rate: int = 44100,
    **kwargs,
) -> Optional[str]:
    """
    Create (or reuse) an H.264/AAC MP4 preview and return its public URL.
    Always produces MP4 so <video> works in the browser.
    """
    try:
        PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        duration = max(0.2, min(max_preview_sec, float(end_sec) - float(start_sec)))
        name = _hash_name(episode_id, clip_id, start_sec, start_sec + duration)
        out_path = PREVIEWS_DIR / f"{episode_id}_{clip_id}_{name}.mp4"
        if out_path.exists():
            return f"/clips/previews/{out_path.name}"

        src = str(source_media)
        dst = str(out_path)
        # Detect if source is video (we'll assume video if not pure audio extension)
        is_video = not source_media.suffix.lower() in {".wav", ".mp3", ".m4a"}

        if is_video:
            # Trim video + audio; faststart for streaming
            argv = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_sec:.3f}", "-i", src, "-t", f"{duration:.3f}",
                "-vf", "scale='min(1280,iw)':-2,format=yuv420p",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(sample_rate),
                "-movflags", "+faststart", dst,
            ]
        else:
            # Audio-only → audiogram on black canvas
            argv = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_sec:.3f}", "-i", src, "-t", f"{duration:.3f}",
                "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration:.3f}",
                "-lavfi", "showspectrum=mode=lines:color=intensity:slide=scroll:scale=log",
                "-shortest",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(sample_rate),
                "-movflags", "+faststart", dst,
            ]
        _run_ffmpeg(argv)
        return f"/clips/previews/{out_path.name}"
    except Exception as e:
        logger.error(f"Preview generation failed for {episode_id}/{clip_id}: {e}", exc_info=True)
        return None

def get_episode_media_path(episode_id: str) -> Optional[Path]:
    """Get the path to the episode's media file (backward-compatible resolver)."""
    # New flat structure: backend/uploads/<episode_id>.{ext}
    for ext in ['.mp4', '.mp3', '.wav', '.m4a', '.mov', '.avi']:
        path = UPLOADS_DIR / f"{episode_id}{ext}"
        if path.exists():
            return path
    
    # Legacy directory structure: backend/uploads/<episode_id>/{source.mp3|input.wav|input.mp3}
    episode_dir = UPLOADS_DIR / episode_id
    if episode_dir.exists():
        for filename in ['source.mp3', 'input.wav', 'input.mp3', 'audio.mp3', 'audio.wav']:
            path = episode_dir / filename
            if path.exists():
                return path
    
    return None
