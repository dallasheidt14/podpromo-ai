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

# Valid spectrum settings
SPECTRUM_FILTER = (
    "showspectrum=s=640x360:mode=combined:color=intensity:scale=cbrt,"
    "format=yuv420p"
)

# Nice looking, widely compatible waveform (line) with A/V sync
WAVES_FILTER = (
    "aformat=channel_layouts=stereo,"
    "aresample=async=1:first_pts=0,"     # keeps A/V in sync
    "showwaves=s=640x360:mode=line:rate=25,"
    "format=yuv420p"
)

def run(cmd):
    timeout_sec = int(os.getenv("FFMPEG_TIMEOUT_SEC", "900"))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
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

def _ffmpeg_preview_args(in_wav: str, out_mp4: str, ss: float, dur: float, filter_expr: str):
    """Build FFmpeg arguments for preview generation with proper audio mapping"""
    return [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-y",
        "-ss", f"{ss}", "-t", f"{dur}",            # trim input (applies to both A & V)
        "-i", in_wav,
        "-filter_complex", f"[0:a]{filter_expr}[v]",  # build the video from audio
        "-map", "[v]", "-map", "0:a",              # ALWAYS include the audio stream
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-shortest",                                # end when either stream ends
        "-movflags", "+faststart",                  # web-friendly
        out_mp4,
    ]

def build_preview_filename(episode_id: str, clip_id: str) -> str:
    """Generate a stable filename for caching and re-use."""
    return f"{_hash(episode_id, clip_id)}.m4a"


def _hash_name(episode_id: str, clip_id: str, start: float, end: float) -> str:
    """Generate a hash-based filename for previews."""
    key = f"{episode_id}:{clip_id}:{round(start,2)}:{round(end,2)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

def _run_ffmpeg(argv: list[str]) -> None:
    """Run ffmpeg command with error handling."""
    # Validate that any "-i <path>" inputs actually exist (skip filters/flags)
    for i, arg in enumerate(argv):
        if arg == "-i" and i + 1 < len(argv):
            in_path = argv[i + 1]
            # If it's a real filesystem path (not a lavfi spec), ensure it exists
            if (":" in in_path or "/" in in_path or "\\" in in_path) and not in_path.startswith("color="):
                if not os.path.exists(in_path):
                    raise ValueError(f"Input file not found: {in_path}")
    
    timeout_sec = int(os.getenv("FFMPEG_TIMEOUT_SEC", "900"))  # default 15 min
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
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
            return out_path.name  # Return filename only, not URL path

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
            # Audio-only → audiogram with fallback system
            # 1) Try waveform lines (most compatible)
            try:
                _run_ffmpeg(_ffmpeg_preview_args(src, dst, start_sec, duration, WAVES_FILTER))
                return out_path.name  # Return filename only, not URL path
            except Exception as e1:
                logger.warning("Waveform preview failed, trying spectrum. Reason: %s", e1)

            # 2) Try spectrum (valid params)
            try:
                _run_ffmpeg(_ffmpeg_preview_args(src, dst, start_sec, duration, SPECTRUM_FILTER))
                return out_path.name  # Return filename only, not URL path
            except Exception as e2:
                logger.error("Spectrum preview failed: %s", e2)
                return None
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
