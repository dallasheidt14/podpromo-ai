"""YouTube service: metadata gating, resilient download, and audio/video prep."""
from __future__ import annotations
import os, re, json, subprocess, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

from config.settings import UPLOAD_DIR
from services.progress_writer import write_progress
from utils.paths import safe_join, ensure_dir

log = logging.getLogger(__name__)

MIN_DURATION = 60
MAX_DURATION = 4 * 60 * 60  # 4h

YTDLP_COMMON = {
    "retries": 8,
    "fragment_retries": 8,
    "http_chunk_size": 10 * 1024 * 1024,  # 10 MB
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "user_agent": os.getenv(
        "YTDLP_UA",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ),
}

YT_REGEX = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/.+", re.I)

@dataclass
class VideoMeta:
    id: str
    title: str
    duration: int
    is_live: bool
    webpage_url: str

def _validate_url(url: str) -> bool:
    """Validate YouTube URL format."""
    return bool(YT_REGEX.match(url or ""))

def probe(url: str) -> VideoMeta:
    """Extract video metadata without downloading."""
    if not _validate_url(url):
        raise ValueError("invalid_url")
    
    with YoutubeDL({**YTDLP_COMMON}) as ydl:
        info = ydl.extract_info(url, download=False)
    
    duration = int(info.get("duration") or 0)
    is_live = bool(info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"})
    video_id = info.get("id") or ""
    title = info.get("title") or ""
    
    # Structured logging
    log.info("yt.probe_ok", extra={
        "video_id": video_id,
        "duration_sec": duration,
        "is_live": is_live,
        "title": title[:100]  # Truncate for logging
    })
    
    if is_live:
        raise ValueError("live_stream_not_supported")
    if duration < MIN_DURATION:
        raise ValueError("too_short")
    if duration > MAX_DURATION:
        raise ValueError("too_long")
    
    return VideoMeta(
        id=video_id,
        title=title,
        duration=duration,
        is_live=is_live,
        webpage_url=info.get("webpage_url") or url,
    )

def _paths(episode_id: str) -> Tuple[Path, Path]:
    """Get deterministic paths for video and audio files."""
    up = Path(UPLOAD_DIR)
    ensure_dir(up)
    return (up / f"{episode_id}.mp4", up / f"{episode_id}.wav")

def _ffmpeg(src: str, dst: str, extra_args: list[str]) -> None:
    """
    Cross-platform ffmpeg call using argv (no shell; no quoting issues on Windows).
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src, *extra_args, dst]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg_failed: {proc.stderr.decode('utf-8', errors='ignore')[:400]}")

def _ffmpeg_audio_to_mp4_audiogram(audio_src: str, dst: str) -> None:
    """
    Build an MP4 with a black video layer so previews can trim 'video' even if
    YouTube provided audio-only. The black canvas is long (15000s) and we use
    -shortest so output matches the audio duration (max audio is 4h).
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", audio_src,
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=15000",
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart",
        dst,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg_failed: {proc.stderr.decode('utf-8', errors='ignore')[:400]}")

def download_and_prepare(url: str, episode_id: str) -> Dict[str, Any]:
    """
    Download best video+audio merged to MP4, extract 16k/mono WAV for ASR.
    Falls back to audio-only if video is truly unavailable.
    """
    meta = probe(url)
    write_progress(episode_id, "fetching_metadata", 5, "Fetched video metadata")
    mp4_path, wav_path = _paths(episode_id)
    
    # Log download start
    log.info("yt.download.start", extra={
        "episode_id": episode_id,
        "video_id": meta.id,
        "duration_sec": meta.duration
    })

    # Primary attempt: video+audio merged MP4
    ytdlp_primary = {
        **YTDLP_COMMON,
        "format": "bv*+ba/b",                 # best video + best audio, else best
        "merge_output_format": "mp4",
        "outtmpl": str(mp4_path.with_suffix(".%(ext)s")),
    }

    # Fallback: audio-only to MP3 (we still render previews as audiogram)
    ytdlp_fallback = {
        **YTDLP_COMMON,
        "format": "bestaudio/best",
        "outtmpl": str(mp4_path.with_suffix(".%(ext)s")),  # may be .webm/.m4a
    }

    try:
        write_progress(episode_id, "downloading", 8, "Downloading video")
        with YoutubeDL(ytdlp_primary) as ydl:
            ydl.download([meta.webpage_url])

        # Normalize extension to .mp4 if merger picked different container
        # If file already .mp4, this is a no-op rename
        guessed = next(mp4_path.parent.glob(f"{episode_id}.*"), None)
        if guessed and guessed.suffix != ".mp4":
            # Convert container to mp4 without re-encode when possible
            tmp_mp4 = mp4_path.with_suffix(".mp4")
            _ffmpeg(str(guessed), str(tmp_mp4), ["-c", "copy"])
            guessed.unlink(missing_ok=True)
            tmp_mp4.rename(mp4_path)
        elif guessed and guessed.suffix == ".mp4":
            guessed.rename(mp4_path)
        else:
            # Shouldn't happen, but be defensive
            raise RuntimeError("download_no_output")

    except (DownloadError, ExtractorError) as e:
        log.warning("yt.download.fallback_audio_only", extra={
            "episode_id": episode_id,
            "video_id": meta.id,
            "reason": type(e).__name__
        })
        write_progress(episode_id, "downloading", 10, "Falling back to audio-only")
        with YoutubeDL(ytdlp_fallback) as ydl:
            ydl.download([meta.webpage_url])
        # Find produced audio and transcode to mp4 audiogram (black video)
        produced = next(mp4_path.parent.glob(f"{episode_id}.*"), None)
        if not produced:
            raise RuntimeError("download_failed")
        # Build an MP4 with black canvas so trims work as 'video'
        tmp_mp4 = mp4_path.with_suffix(".mp4")
        _ffmpeg_audio_to_mp4_audiogram(str(produced), str(tmp_mp4))
        produced.unlink(missing_ok=True)
        tmp_mp4.rename(mp4_path)

    write_progress(episode_id, "extracting_audio", 15, "Extracting audio for transcription")
    
    # Debug: Check if mp4 file exists and log paths
    log.info("yt.audio.extract_start", extra={
        "episode_id": episode_id,
        "mp4_path": str(mp4_path),
        "wav_path": str(wav_path),
        "mp4_exists": mp4_path.exists(),
        "mp4_size": mp4_path.stat().st_size if mp4_path.exists() else 0
    })
    
    _ffmpeg(str(mp4_path), str(wav_path), ["-ac", "1", "-ar", "16000"])
    
    # Log successful completion
    log.info("yt.audio.extract_ok", extra={
        "episode_id": episode_id,
        "video_id": meta.id,
        "video_path": str(mp4_path),
        "wav_path": str(wav_path)
    })
    
    return {
        "meta": {
            "id": meta.id,
            "title": meta.title,
            "duration": meta.duration,
            "url": meta.webpage_url,
        },
        "video_path": str(mp4_path),
        "wav_path": str(wav_path),
    }

def map_error(e: Exception) -> Tuple[int, str]:
    """Map exceptions to HTTP status codes and error strings."""
    if isinstance(e, ValueError):
        msg = str(e)
        if msg in {"invalid_url", "too_short", "too_long", "live_stream_not_supported"}:
            return 400, msg
    if isinstance(e, (DownloadError, ExtractorError)):
        return 502, "download_failed"
    if isinstance(e, RuntimeError) and "ffmpeg_failed" in str(e):
        return 500, "audio_conversion_failed"
    return 500, "internal_error"

# Backward compatibility functions
def is_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    return _validate_url(url)

def validate_youtube_url(url: str) -> Dict[str, Any]:
    """Validate YouTube URL and return validation result (legacy compatibility)."""
    try:
        meta = probe(url)
        return {
            "valid": True,
            "video_id": meta.id,
            "video_info": {
                "id": meta.id,
                "title": meta.title,
                "duration": meta.duration,
                "description": "",
                "uploader": "Unknown",
                "view_count": 0,
                "thumbnail": "",
            },
            "url": meta.webpage_url
        }
    except ValueError as e:
        return {"valid": False, "error": str(e)}
    except Exception as e:
        return {"valid": False, "error": "validation_failed"}

# Global instance for backward compatibility
youtube_service = type('YouTubeService', (), {
    'is_youtube_url': is_youtube_url,
    'validate_youtube_url': validate_youtube_url,
    'download_video': lambda self, video_id, episode_id: download_and_prepare(f"https://www.youtube.com/watch?v={video_id}", episode_id),
})()