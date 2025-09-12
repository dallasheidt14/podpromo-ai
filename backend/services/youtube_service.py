import os, logging, re, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

logger = logging.getLogger(__name__)

@dataclass
class VideoMeta:
    id: str
    title: str
    duration: float
    url: str

# Configuration (keep your defaults)
MIN_DURATION_SEC = int(os.getenv("YOUTUBE_MIN_DURATION", "60"))
MAX_DURATION_SEC = int(os.getenv("YOUTUBE_MAX_DURATION", "14400"))  # 4h limit per your constraints
YOUTUBE_TIMEOUT = int(os.getenv("YOUTUBE_TIMEOUT", "30"))
FFMPEG_TIMEOUT_SEC = int(os.getenv("FFMPEG_TIMEOUT_SEC", "900"))  # 15 min floor

# Optional cookies strategy (admin-configured)
BROWSER_SPEC = os.getenv("YT_COOKIES_FROM_BROWSER", "").strip()  # e.g. "chrome:Default"
COOKIE_FILE = os.getenv("YT_COOKIES_FILE", "").strip()
CLIENTS = [c.strip() for c in os.getenv("YT_PLAYER_CLIENTS", "web,android").split(",") if c.strip()]

YT_REGEX = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", re.I)

def _validate_url(url: str) -> bool:
    """Validate YouTube URL format"""
    if not url or not isinstance(url, str):
        return False
    youtube_patterns = [
        "youtube.com/watch",
        "youtu.be/",
        "youtube.com/embed/",
        "youtube.com/v/",
    ]
    return any(pattern in url.lower() for pattern in youtube_patterns)

def _ydl_opts_base() -> dict:
    """Base yt-dlp options shared by all attempts."""
    return {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'socket_timeout': YOUTUBE_TIMEOUT,
        'format': 'best[height<=720]/best',
        'writesubtitles': False,
        'writeautomaticsub': False,
        # Safe headers to look like a real browser
        'http_headers': {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/122.0 Safari/537.36'),
            'Accept-Language': 'en-US,en;q=0.9',
        },
        # Error handling
        'ignoreerrors': False,
        'retries': 3,
    }

def _player_client_opts(client: str) -> Dict:
    """Extractor args to try different player clients (android often bypasses gates)."""
    client = client.strip().lower()
    if client not in {"web", "android", "ios", "tv"}:
        client = "web"
    return {
        'extractor_args': {
            'youtube': {
                'player_client': [client],
            }
        }
    }

def _cookie_source():
    """Get cookie source info for admin configuration"""
    f = COOKIE_FILE
    b = BROWSER_SPEC
    if f and os.path.exists(f):
        return ("file", f)
    if b:
        return ("browser", b)
    return (None, None)

def _opts_for(mode: str) -> dict:
    """Get yt-dlp options for a specific mode"""
    o = _ydl_opts_base()
    if mode == "web_no_cookies":
        pass
    elif mode == "tv_no_cookies":
        # TV client often bypasses bot/age checks without cookies
        o.setdefault("extractor_args", {}).setdefault("youtube", {})["player_client"] = "tv_embedded"
    elif mode == "android_no_cookies":
        o.setdefault("extractor_args", {}).setdefault("youtube", {})["player_client"] = "android"
    elif mode == "android_with_cookies":
        kind, src = _cookie_source()
        o.setdefault("extractor_args", {}).setdefault("youtube", {})["player_client"] = "android"
        if kind == "file":
            o["cookiefile"] = src
        elif kind == "browser":
            o["cookiesfrombrowser"] = (src,)
    return o

_COOKIE_ERRORS = (
    "confirm you're not a bot",
    "confirm you're not a bot",
    "Sign in to confirm",
    "age verification",
    "This video is only available to signed-in users",
    "requires login",
    "not available without signing in",
)

def _needs_cookies(err_text: str) -> bool:
    t = (err_text or "").lower()
    return any(s.lower() in t for s in _COOKIE_ERRORS)

def probe(url: str) -> VideoMeta:
    """Extract video metadata with retry ladder."""
    if not _validate_url(url):
        raise ValueError("invalid_url")
    
    modes = ["web_no_cookies", "tv_no_cookies", "android_no_cookies"]
    kind, _ = _cookie_source()
    if kind:  # only add this step if a cookie source exists
        modes.append("android_with_cookies")

    last_err = None
    for mode in modes:
        logger.info(f"Probing YouTube URL with mode={mode}")
        try:
            with YoutubeDL(_opts_for(mode)) as ydl:
                info = ydl.extract_info(url, download=False)
            if info.get('is_live'):
                raise ValueError("live_stream_not_supported")
            duration = info.get('duration', 0) or 0
            if duration < MIN_DURATION_SEC:
                raise ValueError("too_short")
            if duration > MAX_DURATION_SEC:
                raise ValueError("too_long")
            logger.info("YouTube probe successful [%s]: %s (%.2fs)", mode, info['title'], duration)
            return VideoMeta(id=info['id'], title=info['title'], duration=duration, url=info['webpage_url'])
        except (DownloadError, ExtractorError) as e:
            last_err = str(e)
            logger.warning("Probe attempt failed [%s]: %s", mode, last_err)
            continue
        except ValueError:
            raise
        except Exception as e:
            last_err = str(e)
            logger.error("Probe error [%s]: %s", mode, last_err)
            continue

    # All attempts failed
    if kind:
        raise ValueError("download_failed")  # cookies were available but still failed
    raise ValueError("download_failed_requires_cookies")

def download_and_prepare(url: str, episode_id: str) -> dict:
    """Download video and prepare files with the same retry ladder."""
    from config.settings import UPLOAD_DIR
    meta = probe(url)  # validates duration & builds best mode implicitly

    # Use the same modes as probe
    modes = ["web_no_cookies", "tv_no_cookies", "android_no_cookies"]
    kind, _ = _cookie_source()
    if kind:  # only add this step if a cookie source exists
        modes.append("android_with_cookies")

    last_err = None
    for mode in modes:
        try:
            logger.info("Downloading YouTube video: %s [%s]", meta.title, mode)
            opts = _opts_for(mode)
            opts.update({
                'outtmpl': f'{UPLOAD_DIR}/{episode_id}.%(ext)s',
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
            })
            
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            # Find downloaded container
            video_path: Optional[Path] = None
            for ext in ['mp4', 'webm', 'mkv']:
                candidate = Path(f"{UPLOAD_DIR}/{episode_id}.{ext}")
                if candidate.exists():
                    video_path = candidate
                    break
            if not video_path:
                raise RuntimeError("video_file_missing")
            # Extract audio
            audio_path = Path(f"{UPLOAD_DIR}/{episode_id}.wav")
            _extract_audio(video_path, audio_path)
            return {
                "meta": {"id": info['id'], "title": info['title'], "duration": info.get('duration', meta.duration)},
                "video_path": str(video_path),
                "wav_path": str(audio_path),
            }
        except (DownloadError, ExtractorError) as e:
            last_err = str(e)
            logger.warning("Download attempt failed [%s]: %s", mode, last_err)
            continue
        except Exception as e:
            last_err = str(e)
            logger.error("Download error [%s]: %s", mode, last_err)
            continue
    # Out of attempts
    if kind:
        raise RuntimeError("download_failed")  # cookies were available but still failed
    raise RuntimeError("download_failed_requires_cookies")

def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio using ffmpeg"""
    import subprocess
    
    argv = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path), "-y"
    ]
    
    try:
        # Dynamic timeout based on content length
        timeout_sec = max(FFMPEG_TIMEOUT_SEC, 900)  # 15 min floor
        subprocess.run(argv, capture_output=True, check=True, timeout=timeout_sec)
        logger.info(f"Audio extraction successful: {audio_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        raise RuntimeError(f"ffmpeg_failed: {e}")
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out")
        raise RuntimeError("ffmpeg_timeout")

def map_error(exc: Exception) -> Tuple[int, str]:
    """Map exceptions to HTTP status codes"""
    if isinstance(exc, ValueError):
        error_map = {
            "invalid_url": (400, "invalid_url"),
            "too_short": (400, "too_short"),
            "too_long": (400, "too_long"),
            "live_stream_not_supported": (400, "live_stream_not_supported"),
            "download_failed_requires_cookies": (402, "download_requires_login"),
            "download_failed": (502, "download_failed"),
            "probe_failed": (502, "probe_failed"),
        }
        return error_map.get(str(exc), (400, "invalid_request"))

    elif isinstance(exc, (DownloadError, ExtractorError)):
        msg = str(exc)
        if _needs_cookies(msg):
            return (402, "download_requires_login")
        return (502, "download_failed")

    elif isinstance(exc, RuntimeError):
        msg = str(exc)
        if "download_failed_requires_cookies" in msg:
            return (402, "download_requires_login")
        if "ffmpeg_failed" in msg:
            return (500, "audio_conversion_failed")
        if "download_failed" in msg:
            return (502, "download_failed")
        return (500, "internal_error")

    else:
        return (500, "internal_error")