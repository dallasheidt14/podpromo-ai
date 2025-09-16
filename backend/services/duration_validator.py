"""
Duration validation utilities to catch trimming issues early
"""
import subprocess
import json
import os
import logging

logger = logging.getLogger(__name__)

def ffprobe_duration_sec(path: str) -> float:
    """Get duration of media file using ffprobe"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            logger.error(f"ffprobe failed: {r.stderr}")
            return 0.0
            
        meta = json.loads(r.stdout or "{}")
        dur = meta.get("format", {}).get("duration")
        return float(dur) if dur else 0.0
    except Exception as e:
        logger.error(f"ffprobe error for {path}: {e}")
        return 0.0

def assert_reasonable_duration(path: str, min_sec: int = 120):
    """Validate that file duration matches its size - catches trimming issues"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    size_mb = os.path.getsize(path) / 1_000_000
    dur = ffprobe_duration_sec(path)
    
    logger.info(f"UPLOAD_CHECK: size={size_mb:.1f}MB dur={dur:.2f}s file={path}")
    
    if size_mb > 50 and dur < min_sec:
        raise ValueError(f"Suspicious media: {size_mb:.1f}MB but only {dur:.2f}s long. "
                        "Check trimming flags in convert/transcribe pipeline.")
    
    return dur
