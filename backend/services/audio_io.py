"""
Audio I/O utilities with ffmpeg pre-processing to eliminate mpg123 errors
"""
import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_pcm_wav(src_path: str) -> str:
    """
    Convert audio file to mono 16kHz WAV using ffmpeg.
    This eliminates mpg123 VBR/corrupt frame errors by pre-decoding to clean PCM.
    
    Args:
        src_path: Path to input audio file (MP3, WAV, etc.)
        
    Returns:
        Path to converted WAV file (cached if already exists)
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Audio file not found: {src_path}")
    
    # Create output path with .__asr16k.wav suffix
    base = src_path.with_suffix("")
    dst = base.with_suffix(".__asr16k.wav")
    
    # Return cached file if it exists
    if dst.exists():
        logger.debug(f"Using cached PCM WAV: {dst}")
        return str(dst)
    
    logger.info(f"Converting {src_path} to PCM WAV: {dst}")
    
    # ffmpeg command to convert to mono 16kHz WAV
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(src_path),
        "-ac", "1",        # mono
        "-ar", "16000",    # 16kHz sample rate
        "-f", "wav",       # WAV format
        "-y",              # overwrite output
        str(dst)
    ]
    
    try:
        # Run ffmpeg conversion
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully converted to PCM WAV: {dst}")
        return str(dst)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Failed to convert audio to PCM WAV: {e.stderr}")
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH. Please install ffmpeg.")
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to use audio pre-processing.")
    except Exception as e:
        logger.error(f"Unexpected error during audio conversion: {e}")
        raise RuntimeError(f"Audio conversion failed: {e}")

def cleanup_temp_audio(audio_path: str) -> None:
    """
    Clean up temporary audio files created by ensure_pcm_wav.
    
    Args:
        audio_path: Path to audio file to clean up
    """
    try:
        path = Path(audio_path)
        if path.exists() and path.suffix == ".__asr16k.wav":
            path.unlink()
            logger.debug(f"Cleaned up temporary audio file: {path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary audio file {audio_path}: {e}")

def is_audio_file(file_path: str) -> bool:
    """
    Check if file is a supported audio format.
    
    Args:
        file_path: Path to file to check
        
    Returns:
        True if file is a supported audio format
    """
    audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions

def get_audio_duration(file_path: str) -> float:
    """
    Get audio duration in seconds using ffprobe.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(file_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0
