"""
Application settings and configuration
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories - moved outside source tree to prevent file watcher restarts
# Use environment variables with sensible defaults for development
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs"))
LOGS_DIR = os.getenv("LOGS_DIR", str(BASE_DIR / "logs"))
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", str(Path(UPLOAD_DIR) / "transcripts"))
CLIPS_DIR = os.getenv("CLIPS_DIR", str(Path(OUTPUT_DIR) / "clips"))

# Audio settings
SAMPLE_RATE = 16000  # Hz
WHISPER_LANGUAGE = "en"  # Language for Whisper transcription

# File settings - now configurable via environment
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_BYTES", "524288000"))  # 500MB default
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".mp4"}

# Progress tracking settings
PROGRESS_TRACKER_TTL = 300  # 5 minutes in seconds

# Whisper settings
WHISPER_MODEL = "base"  # Model size: tiny, base, small, medium, large

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "supabase")

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Preview settings (secure signed URLs)
PREVIEW_FS_DIR = os.getenv("PREVIEW_FS_DIR", str(Path(OUTPUT_DIR) / "previews"))
PREVIEW_SIGNING_KEY = os.getenv("PREVIEW_SIGNING_KEY", SECRET_KEY)  # Use SECRET_KEY as fallback
PREVIEW_URL_TTL_SECONDS = int(os.getenv("PREVIEW_URL_TTL_SECONDS", "600"))  # 10 minutes

# Clip length bounds (hard limits)
CLIP_LEN_MIN = float(os.getenv("CLIP_LEN_MIN", "8.0"))
CLIP_LEN_MAX = float(os.getenv("CLIP_LEN_MAX", "90.0"))

# Soft relax settings for empty results
FT_SOFT_RELAX_ON_ZERO = int(os.getenv("FT_SOFT_RELAX_ON_ZERO", "1"))
FT_SOFT_RELAX_TOPK = int(os.getenv("FT_SOFT_RELAX_TOPK", "3"))

# Ad detection and filtering
ALLOW_ADS = int(os.getenv("ALLOW_ADS", "0"))  # block ads by default
AD_SIG_MIN = float(os.getenv("AD_SIG_MIN", "0.50"))  # >= 0.50 → ad=True

# Hook score clamping
HOOK_AD_CLAMP_MIN = float(os.getenv("HOOK_AD_CLAMP_MIN", "0.40"))  # if ad_likelihood >= this, clamp hook
HOOK_AD_CLAMP = float(os.getenv("HOOK_AD_CLAMP", "0.30"))  # clamp hook to this max
REP_WINDOW_CHARS = int(os.getenv("REP_WINDOW_CHARS", "200"))  # opener window for repetition ratio
HOOK_REP_MIN_RATIO = float(os.getenv("HOOK_REP_MIN_RATIO", "0.30"))  # if repetition >= this, penalize
HOOK_REP_PENALTY_MULT = float(os.getenv("HOOK_REP_PENALTY_MULT", "0.70"))  # multiply hook by this when penalizing

# Question-only rule
Q_ONLY_MIN_Q = float(os.getenv("Q_ONLY_MIN_Q", "0.50"))  # treat as Q/list if >=
Q_ONLY_MAX_PAYOFF = float(os.getenv("Q_ONLY_MAX_PAYOFF", "0.15"))  # drop if payoff below this

# Logging flags
LOG_AD_SIGNALS = int(os.getenv("LOG_AD_SIGNALS", "1"))

# Trail padding and refinement settings
HEAD_PAD_SEC = float(os.getenv("HEAD_PAD_SEC", "0.05"))
TRAIL_PAD_SEC = float(os.getenv("TRAIL_PAD_SEC", "0.25"))
REFINE_SNAP_MAX_NUDGE = float(os.getenv("REFINE_SNAP_MAX_NUDGE", "0.35"))
REFINE_MIN_TAIL_SILENCE = float(os.getenv("REFINE_MIN_TAIL_SILENCE", "0.12"))

# Title generation settings
PLAT_LIMITS = {
    "shorts": 80,
    "reels": 80,
    "tiktok": 80,
    "neutral": 80,
    "default": 80,
}
TITLE_ENGINE_V2 = os.getenv("TITLE_ENGINE_V2", "true").lower() == "true"

# API settings
API_PREFIX = "/api"

# CORS settings - environment-driven
import json
ENV = os.getenv("ENV", "dev")
if ENV == "production":
    CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS_JSON", '["https://app.yourdomain.com"]'))
else:
    CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS_JSON", '["http://localhost:3000", "http://127.0.0.1:3000"]'))

# Monitoring settings
METRICS_INTERVAL = 60  # seconds
CLEANUP_INTERVAL = 300  # seconds

# Two-stage scoring optimization settings
PRERANK_ENABLED = bool(int(os.getenv("PRERANK_ENABLED", "1")))
TOP_K_RATIO = float(os.getenv("TOP_K_RATIO", "0.60"))
TOP_K_MIN = int(os.getenv("TOP_K_MIN", "30"))
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "0")) or None  # 0 means no cap
STRATIFY_ENABLED = bool(int(os.getenv("STRATIFY_ENABLED", "1")))
SAFETY_KEEP_ENABLED = bool(int(os.getenv("SAFETY_KEEP_ENABLED", "1")))
COMPARE_SCORING_MODES = bool(int(os.getenv("COMPARE_SCORING_MODES", "0")))

# Pre-rank scoring weights (cheap features only)
PRERANK_WEIGHTS = {
    "hook": 0.35,
    "arousal": 0.25, 
    "info_density": 0.20,
    "duration_fit": 0.15,
    "has_numbers": 0.05,
    "is_ad_penalty": -0.25
}

# Duration targets for pre-rank scoring
DURATION_TARGET_MIN = int(os.getenv("DURATION_TARGET_MIN", "8"))
DURATION_TARGET_MAX = int(os.getenv("DURATION_TARGET_MAX", "90"))