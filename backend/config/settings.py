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

# -------- Prosody Arousal v2 --------
ENABLE_PROSODY_V2 = os.getenv("ENABLE_PROSODY_V2", "1") == "1"
AROUSAL_V2_BLEND_TEXT = float(os.getenv("AROUSAL_V2_BLEND_TEXT", "0.6"))  # 0..1
AROUSAL_V2_BLEND_AUDIO = 1.0 - AROUSAL_V2_BLEND_TEXT

# -------- Trend Boost --------
ENABLE_TREND_BOOST = os.getenv("ENABLE_TREND_BOOST", "1") == "1"
TREND_BOOST_WEIGHT = float(os.getenv("TREND_BOOST_WEIGHT", "0.05"))  # 0..0.15 rec
# Compatibility shim - prefer new path, fallback to old
TRENDING_TERMS_FILE = os.getenv("TRENDING_TERMS_FILE", os.getenv("TRENDS_FILE", "backend/data/trending_terms.json"))

# Decay + blending
TREND_HALF_LIFE_DAYS = float(os.getenv("TREND_HALF_LIFE_DAYS", "7"))
TREND_GLOBAL_WEIGHT = float(os.getenv("TREND_GLOBAL_WEIGHT", "0.4"))
TREND_CATEGORY_WEIGHT = float(os.getenv("TREND_CATEGORY_WEIGHT", "0.6"))

# Categories (used by collector & provider)
TREND_CATEGORIES = os.getenv("TREND_CATEGORIES",
    "general,tech,business,crypto,sports,entertainment,health,politics").split(",")

# Optional AB test (stable hash % 100 < pct ⇒ enabled)
TREND_AB_PCT = int(os.getenv("TREND_AB_PCT", "100"))  # 0..100
TREND_CONTROL_SCALE = float(os.getenv("TREND_CONTROL_SCALE", "0.25"))  # Control bucket weight

# -------- Prerank Exploration --------
ENABLE_EXPLORATION = os.getenv("ENABLE_EXPLORATION", "1") == "1"
EXPLORATION_QUOTA = float(os.getenv("EXPLORATION_QUOTA", "0.15"))  # fraction of k
EXPLORATION_MIN = int(os.getenv("EXPLORATION_MIN", "2"))

# -------- Spectral Flux Feature (gated) --------
ENABLE_SPECTRAL_FLUX = os.getenv("ENABLE_SPECTRAL_FLUX", "0") == "1"

# Audio-side weights inside the arousal blend (used only if flux enabled+gated)
RMS_AUDIO_WEIGHT = float(os.getenv("RMS_AUDIO_WEIGHT", "0.6"))
F0_AUDIO_WEIGHT = float(os.getenv("F0_AUDIO_WEIGHT", "0.3"))
FLUX_AUDIO_WEIGHT = float(os.getenv("FLUX_AUDIO_WEIGHT", "0.1"))

# Speech gating thresholds
VOICED_FRAC_MIN = float(os.getenv("VOICED_FRAC_MIN", "0.30"))
PAUSE_FRAC_MAX = float(os.getenv("PAUSE_FRAC_MAX", "0.85"))

# Optional AB (0..1). If >0, only this fraction of ab_key traffic gets flux.
FLUX_AB_PCT = float(os.getenv("FLUX_AB_PCT", "0.0"))

# -------- Payoff V2 --------
ENABLE_PAYOFF_V2 = os.getenv("ENABLE_PAYOFF_V2", "1") == "1"
PAYOFF_TAIL_BIAS = float(os.getenv("PAYOFF_TAIL_BIAS", "0.65"))  # cutoff as fraction of text length
PAYOFF_V2_DEBUG = os.getenv("PAYOFF_V2_DEBUG", "0") == "1"

# Title generation settings
PLAT_LIMITS = {
    "shorts": 80,
    "reels": 80,
    "tiktok": 80,
    "neutral": 80,
    "default": 80,
}
TITLE_ENGINE_V2 = os.getenv("TITLE_ENGINE_V2", "true").lower() == "true"

# Title persistence settings
TITLES_INDEX_TTL_SEC = int(os.getenv("TITLES_INDEX_TTL_SEC", "60"))

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

# --- ASR settings (v2) ---
def _bool(v, d=False): 
    s = str(os.getenv(v, str(int(d)))).strip().lower()
    return s in ("1","true","yes","y","on")

def _float(v, d): 
    try: return float(os.getenv(v, str(d)))
    except: return d

def _int(v, d): 
    try: return int(os.getenv(v, str(d)))
    except: return d

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")  # legacy
ASR_MODEL = os.getenv("ASR_MODEL", "Systran/faster-whisper-large-v3")
ASR_DEVICE = os.getenv("ASR_DEVICE", "auto")  # Auto-detect with proper fallback
ASR_IMPL = os.getenv("ASR_IMPL", "faster_whisper")

_VALID_COMPUTE = {"int8","int8_float16","float16","float32"}
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")  # GPU-optimized when available
if ASR_COMPUTE_TYPE not in _VALID_COMPUTE:
    import logging
    logging.warning("ASR_COMPUTE_TYPE=%s invalid; defaulting to int8", ASR_COMPUTE_TYPE)
    ASR_COMPUTE_TYPE = "int8"

ASR_WORD_TS = _bool("ASR_WORD_TIMESTAMPS", True)
ASR_VAD = _bool("ASR_VAD_FILTER", True)
ASR_VAD_SILENCE_MS = _int("ASR_VAD_MIN_SILENCE_MS", 300)
ASR_VAD_SPEECH_PAD_MS = _int("ASR_VAD_SPEECH_PAD_MS", 200)
ASR_COND_PREV = _bool("ASR_COND_PREV", True)

ASR_BEAM_SIZE = _int("ASR_BEAM_SIZE", 1)
ASR_TEMPS = os.getenv("ASR_TEMPERATURES", "0.0,0.2")

# GPU memory management
ASR_GPU_MEMORY_FRACTION = _float("ASR_GPU_MEMORY_FRACTION", 0.85)

# Audio pre-processing
AUDIO_PREDECODE_PCM = _bool("AUDIO_PREDECODE_PCM", True)

# Torch/CUDA alignment (disabled by default on Windows)
ENABLE_TORCH_ALIGNMENT = _bool("ENABLE_TORCH_ALIGNMENT", False)

ENABLE_ASR_V2 = _bool("ENABLE_ASR_V2", True)
ENABLE_QUALITY_RETRY = _bool("ENABLE_QUALITY_RETRY", True)
ASR_LOW_QL_LOGPROB = _float("ASR_LOW_QL_LOGPROB", -1.0)
ASR_LOW_QL_COMPRESS = _float("ASR_LOW_QL_COMPRESS", 2.3)
ASR_LOW_QL_MIN_PUNCT = _float("ASR_LOW_QL_MIN_PUNCT", 0.25)
ASR_HQ_ON_RETRY_COMPUTE_TYPE = os.getenv("ASR_HQ_ON_RETRY_COMPUTE_TYPE", "int8_float16")
if ASR_HQ_ON_RETRY_COMPUTE_TYPE not in _VALID_COMPUTE:
    ASR_HQ_ON_RETRY_COMPUTE_TYPE = "int8_float16"

# UI/progress tweaks
ASR_PROGRESS_LABEL_HQ = os.getenv("ASR_PROGRESS_LABEL_HQ", "transcribing_hq")

# Feature weight adjustments for low ASR quality
ENABLE_ASR_QUALITY_WEIGHTING = _bool("ENABLE_ASR_QUALITY_WEIGHTING", True)