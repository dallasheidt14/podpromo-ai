"""
Application settings and configuration
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Upload and output directories
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Audio settings
SAMPLE_RATE = 16000  # Hz
WHISPER_LANGUAGE = "en"  # Language for Whisper transcription

# File settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac"}

# Progress tracking settings
PROGRESS_TRACKER_TTL = 300  # 5 minutes in seconds

# Whisper settings
WHISPER_MODEL = "base"  # Model size: tiny, base, small, medium, large

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "supabase")

# API settings
API_PREFIX = "/api"
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# Monitoring settings
METRICS_INTERVAL = 60  # seconds
CLEANUP_INTERVAL = 300  # seconds
