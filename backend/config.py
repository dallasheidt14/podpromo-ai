import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "PodPromo AI"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # File Upload Configuration
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: list = [".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi"]
    
    # Audio Processing Configuration
    SAMPLE_RATE: int = 44100
    CHUNK_SIZE: int = 1024
    MIN_CLIP_DURATION: int = 12  # seconds
    MAX_CLIP_DURATION: int = 30  # seconds
    DEFAULT_CLIP_COUNT: int = 3
    
    # Video Generation Configuration
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920
    FPS: int = 30
    VIDEO_BITRATE: str = "2M"
    AUDIO_BITRATE: str = "128k"
    
    # ClipScore Algorithm Configuration
    HOOK_WEIGHT: float = 0.35
    EMOTION_WEIGHT: float = 0.15
    PROSODY_WEIGHT: float = 0.20
    PAYOFF_WEIGHT: float = 0.10
    LOOPABILITY_WEIGHT: float = 0.05
    QUESTION_OR_LIST_WEIGHT: float = 0.10
    INFO_DENSITY_WEIGHT: float = 0.05
    
    # Whisper Configuration
    WHISPER_MODEL: str = "base"  # base, small, medium, large
    WHISPER_LANGUAGE: Optional[str] = None  # auto-detect if None
    
    # Storage Configuration
    ENABLE_PERSISTENCE: bool = True
    DATABASE_URL: Optional[str] = None  # SQLite by default
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Configuration
    MAX_WORKERS: int = 4
    TASK_TIMEOUT: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
