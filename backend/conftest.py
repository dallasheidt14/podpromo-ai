"""
Pytest configuration and fixtures for PodPromo backend tests.
Ensures consistent environment variables across all tests.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """
    Auto-applied fixture: ensures consistent env vars across all tests.
    Mirrors .env / docker-compose settings for platform-aware scoring.
    """
    # Create temporary directories for test data
    test_upload_dir = tempfile.mkdtemp(prefix="podpromo_test_uploads_")
    test_transcript_dir = os.path.join(test_upload_dir, "transcripts")
    os.makedirs(test_transcript_dir, exist_ok=True)
    
    env_overrides = {
        # === Core Settings ===
        "ENV": "test",
        "DEBUG": "true",

        # === Platform & Picker Config ===
        "PL_V2_WEIGHT": "0.5",          # Blend factor (0 = neutral, 1 = strict)
        "PLATFORM_PROTECT": "true",     # Down-weight long clips for shorts/tiktok/reels

        # === Title Service ===
        "TITLE_ENGINE_V2": "enabled",
        "TITLE_CACHE_SIZE": "128",      # Lower for test speed

        # === Logging ===
        "LOG_LEVEL": "INFO",
        "LOG_LEVEL_TITLES": "WARNING",  # Suppress noisy skip logs
        "ONCE_PER_KEY_LOGGING": "true", # No spammy repeats

        # === File System (use tmp paths in tests) ===
        "UPLOAD_DIR": test_upload_dir,
        "TRANSCRIPT_DIR": test_transcript_dir,
        "ATOMIC_WRITE_LOCK": "true",
        
        # === ASR & Processing ===
        "FORCE_CPU_WHISPER": "1",       # Consistent CPU processing
        "CUDA_VISIBLE_DEVICES": "",     # Disable GPU for tests
        "ENABLE_TORCH_ALIGNMENT": "0",  # Disable torch alignment
    }

    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)

    yield  # tests run with these vars
    
    # Cleanup: remove temporary directories
    try:
        shutil.rmtree(test_upload_dir)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def test_upload_dir():
    """Provide the test upload directory path."""
    return os.getenv("UPLOAD_DIR", "/tmp/podpromo/uploads")


@pytest.fixture
def test_transcript_dir():
    """Provide the test transcript directory path."""
    return os.getenv("TRANSCRIPT_DIR", "/tmp/podpromo/uploads/transcripts")


@pytest.fixture
def sample_clip_data():
    """Provide sample clip data for testing."""
    return {
        "id": "clip_test_episode_123_0",
        "text": "This is a test clip about machine learning and artificial intelligence",
        "transcript": {
            "text": "This is a test clip about machine learning and artificial intelligence",
            "words": [
                {"word": "This", "start": 0.0, "end": 0.5},
                {"word": "is", "start": 0.5, "end": 0.8},
                {"word": "a", "start": 0.8, "end": 1.0},
                {"word": "test", "start": 1.0, "end": 1.3},
                {"word": "clip", "start": 1.3, "end": 1.6}
            ]
        },
        "start": 0.0,
        "end": 5.0,
        "features": {
            "is_advertisement": False,
            "language": "en"
        }
    }


@pytest.fixture
def sample_episode_data():
    """Provide sample episode data for testing."""
    return {
        "id": "test_episode_123",
        "title": "Test Episode",
        "text": "This is a test episode about technology and innovation",
        "words": [
            {"word": "This", "start": 0.0, "end": 0.5},
            {"word": "is", "start": 0.5, "end": 0.8},
            {"word": "a", "start": 0.8, "end": 1.0},
            {"word": "test", "start": 1.0, "end": 1.3},
            {"word": "episode", "start": 1.3, "end": 1.8}
        ],
        "clips": []
    }
