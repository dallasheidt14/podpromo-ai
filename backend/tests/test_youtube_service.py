"""Tests for YouTube service functionality."""
import pytest
from unittest.mock import Mock, patch
from services.youtube_service import _validate_url, probe, map_error
from yt_dlp.utils import DownloadError, ExtractorError

class DummyYDL:
    def __init__(self, opts): 
        self.opts = opts
    def __enter__(self): 
        return self
    def __exit__(self, *a): 
        pass
    def extract_info(self, url, download=False):
        return {
            "id": "abc123",
            "title": "Test Video",
            "duration": 120,
            "is_live": False,
            "webpage_url": url
        }
    def download(self, urls): 
        return 0

def test_validate_url():
    """Test URL validation."""
    assert _validate_url("https://www.youtube.com/watch?v=abc123") == True
    assert _validate_url("https://youtu.be/abc123") == True
    assert _validate_url("https://vimeo.com/123") == False
    assert _validate_url("not-a-url") == False
    assert _validate_url("") == False

def test_probe_duration(monkeypatch):
    """Test video metadata extraction."""
    monkeypatch.setattr("services.youtube_service.YoutubeDL", DummyYDL)
    meta = probe("https://youtu.be/abc123")
    assert meta.duration == 120
    assert meta.title == "Test Video"
    assert meta.id == "abc123"

def test_probe_too_short(monkeypatch):
    """Test rejection of videos that are too short."""
    class ShortVideoYDL(DummyYDL):
        def extract_info(self, url, download=False):
            return {
                "id": "short",
                "title": "Short Video",
                "duration": 30,  # Too short
                "is_live": False,
                "webpage_url": url
            }
    
    monkeypatch.setattr("services.youtube_service.YoutubeDL", ShortVideoYDL)
    with pytest.raises(ValueError) as e:
        probe("https://youtu.be/short")
    assert str(e.value) == "too_short"

def test_probe_live_stream(monkeypatch):
    """Test rejection of live streams."""
    class LiveYDL(DummyYDL):
        def extract_info(self, url, download=False):
            return {
                "id": "live",
                "title": "Live Stream",
                "duration": 3600,
                "is_live": True,
                "webpage_url": url
            }
    
    monkeypatch.setattr("services.youtube_service.YoutubeDL", LiveYDL)
    with pytest.raises(ValueError) as e:
        probe("https://youtu.be/live")
    assert str(e.value) == "live_stream_not_supported"

def test_invalid_url():
    """Test invalid URL handling."""
    with pytest.raises(ValueError) as e:
        probe("ftp://bad")
    assert str(e.value) == "invalid_url"

def test_map_error():
    """Test error mapping to HTTP status codes."""
    # ValueError cases
    assert map_error(ValueError("invalid_url")) == (400, "invalid_url")
    assert map_error(ValueError("too_short")) == (400, "too_short")
    assert map_error(ValueError("too_long")) == (400, "too_long")
    assert map_error(ValueError("live_stream_not_supported")) == (400, "live_stream_not_supported")
    
    # DownloadError cases
    assert map_error(DownloadError("Download failed")) == (502, "download_failed")
    assert map_error(ExtractorError("Extraction failed")) == (502, "download_failed")
    
    # RuntimeError cases
    assert map_error(RuntimeError("ffmpeg_failed: some error")) == (500, "audio_conversion_failed")
    
    # Generic error
    assert map_error(Exception("Unknown error")) == (500, "internal_error")
