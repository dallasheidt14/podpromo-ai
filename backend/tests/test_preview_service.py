"""Tests for preview service functionality."""
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from services.preview_service import ensure_preview, _hash_name, get_episode_media_path

def test_hash_name():
    """Test hash name generation."""
    hash1 = _hash_name("ep1", "clip1", 10.5, 20.5)
    hash2 = _hash_name("ep1", "clip1", 10.5, 20.5)
    hash3 = _hash_name("ep1", "clip1", 10.6, 20.5)
    
    # Same inputs should produce same hash
    assert hash1 == hash2
    # Different inputs should produce different hash
    assert hash1 != hash3
    # Hash should be reasonable length
    assert len(hash1) == 16

def test_ensure_preview_idempotent(tmp_path, monkeypatch):
    """Test that ensure_preview is idempotent."""
    # Mock ffmpeg to avoid actual execution
    def mock_ffmpeg(cmd):
        # Create a dummy file to simulate successful ffmpeg execution
        output_path = Path(cmd.split()[-1].strip("'"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy mp4 content")
    
    monkeypatch.setattr("services.preview_service._ffmpeg", mock_ffmpeg)
    
    # Create a dummy source file
    src = tmp_path / "test.mp4"
    src.write_bytes(b"dummy video content")
    
    # First call should create the preview
    url1 = ensure_preview(
        source_media=src,
        episode_id="ep1",
        clip_id="clip1",
        start_sec=0,
        end_sec=10
    )
    
    # Second call should return the same URL (idempotent)
    url2 = ensure_preview(
        source_media=src,
        episode_id="ep1",
        clip_id="clip1",
        start_sec=0,
        end_sec=10
    )
    
    assert url1 == url2
    assert url1 is not None
    assert url1.startswith("/clips/previews/")

def test_ensure_preview_audio_fallback(tmp_path, monkeypatch):
    """Test audiogram fallback for audio-only sources."""
    def mock_ffmpeg(cmd):
        # Check that audiogram command is used for audio files
        if "color=c=black" in cmd and "showspectrum" in cmd:
            output_path = Path(cmd.split()[-1].strip("'"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"dummy audiogram content")
        else:
            raise RuntimeError("Expected audiogram command")
    
    monkeypatch.setattr("services.preview_service._ffmpeg", mock_ffmpeg)
    
    # Create a dummy audio file
    src = tmp_path / "test.wav"
    src.write_bytes(b"dummy audio content")
    
    url = ensure_preview(
        source_media=src,
        episode_id="ep1",
        clip_id="clip1",
        start_sec=0,
        end_sec=10
    )
    
    assert url is not None
    assert url.startswith("/clips/previews/")

def test_get_episode_media_path_flat_structure(tmp_path, monkeypatch):
    """Test path resolution for new flat structure."""
    # Mock the uploads directory
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    
    # Create test files
    (uploads_dir / "episode123.mp4").write_bytes(b"video content")
    (uploads_dir / "episode456.wav").write_bytes(b"audio content")
    
    monkeypatch.setattr("services.preview_service.UPLOADS_DIR", uploads_dir)
    
    # Should find MP4 file
    path1 = get_episode_media_path("episode123")
    assert path1 is not None
    assert path1.name == "episode123.mp4"
    
    # Should find WAV file
    path2 = get_episode_media_path("episode456")
    assert path2 is not None
    assert path2.name == "episode456.wav"
    
    # Should return None for non-existent episode
    path3 = get_episode_media_path("nonexistent")
    assert path3 is None

def test_get_episode_media_path_legacy_structure(tmp_path, monkeypatch):
    """Test path resolution for legacy directory structure."""
    # Mock the uploads directory
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    
    # Create legacy directory structure
    episode_dir = uploads_dir / "episode123"
    episode_dir.mkdir()
    (episode_dir / "audio.mp3").write_bytes(b"audio content")
    
    monkeypatch.setattr("services.preview_service.UPLOADS_DIR", uploads_dir)
    
    # Should find audio file in legacy structure
    path = get_episode_media_path("episode123")
    assert path is not None
    assert path.name == "audio.mp3"
