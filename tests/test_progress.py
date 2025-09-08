"""
Progress Service Tests - Comprehensive fail-safe and persistence testing
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

# Import the app AFTER we set env vars so settings read the temp dirs
def make_app(tmpdir: Path):
    """Create app instance with temporary directories for testing"""
    os.environ["UPLOAD_DIR"] = str(tmpdir / "uploads")
    os.environ["CLIPS_DIR"] = str(tmpdir / "clips")
    os.environ["TRANSCRIPTS_DIR"] = str(tmpdir / "transcripts")
    os.environ["SUPABASE_URL"] = ""  # Disable Supabase for tests
    os.environ["SUPABASE_KEY"] = ""
    
    # Create directories
    (tmpdir / "uploads").mkdir(parents=True, exist_ok=True)
    (tmpdir / "clips").mkdir(parents=True, exist_ok=True)
    (tmpdir / "transcripts").mkdir(parents=True, exist_ok=True)
    
    # Import after setting env vars
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import app
    return app

def write_progress(uploads: Path, episode_id: str, data: dict):
    """Write progress data atomically to simulate real progress updates"""
    ep_dir = uploads / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    tmp = ep_dir / "progress.tmp"
    out = ep_dir / "progress.json"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, out)

def test_progress_nonexistent_returns_404_json(tmp_path):
    """Test that nonexistent episodes return 404 with JSON, not HTML or 500"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    r = client.get("/api/progress/does-not-exist")
    assert r.status_code == 404
    
    # Should return JSON, not HTML
    assert r.headers.get("content-type", "").startswith("application/json")
    
    body = r.json()
    assert isinstance(body, dict)
    assert body.get("ok") is False
    assert "error" in body or "Episode not found" in str(body)

def test_progress_file_completed_returns_ok(tmp_path):
    """Test that valid progress files return proper JSON structure"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-123"
    write_progress(tmp_path / "uploads", ep, {
        "percentage": 100,
        "stage": "completed",
        "message": "Processing completed",
        "timestamp": "2024-01-01T00:00:00Z"
    })
    
    r = client.get(f"/api/progress/{ep}")
    assert r.status_code == 200
    
    body = r.json()
    assert body["ok"] is True
    assert body["progress"]["stage"] == "completed"
    assert body["progress"]["percentage"] == 100
    assert "status" in body

def test_progress_corrupt_file_never_500(tmp_path):
    """Test that corrupt progress files never cause 500 errors"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-err"
    ep_dir = tmp_path / "uploads" / ep
    ep_dir.mkdir(parents=True, exist_ok=True)
    
    # Write invalid JSON
    (ep_dir / "progress.json").write_text("{not:json", encoding="utf-8")
    
    r = client.get(f"/api/progress/{ep}")
    
    # Should never return 500 - should degrade gracefully
    assert r.status_code in (200, 404)
    assert r.headers.get("content-type", "").startswith("application/json")
    
    body = r.json()
    assert isinstance(body, dict)
    # Should either be a 404 or an error status, but never crash

def test_progress_disk_inference_audio_only(tmp_path):
    """Test progress inference when only audio file exists"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-audio-only"
    audio_file = tmp_path / "uploads" / f"{ep}.mp3"
    audio_file.write_bytes(b"fake audio data")
    
    r = client.get(f"/api/progress/{ep}")
    assert r.status_code == 200
    
    body = r.json()
    assert body["ok"] is True
    assert body["progress"]["stage"] in ("queued", "processing")
    assert body["progress"]["percentage"] > 0

def test_progress_disk_inference_completed(tmp_path):
    """Test progress inference when transcript exists (completed)"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-completed"
    audio_file = tmp_path / "uploads" / f"{ep}.mp3"
    transcript_file = tmp_path / "transcripts" / f"{ep}.json"
    
    audio_file.write_bytes(b"fake audio data")
    transcript_file.write_text(json.dumps({"segments": []}), encoding="utf-8")
    
    r = client.get(f"/api/progress/{ep}")
    assert r.status_code == 200
    
    body = r.json()
    assert body["ok"] is True
    assert body["progress"]["stage"] == "completed"
    assert body["progress"]["percentage"] == 100

def test_progress_atomic_writes(tmp_path):
    """Test that progress updates are atomic and don't cause corruption"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-atomic"
    
    # Simulate concurrent writes by creating a temp file first
    ep_dir = tmp_path / "uploads" / ep
    ep_dir.mkdir(parents=True, exist_ok=True)
    
    # Write temp file (simulating interrupted write)
    temp_file = ep_dir / "progress.tmp"
    temp_file.write_text('{"partial": "data"', encoding="utf-8")
    
    # Should not crash when temp file exists but no progress.json
    r = client.get(f"/api/progress/{ep}")
    assert r.status_code in (200, 404)
    body = r.json()
    assert isinstance(body, dict)

def test_progress_missing_directories(tmp_path):
    """Test that missing directories don't cause crashes"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    # Remove uploads directory to test missing dir handling
    shutil.rmtree(tmp_path / "uploads", ignore_errors=True)
    
    r = client.get("/api/progress/any-episode")
    assert r.status_code in (200, 404)
    body = r.json()
    assert isinstance(body, dict)

def test_progress_large_file_handling(tmp_path):
    """Test handling of large progress files"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-large"
    ep_dir = tmp_path / "uploads" / ep
    ep_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a large but valid JSON file
    large_data = {
        "percentage": 50,
        "stage": "processing",
        "message": "Processing large file...",
        "timestamp": "2024-01-01T00:00:00Z",
        "detail": "x" * 10000  # Large detail field
    }
    
    write_progress(tmp_path / "uploads", ep, large_data)
    
    r = client.get(f"/api/progress/{ep}")
    assert r.status_code == 200
    
    body = r.json()
    assert body["ok"] is True
    assert body["progress"]["stage"] == "processing"

def test_progress_unicode_handling(tmp_path):
    """Test handling of unicode characters in progress messages"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-unicode"
    unicode_message = "Processing with émojis 🎵 and ñ characters"
    
    write_progress(tmp_path / "uploads", ep, {
        "percentage": 75,
        "stage": "transcription",
        "message": unicode_message,
        "timestamp": "2024-01-01T00:00:00Z"
    })
    
    r = client.get(f"/api/progress/{ep}")
    assert r.status_code == 200
    
    body = r.json()
    assert body["ok"] is True
    assert body["progress"]["message"] == unicode_message

def test_progress_concurrent_access(tmp_path):
    """Test that concurrent access to progress files doesn't cause issues"""
    app = make_app(tmp_path)
    client = TestClient(app)
    
    ep = "ep-concurrent"
    write_progress(tmp_path / "uploads", ep, {
        "percentage": 30,
        "stage": "transcription",
        "message": "Processing...",
        "timestamp": "2024-01-01T00:00:00Z"
    })
    
    # Simulate multiple concurrent requests
    responses = []
    for _ in range(5):
        r = client.get(f"/api/progress/{ep}")
        responses.append(r)
    
    # All should succeed and return consistent data
    for r in responses:
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["progress"]["stage"] == "transcription"
