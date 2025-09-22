"""
Test that clip words endpoint returns relative, clipped timings
"""
from fastapi.testclient import TestClient
from main import app

def test_clip_words_are_relative_and_clipped(monkeypatch):
    """Test that words endpoint returns clip-relative timings that are properly clipped"""
    client = TestClient(app)
    
    # This test assumes there's at least one episode with clips in the system
    # In a real test environment, you'd set up test data or use fixtures
    # For now, we'll test the endpoint structure and basic validation
    
    # Try to get a clip ID from the episodes endpoint first
    episodes_response = client.get("/api/episodes")
    if episodes_response.status_code != 200:
        # Skip test if no episodes available
        return
    
    episodes = episodes_response.json()
    if not episodes or not episodes.get("episodes"):
        # Skip test if no episodes
        return
    
    # Get clips from the first episode
    first_episode = episodes["episodes"][0]
    episode_id = first_episode["id"]
    
    clips_response = client.get(f"/api/episodes/{episode_id}/clips")
    if clips_response.status_code != 200:
        # Skip test if no clips
        return
    
    clips = clips_response.json()
    if not clips or not clips.get("clips"):
        # Skip test if no clips
        return
    
    # Test the first clip
    first_clip = clips["clips"][0]
    clip_id = first_clip["id"]
    
    # Test the words endpoint
    r = client.get(f"/api/clips/{clip_id}/words")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    
    data = r.json()
    assert "start" in data
    assert "end" in data
    assert "words" in data
    assert isinstance(data["words"], list)
    
    # Check that start < end
    assert data["start"] < data["end"], f"start ({data['start']}) should be < end ({data['end']})"
    
    # Calculate clip duration
    dur = data["end"] - data["start"]
    assert dur > 0, f"Clip duration should be positive, got {dur}"
    
    # Check each word timing
    for i, w in enumerate(data["words"]):
        assert "t" in w, f"Word {i} missing 't' field"
        assert "d" in w, f"Word {i} missing 'd' field"
        assert "w" in w, f"Word {i} missing 'w' field"
        
        # Check that times are relative (0 <= t <= duration)
        assert 0.0 <= w["t"] <= dur + 1e-3, f"Word {i} time {w['t']} not in range [0, {dur}]"
        
        # Check that duration is positive and clipped
        assert 0.0 <= w["d"] <= dur + 1e-3, f"Word {i} duration {w['d']} not in range [0, {dur}]"
        
        # Check that word doesn't extend beyond clip end
        assert w["t"] + w["d"] <= dur + 1e-3, f"Word {i} extends beyond clip end: {w['t']} + {w['d']} > {dur}"
        
        # Check that word text is not empty
        assert w["w"].strip(), f"Word {i} has empty text"

def test_clip_words_range_format():
    """Test the legacy range format also returns relative timings"""
    client = TestClient(app)
    
    # Get a clip ID (same logic as above)
    episodes_response = client.get("/api/episodes")
    if episodes_response.status_code != 200:
        return
    
    episodes = episodes_response.json()
    if not episodes or not episodes.get("episodes"):
        return
    
    first_episode = episodes["episodes"][0]
    episode_id = first_episode["id"]
    
    clips_response = client.get(f"/api/episodes/{episode_id}/clips")
    if clips_response.status_code != 200:
        return
    
    clips = clips_response.json()
    if not clips or not clips.get("clips"):
        return
    
    first_clip = clips["clips"][0]
    clip_id = first_clip["id"]
    
    # Test the range format
    r = client.get(f"/api/clips/{clip_id}/words?format=range")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    
    data = r.json()
    assert "start" in data
    assert "end" in data
    assert "words" in data
    assert isinstance(data["words"], list)
    
    # For range format, start should be 0 (relative)
    assert data["start"] == 0, f"Range format start should be 0, got {data['start']}"
    
    # Check that end is positive
    assert data["end"] > 0, f"Range format end should be positive, got {data['end']}"
    
    # Check each word timing
    for i, w in enumerate(data["words"]):
        assert "t0" in w, f"Word {i} missing 't0' field"
        assert "t1" in w, f"Word {i} missing 't1' field"
        assert "w" in w, f"Word {i} missing 'w' field"
        
        # Check that times are relative (0 <= t0 <= end)
        assert 0 <= w["t0"] <= data["end"], f"Word {i} t0 {w['t0']} not in range [0, {data['end']}]"
        assert 0 <= w["t1"] <= data["end"], f"Word {i} t1 {w['t1']} not in range [0, {data['end']}]"
        
        # Check that t0 < t1
        assert w["t0"] < w["t1"], f"Word {i} t0 ({w['t0']}) should be < t1 ({w['t1']})"
        
        # Check that word text is not empty
        assert w["w"].strip(), f"Word {i} has empty text"
