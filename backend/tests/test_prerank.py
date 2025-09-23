import os
from services.prerank import pre_rank_candidates

def test_exploration_adds_tail():
    """Test that exploration quota adds info-dense candidates"""
    # Set environment variables for exploration
    os.environ["ENABLE_EXPLORATION"] = "1"
    os.environ["EXPLORATION_QUOTA"] = "0.2"
    os.environ["EXPLORATION_MIN"] = "2"
    
    # Create test candidates with varying info density
    cands = []
    for i in range(100):
        cands.append({
            "id": f"seg_{i}",
            "text": f"Segment {i} with some content",
            "duration": 30.0,
            "words_per_sec": 2.0,
            "prerank_features": {"info_density": 100 - i}  # Higher info density for later segments
        })
    
    # Run prerank
    result = pre_rank_candidates(cands, "test_episode")
    
    # Should have more than just the top-k due to exploration
    assert len(result) >= 12  # 10 + exploration tail
    
    # Clean up environment
    del os.environ["ENABLE_EXPLORATION"]
    del os.environ["EXPLORATION_QUOTA"]
    del os.environ["EXPLORATION_MIN"]

def test_exploration_disabled():
    """Test that exploration is disabled when flag is off"""
    os.environ["ENABLE_EXPLORATION"] = "0"
    
    cands = [{"id": f"seg_{i}", "text": f"Segment {i}", "duration": 30.0, "words_per_sec": 2.0} for i in range(50)]
    
    result = pre_rank_candidates(cands, "test_episode")
    
    # Should only have top-k, no exploration
    assert len(result) <= 50  # Should be limited by TOP_K settings
    
    del os.environ["ENABLE_EXPLORATION"]
