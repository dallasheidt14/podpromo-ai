import numpy as np
from services.audio_features import arousal_score_v2

def test_arousal_v2_bounds():
    """Test that arousal_score_v2 returns values in [0,1] range"""
    prosody = {"prosody_rms": 0.2, "prosody_flux": 0.2}
    stats = {"prosody_rms_mean": 0.1, "prosody_rms_std": 0.05, "prosody_flux_mean": 0.1, "prosody_flux_std": 0.05}
    s = arousal_score_v2(0.2, prosody, stats)
    assert 0.0 <= s <= 1.0

def test_arousal_v2_blend():
    """Test that arousal_score_v2 properly blends text and prosody"""
    prosody = {"prosody_rms": 0.5, "prosody_flux": 0.5}
    stats = {"prosody_rms_mean": 0.3, "prosody_rms_std": 0.1, "prosody_flux_mean": 0.3, "prosody_flux_std": 0.1}
    
    # Test with different text arousal values
    s1 = arousal_score_v2(0.0, prosody, stats)  # pure prosody
    s2 = arousal_score_v2(1.0, prosody, stats)  # pure text
    s3 = arousal_score_v2(0.5, prosody, stats)  # blended
    
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0
    assert 0.0 <= s3 <= 1.0
    assert s1 < s2  # text arousal should boost the score
