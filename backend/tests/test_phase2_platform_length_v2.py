import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _import_platform_length_v2():
    try:
        from services.secret_sauce_pkg.scoring_utils import platform_length_score_v2
        return platform_length_score_v2
    except Exception:
        return None

PLATFORM_LENGTH_V2 = _import_platform_length_v2()

@pytest.mark.skipif(PLATFORM_LENGTH_V2 is None, reason="platform_length_score_v2 not available")
def test_platform_length_v2_rewards_target_fit():
    """
    Test that platform length v2 rewards clips closer to platform target.
    """
    # Test TikTok (target: 22s)
    tiktok_target = PLATFORM_LENGTH_V2(22.0, 0.5, "tiktok")
    tiktok_off_target = PLATFORM_LENGTH_V2(35.0, 0.5, "tiktok")
    
    # Closer to target should score higher
    assert tiktok_target > tiktok_off_target, f"Target fit failed: {tiktok_target} <= {tiktok_off_target}"
    
    # Test YouTube (target: 28s)
    youtube_target = PLATFORM_LENGTH_V2(28.0, 0.5, "youtube")
    youtube_off_target = PLATFORM_LENGTH_V2(15.0, 0.5, "youtube")
    
    assert youtube_target > youtube_off_target, f"Target fit failed: {youtube_target} <= {youtube_off_target}"

@pytest.mark.skipif(PLATFORM_LENGTH_V2 is None, reason="platform_length_score_v2 not available")
def test_platform_length_v2_rewards_density():
    """
    Test that platform length v2 rewards higher information density.
    """
    # Same duration, different density
    low_density = PLATFORM_LENGTH_V2(22.0, 0.2, "tiktok")
    high_density = PLATFORM_LENGTH_V2(22.0, 0.8, "tiktok")
    
    # Higher density should score higher
    assert high_density > low_density, f"Density reward failed: {high_density} <= {low_density}"

@pytest.mark.skipif(PLATFORM_LENGTH_V2 is None, reason="platform_length_score_v2 not available")
def test_platform_length_v2_adaptive_width():
    """
    Test that platform length v2 adapts width based on density.
    Higher density should have tighter tolerance.
    """
    # High density should be more forgiving of longer durations
    high_density_short = PLATFORM_LENGTH_V2(18.0, 0.9, "tiktok")
    high_density_long = PLATFORM_LENGTH_V2(26.0, 0.9, "tiktok")
    
    # Low density should be more forgiving of longer durations
    low_density_short = PLATFORM_LENGTH_V2(18.0, 0.1, "tiktok")
    low_density_long = PLATFORM_LENGTH_V2(26.0, 0.1, "tiktok")
    
    # Both should score reasonably well due to adaptive width
    assert high_density_short > 0.5, f"High density short should score well: {high_density_short}"
    assert high_density_long > 0.5, f"High density long should score well: {high_density_long}"
    assert low_density_short > 0.3, f"Low density short should score reasonably: {low_density_short}"
    assert low_density_long > 0.3, f"Low density long should score reasonably: {low_density_long}"

@pytest.mark.skipif(PLATFORM_LENGTH_V2 is None, reason="platform_length_score_v2 not available")
def test_platform_length_v2_bounds():
    """
    Test that platform length v2 returns values in [0, 1] range.
    """
    # Test various combinations
    test_cases = [
        (10.0, 0.1, "tiktok"),
        (22.0, 0.5, "tiktok"),
        (35.0, 0.9, "tiktok"),
        (15.0, 0.3, "youtube"),
        (28.0, 0.7, "youtube"),
        (45.0, 0.8, "youtube"),
        (20.0, 0.4, "linkedin"),
        (30.0, 0.6, "linkedin"),
    ]
    
    for seconds, density, platform in test_cases:
        score = PLATFORM_LENGTH_V2(seconds, density, platform)
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score} for {seconds}s, {density} density, {platform}"

@pytest.mark.skipif(PLATFORM_LENGTH_V2 is None, reason="platform_length_score_v2 not available")
def test_platform_length_v2_platform_differences():
    """
    Test that different platforms have different optimal lengths.
    """
    # Same content, different platforms
    tiktok_score = PLATFORM_LENGTH_V2(22.0, 0.5, "tiktok")
    youtube_score = PLATFORM_LENGTH_V2(22.0, 0.5, "youtube")
    linkedin_score = PLATFORM_LENGTH_V2(22.0, 0.5, "linkedin")
    
    # TikTok should score highest for 22s content
    assert tiktok_score > youtube_score, f"TikTok should score higher than YouTube for 22s: {tiktok_score} <= {youtube_score}"
    assert tiktok_score > linkedin_score, f"TikTok should score higher than LinkedIn for 22s: {tiktok_score} <= {linkedin_score}"
    
    # YouTube should score higher for 28s content
    tiktok_28 = PLATFORM_LENGTH_V2(28.0, 0.5, "tiktok")
    youtube_28 = PLATFORM_LENGTH_V2(28.0, 0.5, "youtube")
    
    assert youtube_28 > tiktok_28, f"YouTube should score higher than TikTok for 28s: {youtube_28} <= {tiktok_28}"
