import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _import_synergy_bonus():
    try:
        from services.secret_sauce_pkg.scoring_utils import synergy_bonus
        return synergy_bonus
    except Exception:
        return None

SYNERGY_BONUS = _import_synergy_bonus()

@pytest.mark.skipif(SYNERGY_BONUS is None, reason="synergy_bonus not available")
def test_unified_synergy_anti_bait_logic():
    """
    Test that unified synergy prevents hook-only bait from scoring high.
    Increasing hook while lowering payoff should not increase final score.
    """
    # Base case: balanced content
    balanced = {
        "hook": 0.7,
        "arousal": 0.6,
        "emotion": 0.5,
        "payoff": 0.7,
        "info_density": 0.6,
        "q_list": 0.4,
        "loopability": 0.5,
        "platform_length": 0.6
    }
    
    # Bait case: high hook, low payoff
    baity = {
        "hook": 0.9,  # Higher hook
        "arousal": 0.6,
        "emotion": 0.5,
        "payoff": 0.2,  # Lower payoff
        "info_density": 0.2,  # Lower info density
        "q_list": 0.4,
        "loopability": 0.5,
        "platform_length": 0.6
    }
    
    balanced_synergy = SYNERGY_BONUS(balanced)
    baity_synergy = SYNERGY_BONUS(baity)
    
    # The baity version should have lower synergy due to anti-bait penalties
    assert baity_synergy < balanced_synergy, f"Anti-bait failed: baity_synergy={baity_synergy} >= balanced_synergy={balanced_synergy}"
    
    # Both should be within bounds
    assert -0.10 <= balanced_synergy <= 0.15, f"Balanced synergy out of bounds: {balanced_synergy}"
    assert -0.10 <= baity_synergy <= 0.15, f"Baity synergy out of bounds: {baity_synergy}"

@pytest.mark.skipif(SYNERGY_BONUS is None, reason="synergy_bonus not available")
def test_unified_synergy_rewards_balance():
    """
    Test that unified synergy rewards well-balanced content.
    """
    # Well-balanced content should get positive synergy
    balanced = {
        "hook": 0.8,
        "arousal": 0.7,
        "emotion": 0.6,
        "payoff": 0.8,
        "info_density": 0.7,
        "q_list": 0.6,
        "loopability": 0.7,
        "platform_length": 0.6
    }
    
    synergy = SYNERGY_BONUS(balanced)
    
    # Well-balanced content should get positive synergy
    assert synergy > 0, f"Well-balanced content should get positive synergy: {synergy}"
    assert synergy <= 0.15, f"Synergy should be bounded: {synergy}"

@pytest.mark.skipif(SYNERGY_BONUS is None, reason="synergy_bonus not available")
def test_unified_synergy_penalizes_imbalance():
    """
    Test that unified synergy penalizes imbalanced content.
    """
    # Imbalanced content: high arousal, low payoff (triggers anti-bait penalty)
    imbalanced = {
        "hook": 0.5,
        "arousal": 0.9,  # High arousal (>0.8 triggers penalty)
        "emotion": 0.6,
        "payoff": 0.2,  # Low payoff (<0.3 triggers penalty)
        "info_density": 0.3,
        "q_list": 0.4,
        "loopability": 0.5,
        "platform_length": 0.6
    }
    
    synergy = SYNERGY_BONUS(imbalanced)
    
    # The penalty should make this lower than a balanced case
    balanced = {
        "hook": 0.5,
        "arousal": 0.6,  # Moderate arousal
        "emotion": 0.6,
        "payoff": 0.6,  # Moderate payoff
        "info_density": 0.6,
        "q_list": 0.4,
        "loopability": 0.5,
        "platform_length": 0.6
    }
    
    balanced_synergy = SYNERGY_BONUS(balanced)
    
    # Imbalanced should score lower than balanced
    assert synergy < balanced_synergy, f"Imbalanced should score lower than balanced: {synergy} >= {balanced_synergy}"
    assert synergy >= -0.10, f"Synergy should be bounded: {synergy}"
