"""
Tests for the PodPromo 'secret sauce' scoring.
These tests verify that:
- Hook presence materially boosts score
- Prosody (energy) materially boosts score
- Removing the hook or flattening prosody reduces score
- Question/List + Payoff influence the score in the right direction
"""

import math
import importlib
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.secret_sauce import compute_features, score_segment, CLIP_WEIGHTS


def _score_from_features(**overrides):
    """Helper: build a features dict with defaults, then score."""
    base = {
        "hook_score": 0.0,
        "prosody_score": 0.0,
        "emotion_score": 0.0,
        "question_score": 0.0,
        "payoff_score": 0.0,
        "info_density": 0.8,   # default: reasonably snappy
        "loopability": 0.6,
    }
    base.update(overrides)
    return score_segment(base, CLIP_WEIGHTS)


def test_hook_materially_increases_score():
    low = _score_from_features(hook_score=0.05)
    high = _score_from_features(hook_score=0.90)
    assert high > low
    assert (high - low) >= 0.2, "Hook should shift ClipScore by a meaningful margin"


def test_prosody_materially_increases_score():
    low = _score_from_features(prosody_score=0.05)
    high = _score_from_features(prosody_score=0.90)
    assert high > low
    assert (high - low) >= 0.12, "Prosody weight should be impactful"


def test_question_list_and_payoff_help():
    base = _score_from_features()
    q_only = _score_from_features(question_score=0.8)
    payoff_only = _score_from_features(payoff_score=0.8)
    q_plus_payoff = _score_from_features(question_score=0.8, payoff_score=0.8)

    assert q_only > base
    assert payoff_only > base
    assert q_plus_payoff > max(q_only, payoff_only), "Q+Payoff should compound"


def test_removing_hook_drops_score():
    with_hook = _score_from_features(hook_score=0.85, prosody_score=0.4, payoff_score=0.6)
    no_hook = _score_from_features(hook_score=0.05, prosody_score=0.4, payoff_score=0.6)
    assert with_hook > no_hook
    assert (with_hook - no_hook) >= 0.2


def test_flat_prosody_drops_score():
    energetic = _score_from_features(hook_score=0.7, prosody_score=0.8)
    flat = _score_from_features(hook_score=0.7, prosody_score=0.1)
    assert energetic > flat
    assert (energetic - flat) >= 0.12
