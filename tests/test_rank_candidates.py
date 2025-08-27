"""
Ranking behavior tests (light integration).
We monkeypatch clip_engine.compute_features so we don't need real audio.
"""

import types
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.clip_score import ClipScoreService


def test_rank_prefers_hooky_segment(monkeypatch):
    # Two fake segments: one 'hooky', one bland
    segments = [
        {"start": 10.0, "end": 28.0, "text": "Most people get this wrong. Here's why..."},
        {"start": 40.0, "end": 58.0, "text": "Today we talk about several things."},
    ]

    def fake_compute_features(segment, audio_file):
        t = segment["text"].lower()
        return {
            "hook_score": 0.9 if "wrong" in t or "here's why" in t else 0.1,
            "prosody_score": 0.5,
            "emotion_score": 0.3,
            "question_score": 0.3 if "?" in t else 0.0,
            "payoff_score": 0.7 if "why" in t else 0.2,
            "info_density": 0.8,
            "loopability": 0.6,
        }

    # Create service instance and patch the compute_features function
    service = ClipScoreService()
    monkeypatch.setattr(service, "_fake_compute_features", fake_compute_features)
    
    # Monkeypatch the secret_sauce import in the service
    monkeypatch.setattr(service, "compute_features", fake_compute_features)

    ranked = service.rank_candidates(segments, audio_file="dummy", top_k=2)
    assert ranked[0]["text"].startswith("Most people get this wrong"), "Hooky segment should rank first"
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_rank_considers_payoff(monkeypatch):
    segments = [
        {"start": 0.0, "end": 18.0, "text": "What if you could 3x your results? The key is focus."},
        {"start": 20.0, "end": 38.0, "text": "What if you could 3x your results? Anyway, let's move on."},
    ]

    def fake_compute_features(segment, audio_file):
        t = segment["text"].lower()
        has_payoff = "the key is" in t or "here's why" in t or "because" in t
        return {
            "hook_score": 0.7,  # both have a hook
            "prosody_score": 0.4,
            "emotion_score": 0.2,
            "question_score": 0.7,  # both ask a question
            "payoff_score": 0.9 if has_payoff else 0.1,
            "info_density": 0.8,
            "loopability": 0.6,
        }

    service = ClipScoreService()
    monkeypatch.setattr(service, "compute_features", fake_compute_features)

    ranked = service.rank_candidates(segments, audio_file="dummy", top_k=2)
    assert "the key is" in ranked[0]["text"].lower(), "Segment with payoff should rank higher"
