import pytest

# If your extend_to_natural_end lives in util.py:
from services.util import extend_to_natural_end

# If you placed tail-snap call-site logic in clip_score, you can still
# unit test the pure function that actually extends the tail.

def test_extend_to_natural_end_does_not_overextend_past_cap():
    # Clip ends at 10.0; nearest EOS is at 14.0 — with max_extend_sec=3.0,
    # should not extend all the way to 14.0.
    clip = {"start": 0.0, "end": 10.0}
    # Words around the tail—EOS would be computed elsewhere; this function
    # should be idempotent and respect max_extend_sec.
    words = [
        {"start": 9.7, "end": 9.9, "text": "end"},
        {"start": 10.1, "end": 10.3, "text": "trailing"},  # "past" current end
        {"start": 13.9, "end": 14.0, "text": "far-eosish"},
    ]

    out = extend_to_natural_end(clip.copy(), words, max_extend_sec=3.0)
    assert out["end"] <= 13.0  # 10.0 + 3.0
    assert out["end"] >= 10.0  # can only extend forward

def test_extend_to_natural_end_snaps_when_eos_is_close():
    # Clip ends at 10.0; a strong boundary ~10.8 should be reachable with max_extend_sec=3.0
    clip = {"start": 8.0, "end": 10.0}
    words = [
        {"start": 9.6, "end": 9.9, "text": "closing"},
        {"start": 10.5, "end": 10.8, "text": "final."},  # ends with period - should trigger snap
        {"start": 11.0, "end": 11.2, "text": "Next"},    # next word starts with capital
    ]

    out = extend_to_natural_end(clip.copy(), words, max_extend_sec=3.0)
    assert out["end"] >= 10.5  # snapped to a more natural end
    assert out["end"] <= 13.0
