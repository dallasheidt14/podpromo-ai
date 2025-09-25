import math
import pytest

# Adjust paths if your modules live elsewhere
from services.util import (
    _coerce_words_list,
    calculate_finish_confidence,
)

# If you placed helpers in clip_score.py instead, import from there:
# from services.clip_score import _nearest_eos_after

# Re-declare for test if _nearest_eos_after is in util or clip_score
try:
    from services.util import _nearest_eos_after
except Exception:
    from services.clip_score import _nearest_eos_after  # pragma: no cover


def _w(t, d, text="word", prob=0.9):
    # Mirrors the shape we accept in _coerce_words_list
    return {"t": t, "d": d, "w": text, "prob": prob}


def test_coerce_words_list_accepts_list_of_dicts():
    words = [_w(1.0, 0.3, "hi"), _w(1.3, 0.2, "there")]
    out = _coerce_words_list(words)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["start"] == pytest.approx(1.0)
    assert out[0]["dur"] == pytest.approx(0.3)
    assert out[0]["end"] == pytest.approx(1.3)
    assert out[0]["text"] == "hi"
    assert out[0]["prob"] == pytest.approx(0.9)


def test_coerce_words_list_drops_bad_entries_and_non_list_inputs():
    # non-list should return []
    assert _coerce_words_list(None) == []
    assert _coerce_words_list("not-words") == []

    # mixed list (only dicts are kept)
    words = [_w(0.0, 0.5, "ok"), "bad", 123, {"start": 1.0, "end": 1.4, "text": "alt"}]
    out = _coerce_words_list(words)
    # Only two dict-like should be kept
    assert len(out) == 2
    assert out[0]["end"] == pytest.approx(0.5)  # 0.0 + 0.5
    assert out[1]["start"] == pytest.approx(1.0)
    assert out[1]["end"] == pytest.approx(1.4)


def test_nearest_eos_after_none_and_empty():
    assert _nearest_eos_after(10.0, None) is None
    assert _nearest_eos_after(10.0, []) is None


def test_nearest_eos_after_picks_min_after_end_time():
    eos = [0.5, 1.2, 3.0, 4.1, 4.15]
    # If end=3.05, the nearest after is 4.1 (delta ~1.05)
    delta = _nearest_eos_after(3.05, eos)
    assert delta == pytest.approx(4.1 - 3.05)


def test_calculate_finish_confidence_safe_defaults_with_bad_inputs():
    # No crash; returns float
    conf = calculate_finish_confidence(clip={"end": 12.0}, words="bad", eos_markers=None)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_calculate_finish_confidence_proximity_boost():
    """
    We give a clip that ends very close to an EOS; the function should add a small boost.
    The exact numbers depend on your implementation; we only assert monotonicity.
    """
    clip = {"end": 10.0}
    words = [_w(9.0, 0.5), _w(9.6, 0.3), _w(9.9, 0.2)]
    eos_close = [10.8]  # delta = 0.8s (within 1.2s window)
    eos_far = [15.0]    # delta = 5.0s (no boost)

    conf_far = calculate_finish_confidence(clip, words, eos_far)
    conf_close = calculate_finish_confidence(clip, words, eos_close)

    assert isinstance(conf_far, float) and isinstance(conf_close, float)
    assert conf_close >= conf_far  # proximity boost applied
    assert 0.0 <= conf_close <= 1.0
