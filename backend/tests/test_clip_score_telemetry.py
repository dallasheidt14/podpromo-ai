import pytest

# Import the helpers where you added them (clip_score.py as suggested)
from services.clip_score import _telemetry_density

def _word(start, dur):
    return {"t": start, "d": dur}

def test_telemetry_density_handles_none_eos_and_empty_words():
    wc, ec, dens = _telemetry_density([], None)
    assert wc == 0
    assert ec == 0
    assert dens == 0.0

def test_telemetry_density_basic_counts_and_density():
    words = [_word(0.0, 0.5), _word(0.6, 0.4), _word(1.2, 0.3)]
    eos = [0.49, 1.0, 1.6, 2.0]
    wc, ec, dens = _telemetry_density(words, eos)
    assert wc == 3
    assert ec == 4
    # density = eos_count / word_count = 4 / 3
    assert dens == pytest.approx(4/3)
