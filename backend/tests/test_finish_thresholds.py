import pytest

from services.util import finish_threshold_for

def test_finish_threshold_for_accepts_indicators_dict():
    # These two calls used to crash when a string was passed.
    # Now we always pass a dict and the function must not error.
    indicators = {"conversational_ratio": 0.8, "interview_format": True}

    t1 = finish_threshold_for("general", indicators)
    t2 = finish_threshold_for("interview", indicators)

    assert isinstance(t1, (int, float))
    assert isinstance(t2, (int, float))
    assert 0.0 <= t1 <= 1.0
    assert 0.0 <= t2 <= 1.0

def test_finish_threshold_for_defaults_when_indicators_missing():
    t = finish_threshold_for("general", {})
    assert isinstance(t, (int, float))
    assert 0.0 <= t <= 1.0
