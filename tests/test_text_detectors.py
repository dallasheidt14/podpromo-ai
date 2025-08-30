"""
Direct tests of the text cue detectors to ensure lexicon changes don't break expectations.
"""

import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.secret_sauce import _hook_score, _question_or_list, _payoff_presence


def test_hook_detector_hits_on_contrarian():
    text = "Everyone thinks this is right, but actually it's wrong."
    assert _hook_score(text) >= 0.33


def test_emotion_detector_hits_on_high_arousal_words():
    text = "This is unbelievable, crazy, hilarious!"
    # Note: _emotion_score was removed in cleanup, emotion detection now part of _hook_score
    assert _hook_score(text) >= 0.4  # Should detect emotional words


def test_question_or_list_detector():
    q = "What if we did this differently?"
    lst = "Here are 3 rules you need to know."
    assert _question_or_list(q) >= 0.8
    assert _question_or_list(lst) >= 0.5


def test_payoff_presence_detector():
    t1 = "Here's why this works: because we reduce friction."
    t2 = "This is something."
    assert _payoff_presence(t1) == 1.0
    assert _payoff_presence(t2) in (0.0, 0.4)
