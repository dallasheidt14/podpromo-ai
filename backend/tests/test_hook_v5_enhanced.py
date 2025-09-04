import pytest
from scoring.hook_v5_enhanced import score_hook_v5_enhanced

def test_evidence_cap_releases_with_numbers():
    s = "Here's why 20% of teams fail in week 1."
    score, _, dbg = score_hook_v5_enhanced(s, arousal=0.7, q_or_list=0.7)
    assert dbg["evidence_ok"] is True
    assert score >= 0.25

def test_anti_intro_not_triggered_on_real_hook():
    s = "Hey—here's why this 30% refinance myth costs you $2,000."
    score, reasons, dbg = score_hook_v5_enhanced(s)
    assert "anti_intro" not in reasons or score > 0.15

def test_synergy_matters_but_is_bounded():
    s = "Here's the one thing nobody tells you: do this, not that."
    s1, _, _ = score_hook_v5_enhanced(s, arousal=0.59, q_or_list=0.59)
    s2, _, dbg = score_hook_v5_enhanced(s, arousal=0.85, q_or_list=0.85)
    assert s2 > s1
    assert dbg["synergy"]["bonus"] <= 0.08
