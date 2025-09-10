import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _imports():
    try:
        from services.candidate_formatter import format_candidate
    except Exception:
        format_candidate = None
    return format_candidate

format_candidate = _imports()

@pytest.mark.skipif(format_candidate is None, reason="candidate formatter not available")
def test_formatter_surfaces_enhanced_fields():
    seg_features = {
        "start": 0.0, "end": 22.0, "text": "Quick list: one, two, three.",
        "final_score": 0.57,
        "hook_score": 0.22, "arousal_score": 0.22, "payoff_score": 0.10, "info_density": 0.86,
        "loopability": 0.33, "insight_score": 0.44,
        "platform_len_match": 0.40, "platform_length_score_v2": 0.65,
        "q_list_score": 0.58, "prosody_arousal": 0.31, "emotion_score": 0.21,
        "insight_conf": 0.47,
        "scoring_version": "v4.8-unified-2025-09",
        "weights_version": "2025-09-01",
        "flags": {"USE_PL_V2": True, "USE_Q_LIST": True},
    }

    cand = format_candidate(seg_features, platform="tiktok", genre="general")

    # present and numeric
    for k in [
        "platform_length_score_v2", "q_list_score", "prosody_arousal", "insight_conf",
        "hook_score","arousal_score","payoff_score","info_density","loopability","insight_score",
        "final_score","display_score","platform_len_match"
    ]:
        assert k in cand, f"missing {k}"
        assert isinstance(cand[k], (int,float)) or k in ("flags","scoring_version","weights_version")

    # display_score derived from final_score
    assert cand["display_score"] == int(round(seg_features["final_score"] * 100))
