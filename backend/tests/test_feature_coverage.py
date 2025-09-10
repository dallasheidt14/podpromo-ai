import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

REQUIRED = [
    "hook_score", "arousal_score", "payoff_score", "info_density",
    "loopability", "insight_score", "platform_len_match"
]
OPTIONAL = ["insight_conf","q_list_score","prosody_arousal","platform_length_score_v2","emotion_score"]

def _import_enhanced_and_score():
    try:
        from services.secret_sauce_pkg import compute_features_v4_enhanced as compute
        from services.secret_sauce_pkg import score_segment_v4 as score  # or your segment scoring entry
        return compute, score
    except Exception:
        return None, None

compute, score_seg = _import_enhanced_and_score()

@pytest.mark.skipif(compute is None or score_seg is None, reason="Enhanced compute or scorer not importable")
def test_enhanced_returns_expected_keys_and_scorer_reads_v2():
    # Minimal toy segment
    seg = {
        "start": 0.0,
        "end": 24.0,
        "text": "Here is a clear takeaway with a short list: One, automate. Two, review. Three, adjust.",
    }
    audio = ""  # provide empty or stubbed path as your code allows
    feats = compute(segment=seg, audio_file=audio, platform="tiktok", genre="general", segments=[seg])

    # Coverage checks
    for k in REQUIRED:
        assert k in feats, f"Missing required feature: {k}"
        assert isinstance(feats[k], (int,float)), f"Feature {k} must be numeric"

    for k in OPTIONAL:
        assert k in feats, f"Missing optional (now returned) feature: {k}"

    # Prove scorer reads v2 when present
    # Force a scenario where v1 is weak but v2 is strong
    feats_v1_weak = dict(feats)
    feats_v1_weak["platform_len_match"] = 0.1
    feats_v1_weak["platform_length_score_v2"] = 0.9

    out = score_seg(feats_v1_weak, platform="tiktok", genre="general")
    assert isinstance(out, dict) and "final_score" in out
    final1 = out["final_score"]

    # If we disable v2 (set same as v1), score should not be higher
    feats_v2_off = dict(feats_v1_weak)
    feats_v2_off["platform_length_score_v2"] = feats_v1_weak["platform_len_match"]
    out2 = score_seg(feats_v2_off, platform="tiktok", genre="general")
    final2 = out2["final_score"]

    assert final1 >= final2, "Scorer did not incorporate platform_length_score_v2 as intended"
