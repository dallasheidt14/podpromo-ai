import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _import_explain():
    try:
        from services.secret_sauce_pkg import explain_segment_from_segment as explain
        return explain
    except Exception:
        return None

EXPLAIN = _import_explain()

@pytest.mark.skipif(EXPLAIN is None, reason="Explain function not available")
@pytest.mark.xfail(reason="Will pass after SYNERGY_MODE='unified' anti-bait enabled")
def test_hook_up_payoff_down_should_not_increase_final():
    base = {"start": 0.0, "end": 20.0, "text": "The #1 mistake is paying only the minimum. Here's why it hurts."}
    baity = {"start": 0.0, "end": 20.0, "text": "You won't believe this secret. Keep watching to find out."}

    audio = ""
    a = EXPLAIN(base, audio, genre="general", platform="tiktok")
    b = EXPLAIN(baity, audio, genre="general", platform="tiktok")

    # Require that the baity version does not score higher than the substantive one
    assert float(b.get("final_score", 0)) <= float(a.get("final_score", 0))
