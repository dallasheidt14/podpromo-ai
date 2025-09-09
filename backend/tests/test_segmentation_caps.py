import os, sys, math, pytest

# Ensure backend imports work (mirrors your existing tests)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Try both modular and monolithic surfaces
def _import_create_dynamic_segments():
    try:
        from services.secret_sauce_pkg import create_dynamic_segments  # preferred
        return create_dynamic_segments
    except Exception:
        try:
            # Some repos expose it via monolithic init or service layer
            from services.secret_sauce_pkg.__init__monolithic import create_dynamic_segments
            return create_dynamic_segments
        except Exception:
            return None

CDS = _import_create_dynamic_segments()

@pytest.mark.skipif(CDS is None, reason="create_dynamic_segments not available yet")
def test_segmentation_growth_and_caps():
    """
    A 10-minute transcript should yield a reasonable number of segments
    and each segment should respect hard caps (words/sec).
    """
    # Synthetic 10-min transcript with natural breaks (~600s total)
    # 30 chunks ~20s each with short sentences.
    transcript = []
    start = 0.0
    for i in range(30):
        text = (
            "Okay so here's the point. We tried one thing, it failed. "
            "Next we changed the plan and it worked. This matters because results compound."
        )
        end = start + 20.0
        transcript.append({"start": start, "end": end, "text": text})
        start = end + 0.2  # small pause

    segments = CDS(transcript, platform="tiktok")

    # Expect a healthy range (tune if you change thresholds)
    assert 5 <= len(segments) <= 120, f"Unexpected segment count: {len(segments)}"

    # Hard caps (mirror your policy: 5–100 words, 8–60 seconds)
    def _wc(s): return len(str(s.get("text","")).split())
    for s in segments:
        dur = float(s["end"] - s["start"])
        words = _wc(s)
        assert 8.0 <= dur <= 60.0, f"Duration cap failed: {dur}s"
        assert 5 <= words <= 100, f"Word cap failed: {words} words"
