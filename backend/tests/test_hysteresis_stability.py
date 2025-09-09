import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _import_create_dynamic_segments():
    try:
        from services.secret_sauce_pkg import create_dynamic_segments
        return create_dynamic_segments
    except Exception:
        try:
            from services.secret_sauce_pkg.__init__monolithic import create_dynamic_segments
            return create_dynamic_segments
        except Exception:
            return None

CDS = _import_create_dynamic_segments()

@pytest.mark.skipif(CDS is None, reason="create_dynamic_segments not available yet")
def test_boundaries_are_stable_under_small_edits():
    """
    Adding a filler word should NOT reshuffle cuts unless the score improvement
    beats the hysteresis threshold (≈0.03). We approximate by checking that
    most boundaries stay within 250ms.
    """
    base = [{"start": 0.0, "end": 18.0, "text": "This is the setup. Here comes the point. Then the punchline."},
            {"start": 18.2, "end": 38.4, "text": "We changed the plan and it worked. The results compounded quickly."},
            {"start": 38.6, "end": 59.0, "text": "So the takeaway is simple. Do the small things that drive outcomes."}]

    edited = [
        dict(base[0]),  # same
        {"start": 18.2, "end": 38.4, "text": "We, um, changed the plan and it worked. The results compounded quickly."},  # +filler
        dict(base[2]),
    ]

    seg_a = CDS(base, platform="tiktok")
    seg_b = CDS(edited, platform="tiktok")

    # Compare aligned boundaries (best-effort: list orders should be similar for small edits)
    # Require at least ~80% of boundaries to be within 250ms
    tol = 0.250  # seconds
    matched = 0
    total = min(len(seg_a), len(seg_b))
    for i in range(total):
        sa, sb = seg_a[i], seg_b[i]
        if abs((sa["start"] - sb["start"])) <= tol and abs((sa["end"] - sb["end"])) <= tol:
            matched += 1

    assert matched / max(1, total) >= 0.8, f"Excessive boundary jitter: matched={matched}, total={total}"
