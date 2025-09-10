import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _imports():
    try:
        from services.quality_filters import iou_time, nms_by_time
        return iou_time, nms_by_time
    except Exception:
        return None, None

iou_time, nms_by_time = _imports()

@pytest.mark.skipif(iou_time is None, reason="quality_filters not importable")
def test_iou_handles_none_and_bad_ranges():
    a = {"start": None, "end": 10}
    b = {"start": 2, "end": 8}
    c = {"start": 8, "end": 2}   # swapped
    d = {"start": 5, "end": 5}   # zero-length

    assert iou_time(a, b) == 0.0
    assert iou_time(b, a) == 0.0
    assert iou_time(c, b) > 0.0  # swapped normalized
    assert iou_time(d, b) == 0.0 # degenerate interval ignored

@pytest.mark.skipif(nms_by_time is None, reason="quality_filters not importable")
def test_nms_does_not_crash_and_keeps_best():
    cands = [
        {"start": None, "end": None, "final_score": 0.60, "payoff_score": 0.40},
        {"start": 0.0,  "end": 10.0, "final_score": 0.62, "payoff_score": 0.41},
        {"start": 2.0,  "end": 8.0,  "final_score": 0.58, "payoff_score": 0.45},
    ]
    kept = nms_by_time(cands, iou_thresh=0.5)
    # Must keep the highest-scoring overlapped clip (0.62), and it should not crash due to None
    assert any(abs(k.get("final_score",0)-0.62) < 1e-6 for k in kept)
    assert len(kept) >= 1
