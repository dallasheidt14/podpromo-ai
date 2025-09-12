from services.progress_writer import write_progress, get_progress
from services.progress_writer import _writer

def setup_function():
    _writer._last_updates.clear()

def test_stage_change_not_throttled_when_percent_same():
    eid = "ep1"
    write_progress(eid, "scoring", 100, "Found 4 clips")
    # Should allow stage change to completed even with same percent=100
    write_progress(eid, "completed", 100, "Ready: 4 clips")
    p = get_progress(eid)
    progress_data = p.get("progress", {})
    assert progress_data["stage"] == "completed"
    assert progress_data["percent"] == 100

def test_same_stage_small_delta_is_throttled():
    eid = "ep2"
    write_progress(eid, "transcribing", 20, "t")
    write_progress(eid, "transcribing", 22, "t")  # < _MIN_DELTA
    p = get_progress(eid)
    progress_data = p.get("progress", {})
    assert progress_data["stage"] == "transcribing"
    assert progress_data["percent"] == 20
