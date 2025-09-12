import importlib
from backend.services import progress_writer as pw

def setup_function():
    importlib.reload(pw)  # reset module state between tests

def test_monotonic_stage_order():
    eid = "ep1"
    pw.write_progress(eid, "downloading", 10)
    pw.write_progress(eid, "transcribing", 20)
    # Attempt to go backwards:
    pw.write_progress(eid, "downloading", 50)
    s = pw.get_progress(eid)
    assert s["stage"] == "transcribing"
    assert s["percent"] == 20

def test_percent_cannot_decrease():
    eid = "ep2"
    pw.write_progress(eid, "transcribing", 30)
    pw.write_progress(eid, "transcribing", 25)  # ignored
    s = pw.get_progress(eid)
    assert s["percent"] == 30

def test_throttle_small_deltas():
    eid = "ep3"
    pw.write_progress(eid, "scoring", 40)
    pw.write_progress(eid, "scoring", 43)  # <5% -> ignored
    s = pw.get_progress(eid)
    assert s["percent"] == 40
    pw.write_progress(eid, "scoring", 45)  # ==5% -> accepted
    s = pw.get_progress(eid)
    assert s["percent"] == 45

def test_freeze_after_terminal():
    eid = "ep4"
    pw.write_progress(eid, "completed", 100)
    # Any later writes are ignored
    pw.write_progress(eid, "scoring", 60)
    s = pw.get_progress(eid)
    assert s["stage"] == "completed"
    assert s["percent"] == 100
