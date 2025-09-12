import json
from services.episode_service import EpisodeService
from services.progress_writer import write_progress, get_progress

def test_finalize_marks_completed(monkeypatch, tmp_path):
    svc = EpisodeService()
    eid = "ep_test_finalize"
    write_progress(eid, "scoring", 100, "Found 4 clips")

    # should flip to completed, idempotently
    svc._finalize_episode_processing(eid, 4)
    p = get_progress(eid)
    progress_data = p.get("progress", {})
    assert progress_data["stage"] == "completed"
    assert progress_data["percent"] == 100

    # calling again shouldn't change terminal state or raise
    svc._finalize_episode_processing(eid, 4)
    p2 = get_progress(eid)
    progress_data2 = p2.get("progress", {})
    assert progress_data2["stage"] == "completed"
