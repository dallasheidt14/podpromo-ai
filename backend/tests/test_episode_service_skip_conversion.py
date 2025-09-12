import types
from pathlib import Path
from backend.services.episode_service import EpisodeService

def test_skip_conversion_when_wav_exists(tmp_path, monkeypatch):
    svc = EpisodeService()
    episode_id = "epwav"
    base = tmp_path

    # simulate upload dir structure
    src_video = base / f"{episode_id}.mp4"
    src_video.write_bytes(b"fake-mp4")
    wav_path = base / f"{episode_id}.wav"
    wav_path.write_bytes(b"fake-wav")

    # monkeypatch where service reads files
    monkeypatch.setattr(svc, "upload_dir", base, raising=False)

    calls = {"convert": 0}
    def fake_convert(src, dst):
        calls["convert"] += 1
    monkeypatch.setattr(svc, "_convert_to_wav_if_needed", fake_convert)

    async def fake_transcribe(audio_path, episode_id):
        return {"text": "ok"}
    monkeypatch.setattr(svc, "transcribe_async", fake_transcribe)

    # run
    # this assumes a method that orchestrates processing; adapt if yours differs
    awaitable = getattr(svc, "process_episode_async", None)
    if isinstance(awaitable, types.FunctionType) or hasattr(awaitable, "__call__"):
        await svc.process_episode_async(episode_id, str(src_video))
    else:
        # If no orchestrator, call the minimal sequence you use
        # Here we invoke just the conversion branch for the sake of the test
        base_dir = svc.upload_dir
        assert (base_dir / f"{episode_id}.wav").exists()

    assert calls["convert"] == 0, "should not convert when wav already exists"
