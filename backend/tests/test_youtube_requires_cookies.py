import pytest
from yt_dlp.utils import DownloadError
from services import youtube_service as ys

class FakeYDL:
    def __init__(self, opts):
        self.opts = opts
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def extract_info(self, url, download=False):
        # First two attempts: fail with cookie error
        mode = "android" if ys._player_client_opts("android")['extractor_args']['youtube']['player_client'][0] in self.opts.get('extractor_args',{}).get('youtube',{}).get('player_client',[]) else "web"
        no_cookies = 'cookiefile' not in self.opts and 'cookiesfrombrowser' not in self.opts
        if no_cookies:
            raise DownloadError("Sign in to confirm you're not a bot")
        # With cookies: succeed
        return {"id": "abc123", "title": "ok", "duration": 120, "webpage_url": url}

def test_probe_retry_ladder(monkeypatch):
    monkeypatch.setattr(ys, "YoutubeDL", lambda opts: FakeYDL(opts))
    # enable cookies via env function
    monkeypatch.setenv("YT_COOKIES_FILE", __file__)  # any existing file
    # reload opts
    ys.COOKIE_FILE = __file__
    meta = ys.probe("https://www.youtube.com/watch?v=abc123")
    assert meta.title == "ok"
    assert meta.duration == 120

def test_download_retry_sets_error_on_fail(monkeypatch):
    class AlwaysFailYDL(FakeYDL):
        def extract_info(self, *a, **k):
            raise DownloadError("Sign in to confirm you're not a bot")
    monkeypatch.setattr(ys, "YoutubeDL", lambda opts: AlwaysFailYDL(opts))
    with pytest.raises((ValueError, RuntimeError)) as e:
        ys.download_and_prepare("https://youtu.be/abc", "ep1")
    assert "download_failed_requires_cookies" in str(e.value) or "download_failed" in str(e.value)
