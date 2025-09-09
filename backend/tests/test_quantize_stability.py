import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def _import_api():
    # Prefer enhanced; fall back to basic
    try:
        from services.secret_sauce_pkg import find_viral_clips_enhanced as api
        return api
    except Exception:
        try:
            from services.secret_sauce_pkg import find_viral_clips as api
            return api
        except Exception:
            return None

def _import_hash():
    try:
        from services.secret_sauce_pkg import create_segment_hash
        return create_segment_hash
    except Exception:
        return None

API = _import_api()
HASH = _import_hash()

@pytest.mark.skipif(API is None or HASH is None, reason="API or hashing not available")
def test_topn_order_is_stable_across_reruns():
    transcript = [
        {"start": 0.0, "end": 14.0, "text": "Here is a useful tip about budgeting. The key is to automate savings."},
        {"start": 14.2, "end": 30.0, "text": "Another insight: pay the full balance to avoid interest snowballing."},
        {"start": 30.2, "end": 46.2, "text": "People often miss the payoff: small wins compound the most."},
        {"start": 46.4, "end": 62.0, "text": "So the takeaway is to start now. Don't wait for perfect timing."},
    ]
    audio = ""  # if your API accepts empty/optional audio

    out1 = API(transcript, audio, genre="general", platform="tiktok")
    out2 = API(transcript, audio, genre="general", platform="tiktok")

    # Extract clips from the response
    clips1 = out1.get('clips', []) if isinstance(out1, dict) else out1
    clips2 = out2.get('clips', []) if isinstance(out2, dict) else out2

    # Sort by final score desc, tie-break by hash/start time
    def key(x): return (-float(x.get("final_score", 0)), float(x.get("start", 0)))
    t1 = sorted(clips1, key=key)
    t2 = sorted(clips2, key=key)

    ids1 = [HASH(s) for s in t1[:10]]
    ids2 = [HASH(s) for s in t2[:10]]
    assert ids1 == ids2, "Top-N ordering changed across identical runs (quantize/clamp missing?)"
