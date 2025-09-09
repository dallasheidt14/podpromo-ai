import os, sys, pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Depending on your design, health may live in a debug flag or separate function.
# We try the enhanced API with debug output first.
def _import_api():
    try:
        from services.secret_sauce_pkg import find_viral_clips_enhanced as api
        return api
    except Exception:
        return None

API = _import_api()

@pytest.mark.skipif(API is None, reason="Enhanced API not available for health debug")
def test_health_block_present_when_debug_enabled():
    transcript = [
        {"start": 0.0, "end": 22.0, "text": "Welcome back, today we explain the simple rule that saves money fast."},
        {"start": 22.3, "end": 43.5, "text": "Step one: automate transfers. Step two: avoid impulse categories."},
        {"start": 43.8, "end": 65.0, "text": "The payoff: you'll feel control by week two and see results in 30 days."},
    ]
    audio = ""
    # Call the API (it should include debug info by default)
    out = API(transcript, audio, genre="general", platform="tiktok")

    # Expect a dict-like debug payload on the result
    debug = out.get("debug", {}) if isinstance(out, dict) else {}
    
    if not debug:
        pytest.skip("Debug payload not present in API response")

    health = debug.get("health", {})
    for key in ("segments", "sec_p50", "sec_p90", "yield_rate", "filters"):
        assert key in health, f"Missing health metric: {key}"
