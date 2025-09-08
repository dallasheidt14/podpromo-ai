"""
Readiness check with timeouts, caching, and concurrent checks
"""
import asyncio
import time
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from config.settings import UPLOAD_DIR

router = APIRouter()

# Cache for readiness results
_LAST = {"ts": 0.0, "result": None}
CACHE_SEC = 5.0
CHECK_BUDGET_MS = 500  # per check

async def _timeout(coro, ms):
    """Timeout a coroutine with budget in milliseconds"""
    return await asyncio.wait_for(coro, timeout=ms/1000)

async def check_disk_write(upload_dir):
    """Quick write/overwrite test"""
    start = time.perf_counter()
    try:
        p = upload_dir / ".ready_probe"
        p.write_text("ok", encoding="utf-8")
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return {"disk": "ok", "timing_ms": elapsed}
    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return {"disk": f"error: {e}", "timing_ms": elapsed}

async def check_progress_read(upload_dir):
    """Check if progress directory is accessible"""
    start = time.perf_counter()
    try:
        # Just stat the folder, not a deep walk
        _ = any(upload_dir.iterdir())
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return {"progress": "ok", "timing_ms": elapsed}
    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return {"progress": f"error: {e}", "timing_ms": elapsed}

async def check_ffmpeg():
    """Check FFmpeg availability (cached at startup)"""
    start = time.perf_counter()
    try:
        # Don't run heavy commands; just check if we can import
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=2)
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        if result.returncode == 0:
            return {"ffmpeg": "ok", "timing_ms": elapsed}
        else:
            return {"ffmpeg": "not_available", "timing_ms": elapsed}
    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return {"ffmpeg": f"error: {e}", "timing_ms": elapsed}

@router.get("/ready")
async def ready():
    """Readiness check with caching and timeouts"""
    now = time.time()
    
    # Return cached result if still fresh
    if _LAST["result"] and now - _LAST["ts"] < CACHE_SEC:
        return _LAST["result"]
    
    start = time.perf_counter()
    try:
        # Run checks concurrently with timeouts
        results = await asyncio.gather(
            _timeout(check_disk_write(Path(UPLOAD_DIR)), CHECK_BUDGET_MS),
            _timeout(check_progress_read(Path(UPLOAD_DIR)), CHECK_BUDGET_MS),
            _timeout(check_ffmpeg(), CHECK_BUDGET_MS),
            return_exceptions=True
        )
        
        # Process results and extract timings
        detail = {}
        timings = {}
        ok = True
        for r in results:
            if isinstance(r, Exception):
                ok = False
                detail.update({"error": str(r)})
            else:
                detail.update(r)
                # Extract timing if available
                if "timing_ms" in r:
                    service = [k for k in r.keys() if k != "timing_ms"][0]
                    timings[service] = r["timing_ms"]
        
        # Add timing breakdown
        if timings:
            detail["timings_ms"] = timings
        
        # Check if any critical services failed
        if not ok or any("error" in str(v) for v in detail.values()):
            ok = False
        
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        
        resp = JSONResponse(
            {"ok": ok, "detail": detail, "elapsed_ms": elapsed_ms},
            status_code=200 if ok else 503
        )
        
    except Exception as e:
        resp = JSONResponse(
            {"ok": False, "error": str(e), "elapsed_ms": round((time.perf_counter() - start) * 1000)},
            status_code=503
        )
    
    # Add no-cache headers
    resp.headers["Cache-Control"] = "no-store"
    
    # Cache the result
    _LAST.update(ts=now, result=resp)
    return resp
