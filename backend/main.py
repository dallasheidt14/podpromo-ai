"""
PodPromo AI - FastAPI Backend
Main application entry point with all API endpoints.
"""

import os
import uuid
import logging
import mimetypes
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor

# Hard guard: prevent accidental Torch/CUDA usage on Windows
os.environ.setdefault("ENABLE_TORCH_ALIGNMENT", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # prevents accidental torch cuda picks
os.environ.setdefault("FORCE_CPU_WHISPER", "1")  # force CPU for Whisper on this Windows host
if os.getenv("FORCE_CPU_WHISPER") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Ensure correct MIME type for .m4a files
mimetypes.add_type("audio/mp4", ".m4a")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from urllib.parse import urlparse

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import (
    Episode, TranscriptSegment, MomentScore, UploadResponse, 
    HealthCheck, ApiError, RenderRequest, UserSignup, UserLogin, 
    PaddleCheckout
)
from pydantic import BaseModel
from config_loader import get_config, reload_config, set_weights, load_preset
from services.episode_service import EpisodeService
from services.clip_score import ClipScoreService
from services.clip_service import ClipService
from services.caption_service import CaptionService
from services.loop_service import LoopService
from services.ab_test_service import ABTestService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PodPromo AI API",
    description="AI-powered podcast clip generation and scoring",
    version="1.0.0"
)

# --- Rate limiting (SlowAPI) ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Validation error handler ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error("TITLE_SAVE_422 payload failed: path=%s errors=%s",
                  request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

_EMERG = os.getenv("EMERGENCY_RATE_LIMIT", "")
_YT_LIMIT = os.getenv("YOUTUBE_UPLOAD_RATE", "10/hour")
_FILE_LIMIT = os.getenv("FILE_UPLOAD_RATE", "5/minute")
if _EMERG:
    _YT_LIMIT = _EMERG
    _FILE_LIMIT = _EMERG

# Global exception handler with context (no leakage in prod)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    ctx = {
        "method": request.method,
        "path": request.url.path,
        "query": str(request.query_params),
        "user_agent": request.headers.get("user-agent", "unknown"),
        "content_length": request.headers.get("content-length", "0"),
    }
    logger.error("Unhandled [%s]: %s | Context: %s",
                 request_id, exc, ctx, exc_info=True)
    if os.getenv("ENVIRONMENT") == "production":
        detail = "Internal server error"
    else:
        detail = str(exc)[:200]
    return JSONResponse(
        status_code=500, 
        headers={"X-Request-ID": request_id}, 
        content={
            "error": "internal_server_error",
            "detail": detail,
            "request_id": request_id,
        }
    )

# ---- CORS (environment-driven, strict in prod) -------------------------------------
from config.settings import CORS_ORIGINS, MAX_FILE_SIZE

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,  # Enable for JWT auth
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Add content-type validation middleware (scoped to /api/upload only)
@app.middleware("http")
async def validate_content_type(request: Request, call_next):
    # Only enforce multipart on the binary file upload endpoint
    if request.method == "POST" and request.url.path == "/api/upload":
        ctype = request.headers.get("content-type", "")
        if not ctype.startswith("multipart/form-data"):
            return JSONResponse(status_code=400, content={"error": "invalid_content_type"})
    return await call_next(request)

# Add timing middleware for progress endpoints
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/progress/"):
        import time
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000
        logging.info(f"Progress served in {dt:.1f}ms")
        
        # Add no-cache headers for progress endpoints
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response
    return await call_next(request)

# Cache headers middleware for preview files
@app.middleware("http")
async def preview_cache_headers(request, call_next):
    resp = await call_next(request)
    p = request.url.path
    
    # Handle preview files
    if p.startswith("/clips/previews/") and resp.status_code == 200:
        resp.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        resp.headers.setdefault("Accept-Ranges", "bytes")
        
        # Set correct MIME type for .m4a files
        if p.endswith(".m4a"):
            resp.headers.setdefault("Content-Type", "audio/mp4")
    
    # Handle caption files
    elif p.startswith("/clips/captions/") and resp.status_code == 200:
        resp.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        if p.endswith(".vtt"):
            resp.headers.setdefault("Content-Type", "text/vtt; charset=utf-8")
        elif p.endswith(".srt"):
            resp.headers.setdefault("Content-Type", "text/plain; charset=utf-8")
    
    return resp

# Import secure routers
from routes.downloads import router as downloads_router
from routes.secure_uploads import router as uploads_router
from routes.previews import router as previews_router

app.include_router(downloads_router)
app.include_router(uploads_router)
app.include_router(previews_router)

# Note: Removed public static file mounts for security
# Use /api/signed-download and /api/download/ endpoints with authentication instead

# Process pool for CPU-intensive scoring (feature flag)
USE_POOL = bool(int(os.getenv("SCORING_PROCESS_POOL", "0")))
POOL = ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) if USE_POOL else None

def run_scoring(ep, segments):
    """Run scoring either in process pool or inline based on feature flag"""
    if POOL:
        # Import here to avoid circular imports
        from services.clip_score import score_episode
        return POOL.submit(score_episode, ep, segments)  # returns Future
    else:
        # Import here to avoid circular imports
        from services.clip_score import score_episode
        return score_episode(ep, segments)               # runs inline in bg task

# Initialize services
episode_service = EpisodeService()
clip_score_service = ClipScoreService(episode_service)
clip_service = ClipService(episode_service, clip_score_service)
caption_service = CaptionService()
loop_service = LoopService()
ab_test_service = ABTestService()

# Initialize production services
from services.file_manager import FileManager
from services.queue_manager import QueueManager
from services.monitoring import MonitoringService
from services.auth_service import AuthService
from services.paddle_service import PaddleService
from config.settings import UPLOAD_DIR, OUTPUT_DIR

# Initialize production services immediately
try:
    file_manager = FileManager()
    logger.info("FileManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FileManager: {e}")
    file_manager = None

try:
    queue_manager = QueueManager()
    logger.info("QueueManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize QueueManager: {e}")
    queue_manager = None

try:
    monitoring_service = MonitoringService()
    logger.info("MonitoringService initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MonitoringService: {e}")
    monitoring_service = None

# Initialize auth and paddle services
auth_service = None
paddle_service = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Verify services are working on startup"""
    try:
        # Ensure directories exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Verify services are initialized and start background tasks
        if not file_manager:
            logger.warning("FileManager not available - file validation will be skipped")
        else:
            logger.info("FileManager verified and ready")
            file_manager.start_background_tasks()
            
        if not queue_manager:
            logger.warning("QueueManager not available - episodes will be processed directly")
        else:
            logger.info("QueueManager verified and ready")
            queue_manager.start_processing()
            
        if not monitoring_service:
            logger.warning("MonitoringService not available - metrics will not be recorded")
        else:
            logger.info("MonitoringService verified and ready")
        
        # Register job handlers for queue manager
        async def episode_processing_handler(job):
            """Handle episode processing jobs"""
            try:
                episode_id = job.metadata.get("episode_id")
                if episode_id:
                    logger.info(f"Processing episode {episode_id} from queue")
                    await episode_service.process_episode(episode_id)
                    return {"status": "completed", "episode_id": episode_id}
                else:
                    raise ValueError("No episode_id in job metadata")
            except Exception as e:
                logger.error(f"Episode processing job failed: {e}")
                raise
        
        queue_manager.register_handler("episode_processing", episode_processing_handler)
        logger.info("Registered episode_processing job handler")
        
        # Initialize authentication and payment services
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            auth_service = AuthService(supabase_url, supabase_key)
            paddle_service = PaddleService(auth_service)
            logger.info("Authentication and payment services initialized successfully")
        else:
            logger.warning("Supabase credentials not found, auth services disabled")
            auth_service = None
            paddle_service = None
        
        # File manager directories are created in __init__
        # Monitoring service starts automatically in __init__
        
        logger.info("Production services initialized successfully")
        
        # Run startup checks
        await startup_checks()
        
    except Exception as e:
        logger.error(f"Failed to initialize production services: {e}")

async def startup_checks():
    """Run critical startup checks"""
    logger.info("Running startup checks...")
    
    # Check required directories
    assert os.path.exists(UPLOAD_DIR) or os.path.exists(os.path.dirname(UPLOAD_DIR)), f"Upload directory {UPLOAD_DIR} not accessible"
    assert os.path.exists(OUTPUT_DIR) or os.path.exists(os.path.dirname(OUTPUT_DIR)), f"Output directory {OUTPUT_DIR} not accessible"
    
    # Check FFmpeg availability
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.warning("FFmpeg not found - audio processing may fail")
    except Exception as e:
        logger.warning(f"FFmpeg check failed: {e}")
    
    # Check Whisper model loading
    try:
        if episode_service.whisper_model is None:
            logger.warning("Whisper model not loaded - transcription may fail")
    except Exception as e:
        logger.warning(f"Whisper model check failed: {e}")
    
    # Check environment variables
    required_env_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            logger.warning(f"Environment variable {var} not set - some features may not work")
    
    logger.info("Startup checks completed")

# Rate limiting storage
rate_limit_storage = {}
RATE_LIMITS = {
    "upload": {"requests": 5, "window": 300},  # 5 uploads per 5 minutes
    "transcription": {"requests": 3, "window": 600},  # 3 transcriptions per 10 minutes
    "rendering": {"requests": 10, "window": 300},  # 10 renders per 5 minutes
}

# Validation functions
def validate_time_range(start: float, end: float, max_duration: float = 3600) -> None:
    """Validate start/end time parameters"""
    if start < 0:
        raise HTTPException(status_code=400, detail="Start time must be non-negative")
    if end <= start:
        raise HTTPException(status_code=400, detail="End time must be greater than start time")
    if end - start > max_duration:
        raise HTTPException(status_code=400, detail=f"Clip duration cannot exceed {max_duration} seconds")
    if end - start < 1:
        raise HTTPException(status_code=400, detail="Clip duration must be at least 1 second")

def check_rate_limit(operation: str, client_ip: str = "default") -> None:
    """Check if client has exceeded rate limit for operation"""
    import time
    current_time = time.time()
    
    if operation not in RATE_LIMITS:
        return
    
    limit = RATE_LIMITS[operation]
    key = f"{operation}:{client_ip}"
    
    if key not in rate_limit_storage:
        rate_limit_storage[key] = []
    
    # Clean old requests outside the window
    rate_limit_storage[key] = [
        req_time for req_time in rate_limit_storage[key] 
        if current_time - req_time < limit["window"]
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[key]) >= limit["requests"]:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Maximum {limit['requests']} {operation} requests per {limit['window']} seconds"
        )
    
    # Add current request
    rate_limit_storage[key].append(current_time)

# Request validation middleware
@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    """Validate request size and other security checks"""
    # Check content length for file uploads
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_FILE_SIZE:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=413, 
                    content={"error": f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"}
                )
        except ValueError:
            pass
    
    # Check for suspicious patterns
    if request.url.path.startswith("/api/upload"):
        # Additional validation for upload endpoints
        pass
    
    response = await call_next(request)
    return response

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup production services on shutdown"""
    try:
        # Monitoring service cleanup is handled automatically
        logger.info("Production services shutdown successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown production services: {e}")

# Test transcription endpoint
@app.post("/api/test-transcription")
async def test_transcription(file_id: str = Form(...), request: Request = None):
    """Test transcription with an existing file without uploading"""
    try:
        # Rate limiting for transcription
        client_ip = request.client.host if request else "default"
        check_rate_limit("transcription", client_ip)
        # Find the file in uploads directory - sanitize file_id to prevent path traversal
        import os
        file_id = os.path.basename(file_id)  # Remove any directory components
        file_path = f"{UPLOAD_DIR}/{file_id}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Create a test episode
        test_episode_id = f"test_{file_id}"
        
        # Create episode object
        episode = Episode(
            id=test_episode_id,
            filename=file_id,
            original_name=file_id,
            size=os.path.getsize(file_path),
            status="processing"
        )
        
        # Store in service
        episode_service.episodes[test_episode_id] = episode
        episode.audio_path = file_path
        
        # Test transcription
        transcript = await episode_service._transcribe_audio(file_path, test_episode_id)
        
        return {
            "success": True,
            "episode_id": test_episode_id,
            "segments": len(transcript),
            "duration": transcript[-1].end - transcript[0].start if transcript else 0,
            "sample_segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text[:100] + "..." if len(seg.text) > 100 else seg.text
                }
                for seg in transcript[:3]  # First 3 segments
            ]
        }
        
    except Exception as e:
        logger.error(f"Test transcription failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Include health and ready routers
from routes.health import router as health_router
from routes.ready import router as ready_router

app.include_router(health_router)
app.include_router(ready_router)

# Configuration endpoints
@app.get("/config/get")
async def get_config_endpoint():
    """Get current configuration"""
    try:
        config = get_config()
        return {"ok": True, "config": config}
    except Exception as e:
        logger.error(f"Config get failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/config/set-weights")
async def set_weights_endpoint(weights: Dict[str, float]):
    """Set custom weights for scoring"""
    try:
        result = set_weights(weights)
        return {"ok": True, "weights": result}
    except Exception as e:
        logger.error(f"Weight setting failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/config/reload")
async def reload_config_endpoint():
    """Reload configuration from files"""
    try:
        config = reload_config()
        return {"ok": True, "config": config, "lexicons": config.get("lexicons", {})}
    except Exception as e:
        logger.error(f"Config reload failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/debug/test-detectors")
async def test_detectors(body: dict):
    """Test individual detectors with sample text"""
    try:
        txt = body.get("text", "")
        from services.secret_sauce_pkg import _hook_score, _payoff_presence, _info_density, _ad_penalty
        # Test original hook detection
        hook_val = _hook_score(txt)
        # Test original ad detection
        ad_result = _ad_penalty(txt)
        return {
            "hook": hook_val,
            "payoff": _payoff_presence(txt),
            "info": _info_density(txt),
            "ad_flag": ad_result["flag"],
            "ad_penalty": ad_result["penalty"],
            "ad_reason": ad_result["reason"]
        }
    except Exception as e:
        logger.error(f"Detector test failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/config/load-preset")
async def load_preset_endpoint(preset_name: str):
    """Load a preset configuration"""
    try:
        result = load_preset(preset_name)
        return {"ok": True, "weights": result}
    except Exception as e:
        logger.error(f"Preset loading failed: {e}")
        return {"ok": False, "error": str(e)}

# Production service endpoints
@app.get("/api/system/status")
async def get_system_status():
    """Get detailed system status and metrics"""
    try:
        if not all([monitoring_service, queue_manager, file_manager]):
            return {"ok": False, "error": "Production services not initialized"}
        
        # Get basic metrics (non-async)
        metrics = monitoring_service.get_metrics_summary()
        
        # Get queue and file stats (async)
        queue_status = await queue_manager.get_queue_stats()
        file_status = await file_manager.get_storage_stats()
        
        return {
            "ok": True,
            "metrics": metrics,
            "queue": queue_status,
            "files": file_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System status failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/api/system/cleanup")
async def cleanup_files():
    """Trigger file cleanup and maintenance"""
    try:
        if not file_manager:
            return {"ok": False, "error": "File manager not initialized"}
        
        # Clean up old files
        cleanup_result = await file_manager.cleanup_old_files()
        
        return {
            "ok": True,
            "uploads_cleaned": cleanup_result,
            "message": "Cleanup completed successfully"
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/system/queue")
async def get_queue_status():
    """Get current job queue status"""
    try:
        if not queue_manager:
            return {"ok": False, "error": "Queue manager not initialized"}
        
        status = await queue_manager.get_queue_stats()
        return {"ok": True, "queue": status}
    except Exception as e:
        logger.error(f"Queue status failed: {e}")
        return {"ok": False, "error": str(e)}

# Metrics endpoint
@app.post("/metrics/log-choice")
async def log_choice(choice_data: Dict):
    """Log user choice for learning"""
    try:
        # In production, this would go to a database
        logger.info(f"User choice logged: {choice_data}")
        return {"ok": True, "logged": True}
    except Exception as e:
        logger.error(f"Choice logging failed: {e}")
        return {"ok": False, "error": str(e)}

# Main API endpoints
@app.post("/api/upload")
@limiter.limit(_FILE_LIMIT)
async def upload_episode(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    request: Request = None
    # TODO: Add auth dependency when JWT is configured
    # user = Depends(require_user_dev)
):
    """Upload and process a podcast episode"""
    try:
        # Import progress writer
        from services.progress_writer import write_progress
        
        # Check if services are initialized
        if not episode_service:
            logger.error("EpisodeService not initialized")
            raise HTTPException(status_code=503, detail="Service not ready. Please try again.")
        
        # Rate limiting for uploads
        client_ip = request.client.host if request else "default"
        check_rate_limit("upload", client_ip)
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Log the upload attempt
        logger.info(f"Upload attempt: {file.filename}, size: {getattr(file, 'size', 'unknown')}")
        
        # Validate file using FileManager (if available)
        if file_manager:
            # Get file size from content length or read file
            file_size = 0
            try:
                # Try to get size from content length header
                if hasattr(file, 'size') and file.size:
                    file_size = file.size
                else:
                    # Read file to get size
                    content = await file.read()
                    file_size = len(content)
                    # Reset file position for later processing
                    await file.seek(0)
            except Exception as e:
                logger.warning(f"Could not determine file size: {e}")
                file_size = 0
            
            try:
                validation_result = await file_manager.validate_upload(file.filename, file_size)
                if not validation_result["valid"]:
                    raise HTTPException(status_code=400, detail=validation_result["error"])
            except Exception as e:
                logger.error(f"File validation failed: {e}")
                raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
        else:
            logger.error("FileManager not available - file validation required")
            raise HTTPException(status_code=503, detail="File validation service not available")
        
        # Create episode and start processing
        episode_id = str(uuid.uuid4())
        episode = await episode_service.create_episode(episode_id, file, file.filename)
        
        # Write initial progress
        write_progress(episode_id, "uploading", 1, "File received")
        
        # Start processing in background (don't await)
        if background_tasks:
            background_tasks.add_task(episode_service.process_episode, episode.id)
            logger.info(f"Episode {episode.id} queued for background processing")
        else:
            # Fallback: process directly if no background tasks
            logger.warning("No background tasks available, processing episode directly")
            await episode_service.process_episode(episode.id)
        
        # Log success to monitoring (if available)
        if monitoring_service:
            try:
                monitoring_service.record_metric("upload_success", 1)
            except Exception as e:
                logger.warning(f"Failed to record monitoring metric: {e}")
        
        return {
            "ok": True, 
            "episodeId": episode.id, 
            "message": "Upload successful and processing started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        # Log error to monitoring service (if available)
        if monitoring_service:
            try:
                monitoring_service.record_error("upload_failed", str(e))
            except Exception as monitoring_error:
                logger.warning(f"Failed to record monitoring error: {monitoring_error}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# YouTube upload feature flag
YT_ENABLED = os.getenv("YT_UPLOAD_ENABLED", "true").lower() in {"1", "true", "yes"}

class UploadYouTubeResponse(BaseModel):
    ok: bool
    episode_id: str

@app.post("/api/upload-youtube", response_model=UploadYouTubeResponse)
@limiter.limit(_YT_LIMIT)
async def upload_youtube_episode(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    request: Request = None
):
    """Upload and process a YouTube video as a podcast episode"""
    if not YT_ENABLED:
        raise HTTPException(status_code=503, detail="youtube_disabled")
    
    try:
        from services.progress_writer import write_progress
        from services.youtube_service import download_and_prepare, map_error
        
        # Check if services are initialized
        if not episode_service:
            logger.error("EpisodeService not initialized")
            raise HTTPException(status_code=503, detail="Service not ready. Please try again.")
        
        # Rate limiting for uploads
        client_ip = request.client.host if request else "default"
        check_rate_limit("upload", client_ip)
        
        episode_id = str(uuid.uuid4())
        write_progress(episode_id, "queued", 1, "Queued YouTube job")

        async def _job():
            try:
                from services.preview_service import ensure_preview
                from services.episode_service import EpisodeService
                from services.clip_score import score_episode
                from services.title_service import generate_titles
                from services.progress_writer import write_progress
                from pathlib import Path

                # 1) Download + prep
                try:
                    info = download_and_prepare(url, episode_id)
                except Exception as e:
                    # Surface YouTube download failures immediately
                    from services.youtube_service import map_error
                    _, key = map_error(e)
                    msg = (
                        "This video needs login/age verification. "
                        "Try a different link or contact support to enable advanced access."
                        if key in {"download_requires_login", "download_failed_requires_cookies"}
                        else "YouTube download failed. Please try again or upload the file directly."
                    )
                    write_progress(episode_id, "error", 0, msg)
                    raise

                # 2) Create episode record and process
                write_progress(episode_id, "transcribing", 20, "Transcribing audio")
                ep_service: EpisodeService = episode_service
                
                # Create a mock episode object for the downloaded files
                from models import Episode
                ep = Episode(
                    id=episode_id,
                    filename=f"{episode_id}.mp4",
                    original_name=info["meta"]["title"],
                    size=0,  # Will be updated
                    status="processing",
                    audio_path=info["wav_path"],
                    title=info["meta"]["title"]
                )
                
                # Add to episode service
                ep_service.episodes[episode_id] = ep
                ep_service._save_episode(ep)
                
                # Process the episode (transcribe + analyze + score clips)
                await ep_service.process_episode(episode_id)
                
                # Get the processed episode
                ep = await ep_service.get_episode(episode_id)
                if not ep:
                    raise RuntimeError("Failed to process episode")

                # Get the clips that were generated during processing
                scored = ep.clips if ep.clips else []
                write_progress(episode_id, "scoring", 60, f"Found {len(scored)} clips")

                # 4) Ensure **video** previews exist for UI (20s default)
                #    Non-breaking: if preview already present, `ensure_preview` is a no-op
                for c in scored:
                    preview_name = ensure_preview(
                        source_media=Path(info["video_path"]),
                        episode_id=episode_id,
                        clip_id=c["id"],
                        start_sec=float(c["start"]),
                        end_sec=float(c["end"]),
                        max_preview_sec=20.0
                    )
                    # Store the preview filename in the clip data
                    if preview_name:
                        c["preview_name"] = preview_name

                # 5) Titles (existing)
                # Generate titles for each scored clip
                for clip in scored:
                    if 'text' in clip:
                        titles = generate_titles(
                            text=clip['text'],
                            platform="tiktok_reels",
                            episode_id=episode_id,
                            clip_id=clip['id']
                        )
                        # Store titles in the clip (if needed)
                        clip['titles'] = titles
                
                # Save the updated clips with preview URLs back to the episode
                ep.clips = scored
                ep_service._save_episode(ep)
                
                write_progress(episode_id, "completed", 100, "Completed")
            except Exception as e:
                # Map error for a nicer message
                from services.youtube_service import map_error
                _, err_code = map_error(e)
                # surface a user-actionable hint for cookie-gated videos
                msg = "YouTube download failed"
                if err_code == "download_requires_login":
                    msg = "This video requires login or has age restrictions. Try a different video or contact support for advanced setup."
                elif err_code == "invalid_url":
                    msg = "Invalid YouTube URL format"
                elif err_code == "too_short":
                    msg = "Video is too short (minimum 60 seconds)"
                elif err_code == "too_long":
                    msg = "Video is too long (maximum 4 hours)"
                elif err_code == "live_stream_not_supported":
                    msg = "Live streams are not supported"
                
                # surface the error to clients polling progress
                try:
                    from services.progress_writer import write_progress
                    write_progress(episode_id, "error", 0, msg)
                finally:
                    # log full traceback to server logs
                    import logging, traceback
                    logging.getLogger(__name__).error("YT job failed for %s\n%s", episode_id, traceback.format_exc())

        if background_tasks:
            background_tasks.add_task(_job)
        else:
            await _job()

        return UploadYouTubeResponse(ok=True, episode_id=episode_id)
        
    except Exception as e:
        from services.youtube_service import map_error
        status, code = map_error(e)
        # progress failure is best-effort
        try:
            from services.progress_writer import write_progress
            write_progress(episode_id, "error", 0, code)
        except Exception:
            pass
        raise HTTPException(status_code=status, detail=code)

@app.get("/api/episodes")
async def list_episodes():
    """Get list of all loaded episodes"""
    try:
        episodes = await episode_service.list_episodes()
        
        # Convert episodes to summary format
        episode_summaries = []
        for episode in episodes:
            summary = {
                "episode_id": episode.id,
                "title": getattr(episode, "title", None),
                "created_at": episode.created_at.isoformat() if hasattr(episode, "created_at") and episode.created_at else None,
                "duration_s": getattr(episode, "duration_s", None),
                "status": episode.status
            }
            episode_summaries.append(summary)
        
        return {"ok": True, "episodes": episode_summaries}
    except Exception as e:
        logger.error(f"Error listing episodes: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/candidates")
async def get_candidates(
    episode_id: str | None = Query(None, description="Specific episode ID to get candidates for"),
    genre: str | None = Query(None, description="Genre for optimization (comedy, fantasy_sports, sports, true_crime, business, news_politics, education, health_wellness)"),
    debug: bool = Query(False, description="Enable debug mode for detailed candidate information")
):
    """Get AI-scored clip candidates with optional platform/tone optimization"""
    try:
        # Set debug environment variable
        os.environ["PODPROMO_DEBUG_ALL"] = "1" if debug else "0"
        
        # Get the target episode (specific ID or latest)
        if episode_id:
            # Get specific episode by ID
            target_episode = await episode_service.get_episode(episode_id)
            if not target_episode:
                return {"ok": False, "error": f"Episode {episode_id} not found."}
        else:
            # Get the most recent episode
            episodes = await episode_service.list_episodes()
            if not episodes:
                return {"ok": False, "error": "No episodes found. Upload a file first."}
            target_episode = episodes[-1]
        
        if target_episode.status != "completed":
            return {"ok": False, "error": "Episode still processing. Please wait."}
        
        # Get candidates using ClipScore with platform/genre optimization
        # Auto-recommend the best platform based on content analysis
        from services.secret_sauce_pkg import find_viral_clips, PLATFORM_GENRE_MULTIPLIERS
        
        # Analyze content to recommend best platform
        recommended_platform = "tiktok"  # Default fallback
        
        if genre and genre in ["comedy", "entertainment"]:
            # Comedy works best on TikTok
            recommended_platform = "tiktok"
        elif genre and genre in ["education", "business"]:
            # Educational/business content works better on YouTube Shorts
            recommended_platform = "youtube_shorts"
        elif genre and genre in ["fantasy_sports", "sports"]:
            # Sports content works better on YouTube Shorts (longer format)
            recommended_platform = "youtube_shorts"
        elif genre and genre in ["true_crime", "news_politics"]:
            # True crime/news works well on both, but slightly better on TikTok
            recommended_platform = "tiktok"
        else:
            # For auto-detected or unknown genres, analyze content characteristics
            # This is a simplified version - in production you'd analyze actual content
            recommended_platform = "youtube_shorts"  # Safer default for most content
        
        # Map to frontend platform name
        platform_mapping = {
            "tiktok": "tiktok_reels",
            "youtube_shorts": "shorts", 
            "linkedin": "linkedin_sq"
        }
        frontend_platform = platform_mapping.get(recommended_platform, "tiktok_reels")
        
        # Get candidates using the auto-recommended platform
        candidates, _ = await clip_score_service.get_candidates(
            target_episode.id, 
            platform=frontend_platform, 
            genre=genre
        )
        
        if not candidates:
            return {"ok": False, "error": "No candidates found for this episode."}
        
        # Add platform mapping info to response
        from services.secret_sauce_pkg import resolve_platform
        backend_platform = resolve_platform(frontend_platform)
        
        # Record successful candidate generation (if monitoring available)
        if monitoring_service:
            monitoring_service.record_metric("candidates_generated", len(candidates))
            monitoring_service.record_metric("candidate_generation_success", 1)
        
        return {
            "ok": True, 
            "candidates": candidates,
            "optimization": {
                "recommended_platform": frontend_platform,
                "frontend_genre": genre,
                "backend_platform": backend_platform,
                "description": f"Auto-recommended {frontend_platform.replace('_', ' ')}" + (f" for {genre.replace('_', ' ')} content" if genre else " based on content analysis")
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get candidates: {e}")
        if monitoring_service:
            monitoring_service.record_error("candidates_failed", str(e))
        return {"ok": False, "error": str(e)}

@app.post("/api/render-one")
async def render_one_clip(
    start: float = Form(...), 
    end: float = Form(...),
    style: str = Form("bold"),
    captions: bool = Form(True),
    punch_ins: bool = Form(True),
    loop_seam: bool = Form(False),
    request: Request = None
):
    """Render a single clip with enhanced options"""
    try:
        # Rate limiting for rendering
        client_ip = request.client.host if request else "default"
        check_rate_limit("rendering", client_ip)
        
        # Validate time range
        validate_time_range(start, end, max_duration=300)  # 5 minutes max for clips
        # For MVP, get the most recent episode
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": False, "error": "No episodes found. Upload a file first."}
        
        latest_episode = episodes[-1]
        if latest_episode.status != "completed":
            return {"ok": False, "error": "Episode still processing. Please wait."}
        
        # Generate clip using ClipService with enhanced options
        clip_id = str(uuid.uuid4())
        output_filename = f"clip_{clip_id}.mp4"
        
        # Start rendering in background with enhanced options
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            clip_service.render_clip_enhanced,
            clip_id,
            latest_episode.audio_path,
            start,
            end,
            output_filename,
            style=style,
            captions=captions,
            punch_ins=punch_ins,
            loop_seam=loop_seam
        )
        
        return {
            "ok": True,
            "clip_id": clip_id,
            "output": output_filename,
            "status": "rendering",
            "options": {
                "style": style,
                "captions": captions,
                "punch_ins": punch_ins,
                "loop_seam": loop_seam
            }
        }
        
    except Exception as e:
        logger.error(f"Render failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/api/create-ab-tests")
async def create_ab_tests(
    start: float = Form(...),
    end: float = Form(...)
):
    """Create A/B test versions for hook testing"""
    try:
        # Validate time range
        validate_time_range(start, end, max_duration=300)  # 5 minutes max for clips
        # Get the most recent episode
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": False, "error": "No episodes found. Upload a file first."}
        
        latest_episode = episodes[-1]
        if latest_episode.status != "completed":
            return {"ok": False, "error": "Episode still processing. Please wait."}
        
        # Create A/B test versions
        result = ab_test_service.create_ab_tests(
            latest_episode.audio_path,
            latest_episode.transcript,
            start,
            end
        )
        
        if result["success"]:
            return {
                "ok": True,
                "ab_tests": result["ab_tests"]
            }
        else:
            return {"ok": False, "error": result.get("error", "A/B test creation failed")}
        
    except Exception as e:
        logger.error(f"A/B test creation failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/api/log-ab-choice")
async def log_ab_choice(choice_data: Dict):
    """Log which A/B test version the user chose"""
    try:
        result = ab_test_service.log_ab_test_choice(
            choice_data["test_id"],
            choice_data["choice"],
            choice_data["clip_id"]
        )
        return {"ok": True, "logged": result["logged"]}
    except Exception as e:
        logger.error(f"A/B choice logging failed: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/history")
async def get_clip_history(limit: int = 50):
    """Get clip creation history"""
    try:
        items = clip_service.list_history(limit=limit)
        return {"ok": True, "items": items}
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/transcript/{episode_id}")
async def get_transcript(episode_id: str):
    """Get transcript for an episode (for debugging)"""
    try:
        episode = await episode_service.get_episode(episode_id)
        if not episode:
            return {"ok": False, "error": "Episode not found"}
        
        if not episode.transcript:
            return {"ok": False, "error": "No transcript available"}
        
        # Return transcript segments
        segments = []
        for segment in episode.transcript:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": segment.confidence
            })
        
        return {
            "ok": True,
            "episode_id": episode_id,
            "status": episode.status,
            "duration": episode.duration,
            "segments": segments,
            "total_segments": len(segments)
        }
    except Exception as e:
        logger.error(f"Failed to get transcript: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/debug-episodes")
async def debug_episodes():
    """Debug endpoint to see what episodes are in memory"""
    try:
        episodes = list(episode_service.episodes.values())
        episode_info = []
        for episode in episodes:
            episode_info.append({
                "id": episode.id,
                "status": episode.status,
                "has_transcript": bool(episode.transcript),
                "transcript_segments": len(episode.transcript) if episode.transcript else 0,
                "duration": episode.duration
            })
        
        return {
            "ok": True,
            "total_episodes": len(episodes),
            "episodes": episode_info
        }
    except Exception as e:
        logger.error(f"Failed to debug episodes: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/api/reload-episodes")
async def reload_episodes():
    """Force reload episodes from storage"""
    try:
        # Clear current episodes
        episode_service.episodes.clear()
        
        # Reload from storage
        episode_service._load_episodes()
        
        episodes = list(episode_service.episodes.values())
        episode_info = []
        for episode in episodes:
            episode_info.append({
                "id": episode.id,
                "status": episode.status,
                "has_transcript": bool(episode.transcript),
                "transcript_segments": len(episode.transcript) if episode.transcript else 0,
                "duration": episode.duration
            })
        
        return {
            "ok": True,
            "message": f"Reloaded {len(episodes)} episodes",
            "total_episodes": len(episodes),
            "episodes": episode_info
        }
    except Exception as e:
        logger.error(f"Failed to reload episodes: {e}")
        return {"ok": False, "error": str(e)}

@app.head("/api/episodes/{episode_id}/clips")
async def head_episode_clips(episode_id: str):
    """HEAD request to check if clips are ready"""
    try:
        # Check readiness first using progress service
        try:
            from services.progress_service import progress_service
            progress = progress_service.get_progress(episode_id)
            
            # Check if clips are ready
            is_ready = False
            if progress and isinstance(progress, dict):
                # Extract progress data from the service response
                progress_data = progress.get("progress", {})
                if progress_data.get("stage") == "completed":
                    is_ready = True
                elif int(progress_data.get("percent", 0)) >= 100:
                    is_ready = True
            
            if not is_ready:
                # Check for persisted output files
                from pathlib import Path
                from config.settings import UPLOAD_DIR
                ep_path = Path(UPLOAD_DIR) / episode_id
                clips_file = ep_path / "clips.json"
                clips_dir = ep_path / "clips"
                is_ready = clips_file.exists() or clips_dir.exists()
        except Exception:
            is_ready = False
        
        if is_ready:
            from fastapi.responses import Response
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        else:
            from fastapi.responses import Response
            return Response(status_code=202, headers={"Retry-After": "15", "Cache-Control": "no-store"})
    except Exception:
        from fastapi.responses import Response
        return Response(status_code=202, headers={"Retry-After": "15", "Cache-Control": "no-store"})

@app.get("/api/episodes/{episode_id}/clips")
async def get_episode_clips(episode_id: str, regenerate: bool = False, background_tasks: BackgroundTasks = None, request: Request = None):
    """Get clips for a specific episode - returns 202 if not ready, 200 with data if ready"""
    try:
        # Check readiness first using progress service
        try:
            from services.progress_service import progress_service
            progress = progress_service.get_progress(episode_id)
            
            # Check if clips are ready
            is_ready = False
            if progress and isinstance(progress, dict):
                # Extract progress data from the service response
                progress_data = progress.get("progress", {})
                if progress_data.get("stage") == "completed":
                    is_ready = True
                elif int(progress_data.get("percent", 0)) >= 100:
                    is_ready = True
            
            if not is_ready:
                # Check for persisted output files
                from pathlib import Path
                from config.settings import UPLOAD_DIR
                ep_path = Path(UPLOAD_DIR) / episode_id
                clips_file = ep_path / "clips.json"
                clips_dir = ep_path / "clips"
                is_ready = clips_file.exists() or clips_dir.exists()
            
            if not is_ready:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    {"ok": False, "ready": False, "message": "Scoring in progress"},
                    status_code=202,
                    headers={"Retry-After": "15", "Cache-Control": "no-store"}
                )
        except Exception as e:
            logger.warning(f"Progress check failed: {e}, proceeding with episode check")
        
        # Get the episode from the service
        episode = await episode_service.get_episode(episode_id)
        if not episode:
            return {
                "ok": False,
                "clips": [],
                "count": 0,
                "episode_id": episode_id,
                "message": "Episode not found"
            }
        
        # Get clips from episode (already generated during processing)
        try:
            if hasattr(episode, 'clips') and episode.clips:
                # Use already generated clips from episode processing
                clips = episode.clips
                # For cached clips, we need to add the new metadata
                for clip in clips:
                    if "protected_long" not in clip:
                        clip["protected_long"] = clip.get("protected", False)
                    if "duration" not in clip:
                        clip["duration"] = round(clip.get('end', 0) - clip.get('start', 0), 2)
                    if "pl_v2" not in clip:
                        clip["pl_v2"] = clip.get("platform_length_score_v2", 0.0)
                    if "finished_thought" not in clip:
                        clip["finished_thought"] = clip.get("text", "").strip().endswith(('.', '!', '?'))
                    if "rank_primary" not in clip:
                        clip["rank_primary"] = clips.index(clip)
                
                # Choose default clip for cached clips
                from services.clip_score import choose_default_clip
                default_clip_id = choose_default_clip(clips)
                logger.info(f"Using cached clips for episode {episode_id}: {len(clips)} clips")
            else:
                # Fallback: generate clips if not available (shouldn't happen)
                logger.warning(f"No cached clips found for episode {episode_id}, generating now...")
                result = await clip_score_service.get_candidates(episode_id)
                if isinstance(result, tuple) and len(result) == 2:
                    clips, default_clip_id = result
                else:
                    clips = result if isinstance(result, list) else []
                    default_clip_id = None
            
            # Handle empty clips gracefully
            if not clips:
                logger.warning(f"REASON=EMPTY_AFTER_SALVAGE: episode {episode_id} has 0 clips")
                return {
                    "ok": True,
                    "clips": [],
                    "count": 0,
                    "episode_id": episode_id,
                    "message": "Ready: 0 clips"
                }
            
            # Import preview service
            from services.preview_service import ensure_preview, get_episode_media_path
            
            # Get episode media path for preview generation
            source_media = get_episode_media_path(episode_id)
            
            # Format clips for frontend
            formatted_clips = []
            for i, clip in enumerate(clips):
                clip_id = clip.get("id", f"clip_{i+1}")
                start_sec = float(clip.get("start", 0))
                end_sec = float(clip.get("end", 0))
                
                # Check if preview already exists or generate it
                preview_name = None
                if source_media and source_media.exists():
                    # Try to get existing preview first
                    from services.preview_service import build_preview_filename, ensure_preview
                    from pathlib import Path
                    from config.settings import OUTPUT_DIR
                    
                    preview_filename = build_preview_filename(episode_id, clip_id)
                    preview_path = Path(OUTPUT_DIR) / "previews" / preview_filename
                    
                    if preview_path.exists() and preview_path.stat().st_size > 0:
                        preview_name = preview_filename
                    else:
                        # Generate preview synchronously for immediate availability
                        try:
                            generated_name = ensure_preview(
                                source_media=source_media,
                                episode_id=episode_id,
                                clip_id=clip_id,
                                start_sec=start_sec,
                                end_sec=end_sec,
                                max_preview_sec=min(30.0, end_sec - start_sec or 20.0),
                                pad_start_sec=-0.08,  # Fix E: -80ms start padding
                                pad_end_sec=0.24,     # Fix E: +240ms end padding
                            )
                            if generated_name:
                                preview_name = generated_name
                                logger.info(f"Generated preview for clip {clip_id}: {preview_name}")
                            else:
                                logger.warning(f"Failed to generate preview for clip {clip_id}")
                        except Exception as e:
                            logger.error(f"Preview generation failed for clip {clip_id}: {e}")
                            # Continue without preview
                
                # Calculate duration and add debug fields
                duration_s = round(end_sec - start_sec, 1)
                preview_duration_s = None
                slice_src = "variant_cut" if duration_s > 12.0 else "default"
                
                # Try to get actual preview duration if preview exists
                if preview_name and preview_path.exists():
                    try:
                        import subprocess
                        result = subprocess.run([
                            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(preview_path)
                        ], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            preview_duration_s = round(float(result.stdout.strip()), 1)
                    except:
                        pass
                
                formatted_clips.append({
                    "id": clip_id,
                    "title": clip.get("title", f"Clip {i+1}"),
                    "text": clip.get("text", ""),
                    "score": clip.get("score", 0),
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "duration": duration_s,
                    "duration_s": duration_s,  # Explicit field for frontend
                    "preview_duration_s": preview_duration_s,  # Server-measured duration
                    "preview_name": preview_name,  # Add preview filename to the response
                    "slice_src": slice_src,  # "variant_cut" vs "default"
                    "protected": clip.get("protected", False),  # Protected flag
                    "protected_long": clip.get("protected_long", False),  # Fix A: protected_long flag
                    "rank_primary": clip.get("rank_primary", i),  # Fix A: explicit ranking
                    "reason": clip.get("reason", ""),
                    "features": clip.get("features", {}),
                    "is_advertisement": clip.get("is_advertisement", False)
                })
            
            # Server-side assertions (don't regress checks)
            previews_generated = [c for c in formatted_clips if c.get("preview_name")]
            logger.info(f"PREVIEW_ASSERT: generated {len(previews_generated)}/{len(formatted_clips)} previews")
            
            # Check for duration drift warnings
            for clip in formatted_clips:
                if clip.get("preview_duration_s") and clip.get("duration_s"):
                    intended = clip["duration_s"]
                    actual = clip["preview_duration_s"]
                    drift = abs(actual - intended)
                    if drift > 0.4:  # Warn if drift > 400ms
                        logger.warning(f"PREVIEW_DRIFT: clip {clip['id']} intended={intended:.3f}s actual={actual:.3f}s drift={drift:.3f}s")
            
            # Add ETag and caching for immutable clips
            import json
            import hashlib
            from fastapi.responses import JSONResponse
            
            payload = {"ok": True, "ready": True, "clips": formatted_clips, "default_clip_id": default_clip_id}
            body = json.dumps(formatted_clips, sort_keys=True, separators=(",",":")).encode()
            etag = hashlib.md5(body).hexdigest()
            
            # Check If-None-Match header for 304 responses
            if_none_match = request.headers.get("if-none-match") if request else None
            if if_none_match == etag:
                from fastapi.responses import Response
                return Response(status_code=304, headers={"ETag": etag})
            
            return JSONResponse(
                payload,
                headers={
                    "ETag": etag,
                    "Cache-Control": "no-store, max-age=0"
                }
            )
            
        except Exception as e:
            logger.error(f"Clip generation failed: {e}")
            return {
                "ok": False,
                "clips": [],
                "count": 0,
                "episode_id": episode_id,
                "message": f"Clip generation failed: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Failed to get clips: {e}")
        return {
            "ok": False,
            "clips": [],
            "count": 0,
            "episode_id": episode_id,
            "error": str(e),
            "message": f"Clips fetch failed: {str(e)}"
        }

@app.post("/api/force-candidates")
async def force_generate_candidates():
    """Force generate candidates for the most recent episode (for debugging)"""
    try:
        # Get all episodes directly from the service
        episodes = list(episode_service.episodes.values())
        if not episodes:
            return {"ok": False, "error": "No episodes found"}
        
        # Get the most recent episode
        latest_episode = episodes[-1]
        if not latest_episode.transcript:
            return {"ok": False, "error": "No transcript available for latest episode"}
        
        # Generate candidates
        candidates, _ = await clip_score_service.get_candidates(
            latest_episode.id, 
            platform="tiktok_reels"
        )
        
        return {
            "ok": True,
            "episode_id": latest_episode.id,
            "candidates": candidates,
            "total_candidates": len(candidates)
        }
    except Exception as e:
        logger.error(f"Failed to force generate candidates: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/api/render-variant")
async def render_variant(request: Dict):
    """Render a variant format of an existing clip"""
    try:
        clip_id = request.get("clip_id")
        variant = request.get("variant", "square")
        style = request.get("style", "bold")
        captions = request.get("captions", True)
        
        if not clip_id:
            return {"ok": False, "error": "clip_id is required"}
        
        result = await clip_service.render_variant(
            clip_id=clip_id,
            variant=variant,
            style=style,
            captions=captions
        )
        
        if result["success"]:
            return {"ok": True, **result}
        else:
            return {"ok": False, "error": result.get("error", "Variant rendering failed")}
            
    except Exception as e:
        logger.error(f"Variant rendering failed: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/progress")
async def get_progress():
    """Get processing progress for the latest episode (normalized to { ok, progress: { percent, stage, message } })"""
    try:
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {
                "ok": False,
                "error": "No episodes found",
                "progress": {"percent": 0, "stage": "unknown", "message": "No episodes yet"},
            }
        latest_episode = episodes[-1]
        from services.progress_service import progress_service
        result = progress_service.get_progress(latest_episode.id) or {}
        progress_data = result.get("progress") or {}
        # Normalize keys
        pct = progress_data.get("percent")
        if pct is None:
            p = progress_data.get("percentage")
            progress_data["percent"] = int(p) if p is not None else 0
        progress_data.setdefault("stage", getattr(latest_episode, "status", None) or "unknown")
        progress_data.setdefault(
            "message", "Processing..." if progress_data.get("percent", 0) < 100 else "Completed"
        )
        return {"ok": True, "progress": progress_data}
    except Exception as e:
        logger.error(f"Progress check failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "progress": {"percent": 0, "stage": "error", "message": "Progress unavailable"},
        }

@app.get("/api/progress/{episode_id}")
async def get_episode_progress(episode_id: str, request: Request):
    """Return the same normalized shape as /api/progress with ETag caching"""
    import hashlib
    from fastapi import Response
    
    try:
        from services.progress_service import progress_service
        result = progress_service.get_progress(episode_id) or {}
        progress_data = result.get("progress") or {}
        if "percent" not in progress_data:
            p = progress_data.get("percentage")
            progress_data["percent"] = int(p) if p is not None else 0
        progress_data.setdefault("stage", progress_data.get("stage") or "unknown")
        progress_data.setdefault(
            "message", "Processing..." if progress_data.get("percent", 0) < 100 else "Completed"
        )
        
        # Generate weak ETag from progress data (stable, cheap)
        etag_data = f"{episode_id}:{progress_data['stage']}:{progress_data['percent']}:{progress_data.get('updated_at', 'unknown')}"
        etag = f"W/{hashlib.sha1(etag_data.encode()).hexdigest()}"
        
        # Check If-None-Match header
        if_none_match = request.headers.get("if-none-match")
        if if_none_match == etag:
            logger.debug(f"PROGRESS_ETAG_HIT {episode_id} etag={etag}")
            return Response(status_code=304, headers={"ETag": etag})
        
        # Log progress request (for monitoring)
        logger.debug(f"PROGRESS_ETAG_MISS {episode_id} etag={etag} stage={progress_data['stage']} percent={progress_data['percent']}")
        
        # Return response with ETag header
        response_data = {"ok": True, "progress": progress_data}
        response = JSONResponse(response_data)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "private, max-age=0, must-revalidate"
        return response
        
    except Exception as e:
        logger.error(f"Episode progress failed for {episode_id}: {e}")
        return {
            "ok": False,
            "error": str(e),
            "progress": {"percent": 0, "stage": "error", "message": "Progress unavailable"},
        }

@app.get("/api/candidate/debug")
async def candidate_debug(
    file_id: str = Query(..., description="Episode ID to analyze"),
    start: float = Query(..., description="Start time in seconds"),
    end: float = Query(..., description="End time in seconds"),
    platform: str = Query("tiktok_reels", description="Platform for optimization")
):
    """
    Recompute features/score for a single [start,end] to inspect details.
    """
    try:
        from services.secret_sauce_pkg import compute_features, score_segment, explain_segment
        from config_loader import get_config
        
        # Get episode
        episode = await episode_service.get_episode(file_id)
        if not episode or not episode.transcript:
            return {"ok": False, "error": "Episode not found or no transcript available"}
        
        # Create segment
        seg = {"start": start, "end": end, "text": ""}
        
        # Extract text between timestamps
        for segment in episode.transcript:
            if segment.start <= start and segment.end >= end:
                # Find overlapping text
                text_parts = []
                for seg_part in episode.transcript:
                    if seg_part.start < end and seg_part.end > start:
                        text_parts.append(seg_part.text)
                seg["text"] = " ".join(text_parts)
                break
        
        if not seg["text"]:
            return {"ok": False, "error": "No text found for the specified time range"}
        
        # Compute features and score
        feats = compute_features(seg, episode.audio_path)
        w = get_config()["weights"]
        raw = score_segment(feats, w)
        
        # Get explanation
        explanation = explain_segment(feats, w)
        
        return {
            "ok": True,
            "features": feats,
            "weights": w,
            "raw": raw,
            "explanation": explanation,
            "text": seg["text"]
        }
        
    except Exception as e:
        logger.error(f"Candidate debug failed: {e}")
        return {"ok": False, "error": str(e)}

# Weight sandbox for testing different scoring configurations
class WeightSandbox(BaseModel):
    file_id: str
    start: float
    end: float
    platform: str = "tiktok_reels"
    weights: dict  # partial or full override

@app.post("/api/sandbox/score")
async def sandbox_score(body: WeightSandbox):
    """Test different weight configurations on a specific clip segment"""
    try:
        from services.secret_sauce_pkg import compute_features, score_segment
        
        # Get episode
        episode = await episode_service.get_episode(body.file_id)
        if not episode or not episode.transcript:
            return {"ok": False, "error": "Episode not found or no transcript available"}
        
        # Create segment
        seg = {"start": body.start, "end": body.end, "text": ""}
        
        # Extract text between timestamps
        for segment in episode.transcript:
            if segment.start <= body.start and segment.end >= body.end:
                # Find overlapping text
                text_parts = []
                for seg_part in episode.transcript:
                    if seg_part.start < body.end and seg_part.end > body.start:
                        text_parts.append(seg_part.text)
                seg["text"] = " ".join(text_parts)
                break
        
        if not seg["text"]:
            return {"ok": False, "error": "No text found for the specified time range"}
        
        # Compute features
        feats = compute_features(seg, episode.audio_path)
        
        # Apply custom weights
        w = get_config()["weights"].copy()
        w.update(body.weights or {})
        
        # Score with custom weights
        raw = score_segment(feats, w)
        
        return {
            "ok": True, 
            "raw": raw, 
            "features": feats, 
            "weights_used": w,
            "text": seg["text"]
        }
        
    except Exception as e:
        logger.error(f"Weight sandbox failed: {e}")
        return {"ok": False, "error": str(e)}

# Feedback endpoint for user corrections
class FeedbackData(BaseModel):
    timestamp: str
    episode_id: str
    clip_start: float
    clip_end: float
    clip_score: float
    features: dict
    synergy_mult: float
    reason: str
    user_rating: str

@app.post("/api/feedback/flag-wrong")
async def flag_wrong_clip(feedback: FeedbackData):
    """Log user feedback when a clip is flagged as wrong"""
    try:
        import json
        import os
        from datetime import datetime
        
        # Create metrics directory if it doesn't exist
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Load existing feedback or create new
        feedback_file = os.path.join(metrics_dir, "feedback.json")
        existing_feedback = []
        
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    existing_feedback = json.load(f)
            except json.JSONDecodeError:
                existing_feedback = []
        
        # Add new feedback
        feedback_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": feedback.timestamp,
            "episode_id": feedback.episode_id,
            "clip_start": feedback.clip_start,
            "clip_end": feedback.clip_end,
            "clip_score": feedback.clip_score,
            "features": feedback.features,
            "synergy_mult": feedback.synergy_mult,
            "reason": feedback.reason,
            "user_rating": feedback.user_rating,
            "logged_at": datetime.now().isoformat()
        }
        
        existing_feedback.append(feedback_entry)
        
        # Save back to file
        with open(feedback_file, 'w') as f:
            json.dump(existing_feedback, f, indent=2)
        
        logger.info(f"Feedback logged: {feedback.reason} for clip {feedback.clip_start}-{feedback.clip_end}")
        
        return {
            "ok": True,
            "message": "Feedback logged successfully",
            "feedback_id": feedback_entry["id"]
        }
        
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        return {"ok": False, "error": str(e)}

# Debug Router for Quality Gate Inspection
from fastapi import APIRouter
debug_router = APIRouter(prefix="/api/debug")

@debug_router.post("/inspect-clip")
async def inspect_clip(payload: dict):
    """Inspect why a clip would pass/fail quality gates"""
    try:
        from services.secret_sauce_pkg import _hook_score, _ad_penalty, _payoff_presence
        
        # Extract payload data
        text = payload.get("text", "")
        start = float(payload.get("start", 0))
        end = float(payload.get("end", start + 20))
        arousal = float(payload.get("arousal", 0.3))
        qscore = float(payload.get("question_score", 0.0))
        
        # Compute hook score with original detection
        hook_val = _hook_score(text)
        
        # Detect ad content
        ad_result = _ad_penalty(text)
        
        # Compute payoff score
        payoff_score = _payoff_presence(text)
        
        # Check quality gates with new conditional logic
        payoff = payoff_score
        arousal = arousal
        
        # Conditional hook threshold based on payoff and arousal
        if payoff >= 0.5 and arousal >= 0.40:
            hook_threshold = 0.08  # τ_cond for high-quality clips
        else:
            hook_threshold = 0.12  # τ for standard clips
        
        weak_hook = hook_val < hook_threshold
        has_early_question = qscore >= 0.50
        no_payoff = payoff < 0.25
        ad_like = ad_result["flag"] and (weak_hook or no_payoff)
        
        # Determine gate reason with new soft penalty logic
        gate_reason = None
        if ad_like:
            if weak_hook and no_payoff:
                gate_reason = "ad_like;weak_hook;no_payoff"
            elif weak_hook:
                gate_reason = "ad_like;weak_hook"
            else:
                gate_reason = "ad_like;no_payoff"
        elif weak_hook and not has_early_question:
            gate_reason = "weak_hook_soft"  # Will get penalty, not hard fail
        elif no_payoff and not has_early_question:
            gate_reason = "no_payoff"
        
        return {
            "hook_score": round(hook_val, 3),
            "payoff_score": round(payoff_score, 3),
            "ad_flag": ad_result["flag"],
            "ad_penalty": round(ad_result["penalty"], 3),
            "ad_reason": ad_result["reason"],
            "question_score": round(qscore, 3),
            "weak_hook_gate": weak_hook,
            "no_payoff_gate": no_payoff,
            "ad_like_gate": ad_like,
            "gate_reason": gate_reason,
            "would_pass": gate_reason is None,
            "note": "question_score>=0.5 bypasses weak_hook and no_payoff gates"
        }
        
    except Exception as e:
        logger.error(f"Clip inspection failed: {e}")
        return {"ok": False, "error": str(e)}

# Include debug router
app.include_router(debug_router)

# Include clips router
from routes.clips import router as clips_router
app.include_router(clips_router)

# Include titles router
from routes_titles import router as titles_router
app.include_router(titles_router)

# Authentication Endpoints
@app.post("/api/auth/signup")
async def signup(user_data: UserSignup):
    """User signup endpoint"""
    try:
        result = await auth_service.signup_user(user_data)
        return result
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    """User login endpoint"""
    try:
        result = await auth_service.login_user(credentials)
        return result
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/auth/profile")
async def get_profile(user_id: str):
    """Get user profile and membership"""
    try:
        profile = await auth_service.get_user_profile(user_id)
        membership = await auth_service.get_user_membership(user_id)
        return {
            "success": True,
            "profile": profile,
            "membership": membership
        }
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        return {"success": False, "error": str(e)}

# Paddle Payment Endpoints
@app.get("/api/paddle/plans")
async def get_plans():
    """Get available subscription plans"""
    try:
        plans = paddle_service.get_available_plans()
        return {"success": True, "plans": plans}
    except Exception as e:
        logger.error(f"Failed to get plans: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/paddle/checkout")
async def create_checkout(checkout_data: PaddleCheckout):
    """Generate Paddle checkout URL"""
    try:
        checkout_url = paddle_service.get_checkout_url(
            checkout_data.product_id, 
            checkout_data.user_email
        )
        return {
            "success": True,
            "checkout_url": checkout_url,
            "plan_type": checkout_data.plan_type
        }
    except Exception as e:
        logger.error(f"Failed to create checkout: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/paddle/webhook")
async def paddle_webhook(request: Request):
    """Handle Paddle webhook events"""
    try:
        # Get webhook data
        webhook_data = await request.json()
        
        # Process webhook
        result = await paddle_service.process_webhook(webhook_data)
        
        return result
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return {"status": "error", "error": str(e)}

# Usage Tracking Endpoints
@app.get("/api/usage/summary/{user_id}")
async def get_usage_summary(user_id: str):
    """Get user usage summary"""
    try:
        # Get current usage vs limits
        membership = await auth_service.get_user_membership(user_id)
        if not membership:
            return {"success": False, "error": "No active membership"}
        
        # Check usage for current month
        usage_check = await auth_service.check_usage_limits(user_id, "upload")
        
        return {
            "success": True,
            "usage": usage_check
        }
    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        return {"success": False, "error": str(e)}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PodPromo AI API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload",
            "candidates": "/api/candidates",
            "render": "/api/render-one",
            "ab_tests": "/api/create-ab-tests",
            "history": "/api/history",
            "variant": "/api/render-variant",
            "config": "/config/get",
            "debug": "/api/candidate/debug",
            "sandbox": "/api/sandbox/score",
            "feedback": "/api/feedback/flag-wrong",
            "inspect": "/api/debug/inspect-clip",
            "auth": {
                "signup": "/api/auth/signup",
                "login": "/api/auth/login",
                "profile": "/api/auth/profile"
            },
            "paddle": {
                "plans": "/api/paddle/plans",
                "checkout": "/api/paddle/checkout",
                "webhook": "/api/paddle/webhook"
            },
            "usage": "/api/usage/summary/{user_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
