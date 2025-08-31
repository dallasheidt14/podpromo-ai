"""
PodPromo AI - FastAPI Backend
Main application entry point with all API endpoints.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import (
    Episode, TranscriptSegment, MomentScore, UploadResponse, 
    HealthCheck, ApiError, RenderRequest
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/files", StaticFiles(directory="./uploads"), name="files")
app.mount("/clips", StaticFiles(directory="./outputs"), name="clips")

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

# Initialize services as None, will be created in startup event
file_manager = None
queue_manager = None
monitoring_service = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize production services on startup"""
    global file_manager, queue_manager, monitoring_service
    
    try:
        # Ensure directories exist
        os.makedirs("./uploads", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Initialize production services
        file_manager = FileManager()
        queue_manager = QueueManager()
        monitoring_service = MonitoringService()
        
        # File manager directories are created in __init__
        # Monitoring service starts automatically in __init__
        
        logger.info("Production services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize production services: {e}")

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
async def test_transcription(file_id: str = Form(...)):
    """Test transcription with an existing file without uploading"""
    try:
        # Find the file in uploads directory
        file_path = f"./uploads/{file_id}"
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

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check system health and service status"""
    try:
        # Check FFmpeg
        ffmpeg_ok = clip_service.check_ffmpeg()
        
        # Check Whisper (if model is loaded)
        whisper_ok = clip_service.check_whisper()
        
        # Check storage
        storage_ok = episode_service.check_storage()
        
        # Check outputs directory
        outputs_ok = os.path.exists("./outputs") or os.makedirs("./outputs", exist_ok=True)
        
        # Check production services (simplified for now)
        file_manager_ok = file_manager is not None
        queue_manager_ok = queue_manager is not None
        monitoring_ok = monitoring_service is not None
        
        status = "healthy" if all([ffmpeg_ok, whisper_ok, storage_ok, outputs_ok, 
                                  file_manager_ok, queue_manager_ok, monitoring_ok]) else "unhealthy"
        
        return HealthCheck(
            status=status,
            timestamp=datetime.now(),
            version="1.0.0",
            services={
                "ffmpeg": ffmpeg_ok,
                "whisper": whisper_ok,
                "storage": storage_ok,
                "file_manager": file_manager_ok,
                "queue_manager": queue_manager_ok,
                "monitoring": monitoring_ok
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            services={
                "ffmpeg": False, 
                "whisper": False, 
                "storage": False,
                "file_manager": False,
                "queue_manager": False,
                "monitoring": False
            }
        )

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
        from services.secret_sauce import _hook_score, _payoff_presence, _info_density, _ad_penalty
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
async def upload_episode(file: UploadFile = File(...)):
    """Upload and process a podcast episode"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file using FileManager (if available)
        if file_manager:
            validation_result = file_manager.validate_upload_file(file)
            if not validation_result["valid"]:
                raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Create episode and start processing
        episode_id = str(uuid.uuid4())
        episode = await episode_service.create_episode(episode_id, file, file.filename)
        
        # Add to processing queue (if available)
        job_id = None
        if queue_manager:
            job_id = queue_manager.add_job(
                job_type="episode_processing",
                priority="high",
                data={"episode_id": episode.id}
            )
        
        # Start background processing with queue management
        await episode_service.process_episode(episode.id)
        
        # Update queue status (if available)
        if queue_manager and job_id:
            queue_manager.update_job_status(job_id, "completed")
        
        # Log success to monitoring (if available)
        if monitoring_service:
            monitoring_service.record_metric("upload_success", 1)
        
        return {
            "ok": True, 
            "episodeId": episode.id, 
            "jobId": job_id,
            "message": "Upload successful and processing started"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        # Log error to monitoring service (if available)
        if monitoring_service:
            monitoring_service.record_error("upload_failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/candidates")
async def get_candidates(
    platform: str = Query("tiktok_reels", description="Platform for optimization (tiktok_reels, shorts, linkedin_sq)"),
    genre: str | None = Query(None, description="Genre for optimization (comedy, fantasy_sports, sports, true_crime, business, news_politics, education, health_wellness)"),
    debug: bool = Query(False, description="Enable debug mode for detailed candidate information")
):
    """Get AI-scored clip candidates with optional platform/tone optimization"""
    try:
        # Set debug environment variable
        os.environ["PODPROMO_DEBUG_ALL"] = "1" if debug else "0"
        
        # Get the most recent episode
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": False, "error": "No episodes found. Upload a file first."}
        
        latest_episode = episodes[-1]
        
        if latest_episode.status != "completed":
            return {"ok": False, "error": "Episode still processing. Please wait."}
        
        # Get candidates using ClipScore with platform/genre optimization
        candidates = await clip_score_service.get_candidates(
            latest_episode.id, 
            platform=platform, 
            genre=genre
        )
        
        if not candidates:
            return {"ok": False, "error": "No candidates found for this episode."}
        
        # Add platform mapping info to response
        from services.secret_sauce import resolve_platform
        backend_platform = resolve_platform(platform)
        
        # Record successful candidate generation (if monitoring available)
        if monitoring_service:
            monitoring_service.record_metric("candidates_generated", len(candidates))
            monitoring_service.record_metric("candidate_generation_success", 1)
        
        return {
            "ok": True, 
            "candidates": candidates,
            "optimization": {
                "frontend_platform": platform,
                "frontend_genre": genre,
                "backend_platform": backend_platform,
                "description": f"Optimized for {platform}" + (f" in {genre} genre" if genre else "")
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
    loop_seam: bool = Form(False)
):
    """Render a single clip with enhanced options"""
    try:
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
    """Get processing progress for the latest episode"""
    try:
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": False, "error": "No episodes found"}
        
        latest_episode = episodes[-1]
        
        if latest_episode.status == "completed":
            return {"ok": True, "progress": 100, "status": "completed"}
        elif latest_episode.status == "processing":
            # Estimate progress based on episode duration and processing time
            if hasattr(latest_episode, 'duration') and latest_episode.duration:
                # Rough estimate: assume transcription takes 2-3x real-time
                estimated_time = latest_episode.duration * 2.5
                elapsed = (datetime.now() - latest_episode.uploaded_at).total_seconds()
                progress = min(95, int((elapsed / estimated_time) * 100))
                return {"ok": True, "progress": progress, "status": "processing"}
            else:
                return {"ok": True, "progress": 50, "status": "processing"}
        else:
            return {"ok": True, "progress": 0, "status": latest_episode.status}
            
    except Exception as e:
        logger.error(f"Progress check failed: {e}")
        return {"ok": False, "error": str(e)}

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
        from services.secret_sauce import compute_features, score_segment, explain_segment
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
        from services.secret_sauce import compute_features, score_segment
        
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
        from services.secret_sauce import _hook_score, _ad_penalty, _payoff_presence
        
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
            "inspect": "/api/debug/inspect-clip"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
