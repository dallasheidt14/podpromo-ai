from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
import logging

# Import your services
from services.clip_score import ClipScoreService
from services.clip_service import ClipService
from services.episode_service import EpisodeService
from models import (
    Episode, Clip, ClipGenerationRequest, 
    ClipGenerationResponse, MomentScore, TranscriptSegment, UploadResponse, HealthCheck, ApiError
)
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PodPromo AI API",
    description="AI-powered podcast clip generation service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
episode_service = EpisodeService()
clip_score_service = ClipScoreService(episode_service)
clip_service = ClipService(episode_service, clip_score_service)

# Mount static files for serving generated clips
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/clips", StaticFiles(directory=settings.OUTPUT_DIR), name="clips")

# Mount static files for general file access (including config)
app.mount("/files", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="files")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PodPromo AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if required services are available
        ffmpeg_available = clip_service.check_ffmpeg()
        whisper_available = clip_service.check_whisper()
        storage_available = episode_service.check_storage()
        
        return HealthCheck(
            status="healthy" if all([ffmpeg_available, whisper_available, storage_available]) else "unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services={
                "ffmpeg": ffmpeg_available,
                "whisper": whisper_available,
                "storage": storage_available
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services={"ffmpeg": False, "whisper": False, "storage": False}
        )

@app.post("/api/upload")
async def upload_episode(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a podcast episode file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Check file type
        allowed_types = ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Generate unique episode ID
        episode_id = str(uuid.uuid4())
        
        # Save file and create episode record
        episode = await episode_service.create_episode(
            episode_id=episode_id,
            file=file,
            original_filename=file.filename
        )
        
        # Process episode in background
        background_tasks.add_task(
            episode_service.process_episode,
            episode_id
        )
        
        return {
            "ok": True,
            "episodeId": episode_id,
            "filename": episode.filename,
            "size": episode.size,
            "status": episode.status,
            "message": "Episode uploaded successfully and processing started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/api/candidates")
async def get_candidates():
    """Get AI-scored clip candidates for the latest uploaded episode"""
    try:
        # For MVP, get the most recent episode
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": False, "error": "No episodes found. Upload a file first."}
        
        latest_episode = episodes[-1]  # Most recent
        if latest_episode.status != "completed":
            return {"ok": False, "error": "Episode still processing. Please wait."}
        
        # Get candidates using ClipScore service
        candidates = await clip_score_service.get_candidates(latest_episode.id)
        
        return {
            "ok": True, 
            "candidates": candidates,
            "episode_id": latest_episode.id
        }
        
    except Exception as e:
        logger.error(f"Failed to get candidates: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/progress")
async def get_progress():
    """Get current processing progress for all episodes"""
    try:
        from datetime import datetime
        
        # Get all episodes and their status
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": True, "progress": 0, "status": "No episodes"}
        
        # Find the most recent episode that's processing
        processing_episodes = [ep for ep in episodes if ep.status == "processing"]
        if not processing_episodes:
            return {"ok": True, "progress": 100, "status": "No processing episodes"}
        
        # Get the most recent processing episode
        latest = max(processing_episodes, key=lambda ep: ep.uploaded_at or datetime.min)
        
        # If it has a transcript, it's done
        if latest.transcript:
            return {"ok": True, "progress": 100, "status": "Transcription complete"}
        
        # Estimate progress based on time elapsed (rough approximation)
        if latest.uploaded_at:
            elapsed = (datetime.utcnow() - latest.uploaded_at).total_seconds()
            # Assume average transcription takes 2 minutes for a typical podcast
            estimated_total = 120  # seconds
            progress = min(95, int((elapsed / estimated_total) * 100))
            return {"ok": True, "progress": progress, "status": "Transcribing..."}
        
        return {"ok": True, "progress": 0, "status": "Starting..."}
        
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/episodes/{episode_id}", response_model=Episode)
async def get_episode(episode_id: str):
    """Get episode details"""
    try:
        episode = await episode_service.get_episode(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        return episode
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get episode {episode_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get episode")

@app.post("/api/generate-clips", response_model=ClipGenerationResponse)
async def generate_clips(
    request: ClipGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate clips from an episode"""
    try:
        # Validate episode exists and is processed
        episode = await episode_service.get_episode(request.episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        
        if episode.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail="Episode must be fully processed before generating clips"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Start clip generation in background
        background_tasks.add_task(
            clip_service.generate_clips,
            job_id,
            request.episode_id,
            request.target_count or 3,
            request.min_duration or 12,
            request.max_duration or 30
        )
        
        return ClipGenerationResponse(
            jobId=job_id,
            status="queued",
            clips=[],
            estimatedTime=300  # 5 minutes target
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clip generation failed: {e}")
        raise HTTPException(status_code=500, detail="Clip generation failed")

@app.get("/api/clips/{clip_id}", response_model=Clip)
async def get_clip(clip_id: str):
    """Get clip details and download URL"""
    try:
        clip = await clip_service.get_clip(clip_id)
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")
        return clip
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clip {clip_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get clip")

@app.get("/api/clips/{clip_id}/download")
async def download_clip(clip_id: str):
    """Download a generated clip"""
    try:
        clip = await clip_service.get_clip(clip_id)
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")
        
        if clip.status != "completed":
            raise HTTPException(status_code=400, detail="Clip not ready for download")
        
        if not clip.output_path or not os.path.exists(clip.output_path):
            raise HTTPException(status_code=404, detail="Clip file not found")
        
        return FileResponse(
            clip.output_path,
            media_type="video/mp4",
            filename=f"clip_{clip_id}.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for clip {clip_id}: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

@app.get("/api/episodes/{episode_id}/clips", response_model=List[Clip])
async def get_episode_clips(episode_id: str):
    """Get all clips for an episode"""
    try:
        clips = await clip_service.get_episode_clips(episode_id)
        return clips
    except Exception as e:
        logger.error(f"Failed to get clips for episode {episode_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get episode clips")

@app.post("/config/reload")
async def reload_config():
    """Reload the secret sauce configuration"""
    try:
        from .config_loader import load_config
        cfg = load_config()
        return {"ok": True, "weights": cfg["weights"], "lexicons": list(cfg["lexicons"].keys())}
    except Exception as e:
        logger.error(f"Config reload failed: {e}")
        raise HTTPException(status_code=500, detail="Config reload failed")

@app.post("/config/load-preset")
async def load_preset(name: str):
    """Load a genre preset configuration"""
    try:
        import json
        import os
        from .config_loader import load_config, BASE_DIR
        
        preset_path = os.path.join(BASE_DIR, "config", "presets", f"{name}.json")
        if not os.path.exists(preset_path):
            return {"ok": False, "error": "Preset not found"}
        
        # Read preset
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        
        # Overwrite main config with preset
        main_config_path = os.path.join(BASE_DIR, "config", "secret_config.json")
        with open(main_config_path, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2)
        
        # Reload configuration
        from .config_loader import load_config as _reload
        _reload()
        
        cfg = load_config()
        return {"ok": True, "active": name, "weights": cfg["weights"]}
        
    except Exception as e:
        logger.error(f"Preset loading failed: {e}")
        raise HTTPException(status_code=500, detail="Preset loading failed")

@app.get("/config/get")
async def config_get():
    """Get current configuration"""
    try:
        from config_loader import get_config
        cfg = get_config()
        return {"ok": True, "config": cfg}
    except Exception as e:
        logger.error(f"Config get failed: {e}")
        raise HTTPException(status_code=500, detail="Config get failed")

@app.post("/config/set-weights")
async def set_weights(weights: dict):
    """Set custom weights for the scoring algorithm"""
    try:
        import json
        import os
        from config_loader import BASE_DIR
        
        # Validate weights
        required_keys = ["hook", "prosody", "emotion", "q_or_list", "payoff", "info", "loop"]
        if not all(key in weights for key in required_keys):
            return {"ok": False, "error": "Missing required weight keys"}
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total == 0:
            return {"ok": False, "error": "Weights cannot all be zero"}
        
        normalized_weights = {k: v/total for k, v in weights.items()}
        
        # Load current config
        main_config_path = os.path.join(BASE_DIR, "config", "secret_config.json")
        current_config = {}
        if os.path.exists(main_config_path):
            with open(main_config_path, "r", encoding="utf-8") as f:
                current_config = json.load(f)
        
        # Update weights
        current_config["weights"] = normalized_weights
        
        # Save updated config
        with open(main_config_path, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=2)
        
        # Reload configuration
        from config_loader import load_config as _reload
        _reload()
        
        return {"ok": True, "weights": normalized_weights}
        
    except Exception as e:
        logger.error(f"Weight setting failed: {e}")
        raise HTTPException(status_code=500, detail="Weight setting failed")

@app.post("/metrics/log-choice")
async def log_choice(data: dict):
    """Log user choice for weight tuning analysis"""
    try:
        # For MVP, just log to console. In production, this would go to a database
        logger.info(f"User choice logged: {data}")
        
        # You could also save to a simple JSON file for analysis
        import json
        import os
        from datetime import datetime
        
        metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = os.path.join(metrics_dir, "user_choices.json")
        
        # Load existing metrics or create new
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = []
        
        # Add timestamp and append
        data["timestamp"] = datetime.utcnow().isoformat()
        metrics.append(data)
        
        # Keep only last 1000 entries for MVP
        if len(metrics) > 1000:
            metrics = metrics[-1000:]
        
        # Save updated metrics
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        return {"ok": True, "logged": True}
        
    except Exception as e:
        logger.error(f"Metrics logging failed: {e}")
        # Don't fail the request for metrics issues

@app.post("/api/render-one")
async def render_one_clip(start: float = Form(...), end: float = Form(...)):
    """Render a single clip from the selected time range"""
    try:
        # For MVP, get the most recent episode
        episodes = await episode_service.list_episodes()
        if not episodes:
            return {"ok": False, "error": "No episodes found. Upload a file first."}
        
        latest_episode = episodes[-1]
        if latest_episode.status != "completed":
            return {"ok": False, "error": "Episode still processing. Please wait."}
        
        # Generate clip using ClipService
        clip_id = str(uuid.uuid4())
        output_filename = f"clip_{clip_id}.mp4"
        
        # Start rendering in background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            clip_service.render_clip,
            clip_id,
            latest_episode.audio_path,
            start,
            end,
            output_filename
        )
        
        return {
            "ok": True,
            "clip_id": clip_id,
            "output": output_filename,
            "status": "rendering"
        }
        
    except Exception as e:
        logger.error(f"Render failed: {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
