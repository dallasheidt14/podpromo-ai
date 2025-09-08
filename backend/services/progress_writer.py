"""
Progress Writer - Helper for writing progress with monotonic percent and normalized stages
"""

import logging
from typing import Optional
from services.progress_service import progress_service

logger = logging.getLogger(__name__)

def clamp_percent(episode_id: str, new_percent: int) -> int:
    """Ensure percent never goes backwards (monotonic)"""
    try:
        # Load current progress to get last percent
        current = progress_service._load_progress(episode_id)
        if current:
            old_percent = int(current.get("percent", current.get("percentage", 0)))
            return max(old_percent, min(100, int(new_percent)))
        return max(0, min(100, int(new_percent)))
    except Exception as e:
        logger.warning(f"Failed to clamp percent for {episode_id}: {e}")
        return max(0, min(100, int(new_percent)))

def write_progress(
    episode_id: str, 
    stage: str, 
    percent: Optional[int] = None, 
    message: Optional[str] = None
) -> None:
    """Write progress with monotonic percent and normalized stage"""
    try:
        # Normalize stage names to match frontend expectations
        stage_map = {
            "queue": "queued",
            "queued": "queued", 
            "waiting": "queued",
            "preparing": "queued",
            "upload": "uploading",
            "uploading": "uploading",
            "convert": "converting",
            "converting": "converting",
            "muxing": "converting",
            "transcribe": "transcribing",
            "transcribing": "transcribing",
            "asr": "transcribing",
            "scoring": "scoring",
            "scoring:prerank": "scoring",
            "scoring:full": "scoring", 
            "scoring:completed": "scoring",
            "analyze": "processing",
            "analysing": "processing",
            "processing": "processing",
            "process": "processing",
            "generate": "processing",
            "generating": "processing",
            "completed": "completed",
            "complete": "completed",
            "done": "completed",
            "success": "completed",
            "failed": "error",
            "fail": "error",
            "error": "error",
        }
        
        normalized_stage = stage_map.get(stage.lower().strip(), "processing")
        
        # Prepare progress data
        progress_data = {
            "stage": normalized_stage,
            "message": message or f"Processing {normalized_stage}...",
        }
        
        # Add percent if provided, ensuring monotonic behavior
        if percent is not None:
            progress_data["percent"] = clamp_percent(episode_id, percent)
        
        # Write to progress service
        progress_service._save_progress_atomic(episode_id, progress_data)
        
        logger.info(f"Progress updated: {episode_id} -> {normalized_stage} {percent}%")
        
    except Exception as e:
        logger.error(f"Failed to write progress for {episode_id}: {e}")

def write_stage(episode_id: str, stage: str, message: Optional[str] = None) -> None:
    """Write stage without changing percent"""
    write_progress(episode_id, stage, None, message)

def write_percent(episode_id: str, percent: int, message: Optional[str] = None) -> None:
    """Write percent without changing stage"""
    write_progress(episode_id, "processing", percent, message)
