"""
Progress Service - Atomic file-based progress tracking with restart safety
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from starlette import status

from config.settings import UPLOAD_DIR, TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)

class ProgressService:
    """Handles atomic progress persistence and retrieval"""
    
    def __init__(self):
        self.upload_dir = Path(UPLOAD_DIR)
        self.transcripts_dir = Path(TRANSCRIPTS_DIR)
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_progress_file_path(self, episode_id: str) -> Path:
        """Get the path for episode progress file"""
        episode_dir = self.upload_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir / "progress.json"
    
    def _save_progress_atomic(self, episode_id: str, data: Dict[str, Any]) -> None:
        """Atomically save progress data to prevent corruption"""
        try:
            progress_file = self._get_progress_file_path(episode_id)
            temp_file = progress_file.with_suffix(".tmp")
            
            # Write to temp file first
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic replace
            os.replace(temp_file, progress_file)
            
        except Exception as e:
            logger.error(f"Failed to save progress for {episode_id}: {e}")
            # Clean up temp file if it exists
            temp_file = self._get_progress_file_path(episode_id).with_suffix(".tmp")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
    
    def _load_progress(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Load progress data from file"""
        try:
            progress_file = self._get_progress_file_path(episode_id)
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load progress for {episode_id}: {e}")
            return None
    
    def _infer_progress_from_disk(self, episode_id: str) -> Dict[str, Any]:
        """Infer progress from disk state when no progress file exists"""
        try:
            audio_file = self.upload_dir / f"{episode_id}.mp3"
            transcript_file = self.transcripts_dir / f"{episode_id}.json"
            
            # Check if transcript exists and is valid
            if transcript_file.exists() and transcript_file.stat().st_size > 10:
                return {
                    "ok": True,
                    "progress": {
                        "percent": 100,
                        "stage": "completed",
                        "message": "Processing completed",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    "status": "completed"
                }
            
            # Check if audio exists but no transcript yet
            if audio_file.exists() and audio_file.stat().st_size > 10:
                return {
                    "ok": True,
                    "progress": {
                        "percent": 10,
                        "stage": "queued",
                        "message": "Audio uploaded, waiting for processing",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    "status": "queued"
                }
            
            # No files found
            return {
                "ok": False,
                "progress": {
                    "percent": 0,
                    "stage": "unknown",
                    "message": "Episode not found",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "status": "unknown"
            }
            
        except Exception as e:
            logger.error(f"Failed to infer progress from disk for {episode_id}: {e}")
            return {
                "ok": False,
                "progress": {
                    "percent": 0,
                    "stage": "error",
                    "message": f"Progress inference failed: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "status": "error"
            }
    
    def get_progress(self, episode_id: str) -> dict:
        """
        Get progress for an episode - NEVER returns 500, always returns JSON
        """
        try:
            # First try to load from progress file
            progress_data = self._load_progress(episode_id)
            
            if progress_data:
                # Normalize percentage → percent for consistent contract
                # The progress data is directly at root level, not nested under "progress"
                progress = progress_data
                if "percentage" in progress and "percent" not in progress:
                    progress["percent"] = progress.pop("percentage")
                
                logger.info(f"Progress service returning: {progress}")
                return {
                    "ok": True,
                    "progress": progress,
                    "status": progress_data.get("status", "unknown")
                }
            
            # No progress file found - infer from disk state
            disk_progress = self._infer_progress_from_disk(episode_id)
            
            if disk_progress["ok"]:
                # Normalize percentage → percent for consistent contract
                progress = disk_progress["progress"]
                if "percentage" in progress and "percent" not in progress:
                    progress["percent"] = progress.pop("percentage")
                
                return {
                    "ok": True,
                    "progress": progress,
                    "status": disk_progress["status"]
                }
            else:
                # Episode not found - return 404 with JSON
                return {
                    "ok": False,
                    "error": "Episode not found",
                    "progress": {
                        "percent": 0,
                        "stage": "unknown",
                        "message": "Episode not found",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
                
        except Exception as e:
            logger.exception(f"Progress check failed for {episode_id}", extra={"episode_id": episode_id})
            
            # Return 200 with error status to prevent frontend retry loops
            return {
                "ok": True,
                "progress": {
                    "percent": 0,
                    "stage": "error",
                    "message": "Progress read failure - please retry",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "status": "error"
            }
    
    def update_progress(self, episode_id: str, stage: str, percentage: int, message: str = None, detail: str = None) -> None:
        """Update progress for an episode"""
        try:
            progress_data = {
                "percentage": max(0, min(100, percentage)),
                "stage": stage,
                "message": message or f"Processing {stage}...",
                "detail": detail or "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            self._save_progress_atomic(episode_id, progress_data)
            
        except Exception as e:
            logger.error(f"Failed to update progress for {episode_id}: {e}")
    
    def mark_completed(self, episode_id: str, message: str = "Processing completed") -> None:
        """Mark episode as completed"""
        self.update_progress(episode_id, "completed", 100, message)
    
    def mark_error(self, episode_id: str, error_message: str) -> None:
        """Mark episode as errored"""
        self.update_progress(episode_id, "error", 0, f"Error: {error_message}")

# Global instance
progress_service = ProgressService()
