"""
Progress Writer - Monotonic progress with throttling and terminal freeze
"""

import logging
import time
from typing import Optional, Dict, Any
from services.progress_service import progress_service

logger = logging.getLogger(__name__)

# Stage progression order (monotonic)
STAGE_ORDER = [
    "queued", "uploading", "converting", "transcribing", 
    "scoring", "processing", "completed", "error"
]

# Terminal stages that freeze progress
TERMINAL_STAGES = {"completed", "error"}

# Throttle configuration
THROTTLE_DELTA_PERCENT = 5  # Minimum percent change to log
THROTTLE_DELTA_SEC = 2.0    # Minimum time between updates

class ProgressWriter:
    def __init__(self):
        self._last_updates: Dict[str, Dict[str, Any]] = {}
    
    def _get_stage_index(self, stage: str) -> int:
        """Get stage position in progression order"""
        try:
            return STAGE_ORDER.index(stage.lower().strip())
        except ValueError:
            return len(STAGE_ORDER) - 1  # Default to last stage
    
    def _is_stage_progression_valid(self, episode_id: str, new_stage: str) -> bool:
        """Check if stage progression is monotonic"""
        if episode_id not in self._last_updates:
            return True
        
        last_stage = self._last_updates[episode_id].get("stage", "queued")
        last_index = self._get_stage_index(last_stage)
        new_index = self._get_stage_index(new_stage)
        
        logger.debug(f"Stage progression check: {episode_id} {last_stage}({last_index}) -> {new_stage}({new_index}) = {new_index >= last_index}")
        return new_index >= last_index
    
    def _is_percent_increase_valid(self, episode_id: str, new_percent: int) -> bool:
        """Check if percent increase is valid (monotonic)"""
        if episode_id not in self._last_updates:
            return True
        
        last_percent = self._last_updates[episode_id].get("percent", 0)
        return new_percent >= last_percent
    
    def _should_throttle(self, episode_id: str, new_percent: int) -> bool:
        """Check if update should be throttled"""
        if episode_id not in self._last_updates:
            return False
        
        last_update = self._last_updates[episode_id]
        last_percent = last_update.get("percent", 0)
        last_time = last_update.get("timestamp", 0)
        
        # Check percent delta
        percent_delta = abs(new_percent - last_percent)
        if percent_delta < THROTTLE_DELTA_PERCENT:
            return True
        
        # Check time delta
        time_delta = time.time() - last_time
        if time_delta < THROTTLE_DELTA_SEC:
            return True
        
        return False
    
    def _is_terminal_frozen(self, episode_id: str) -> bool:
        """Check if episode is in terminal state and frozen"""
        if episode_id not in self._last_updates:
            return False
        
        last_stage = self._last_updates[episode_id].get("stage", "")
        return last_stage in TERMINAL_STAGES
    
    def write_progress(
        self,
        episode_id: str,
        stage: str,
        percent: Optional[int] = None,
        message: Optional[str] = None
    ) -> None:
        """Write progress with monotonic enforcement, throttling, and terminal freeze"""
        try:
            # Check if terminal frozen
            if self._is_terminal_frozen(episode_id):
                logger.debug(f"Progress frozen for terminal episode {episode_id}")
                return
            
            # Normalize stage
            normalized_stage = stage.lower().strip()
            if normalized_stage not in STAGE_ORDER:
                normalized_stage = "processing"
            
            # Check stage progression
            if not self._is_stage_progression_valid(episode_id, normalized_stage):
                logger.debug(f"Stage regression ignored for {episode_id}: {stage}")
                return
            
            # Prepare progress data
            progress_data = {
                "stage": normalized_stage,
                "message": message or f"Processing {normalized_stage}...",
            }
            # Handle percent with monotonic enforcement
            if percent is not None:
                if not self._is_percent_increase_valid(episode_id, percent):
                    logger.debug(f"Percent regression ignored for {episode_id}: {percent}%")
                    return
                
            # ✨ KEY FIX: if the stage is changing (e.g., scoring→completed), don't throttle
            if normalized_stage != self._last_updates.get(episode_id, {}).get("stage", ""):
                # Stage change - always allow, don't throttle
                if percent is not None:
                    progress_data["percent"] = min(100, max(0, percent))
            else:
                # Same stage -> apply throttling
                if self._should_throttle(episode_id, percent):
                    logger.debug(f"Progress throttled for {episode_id}: {percent}%")
                    return
                if percent is not None:
                    progress_data["percent"] = min(100, max(0, percent))
            # Write to progress service
            progress_service._save_progress_atomic(episode_id, progress_data)
            
            # Update internal state
            self._last_updates[episode_id] = {
                "stage": normalized_stage,
                "percent": progress_data.get("percent", 0),
                "timestamp": time.time()
            }
            
            # Extra breadcrumb for debugging stage transitions
            try:
                prev_stage = self._last_updates.get(episode_id, {}).get("stage", "")
                if prev_stage != normalized_stage:
                    logger.info("Progress stage change: %s %s%% -> %s %s%%", episode_id, self._last_updates.get(episode_id, {}).get("percent", ""), normalized_stage, progress_data.get('percent', ''))
            except Exception:
                pass
            
            logger.info("Progress updated: %s -> %s %s%%", episode_id, normalized_stage, progress_data.get('percent', ''))
            
        except Exception as e:
            logger.error(f"Failed to write progress for {episode_id}: {e}")

# Global instance
_writer = ProgressWriter()

def write_progress(
    episode_id: str,
    stage: str,
    percent: Optional[int] = None,
    message: Optional[str] = None
) -> None:
    """Write progress with monotonic enforcement, throttling, and terminal freeze"""
    _writer.write_progress(episode_id, stage, percent, message)

def write_stage(episode_id: str, stage: str, message: Optional[str] = None) -> None:
    """Write stage without changing percent"""
    write_progress(episode_id, stage, None, message)

def write_percent(episode_id: str, percent: int, message: Optional[str] = None) -> None:
    """Write percent without changing stage"""
    write_progress(episode_id, "processing", percent, message)

def get_progress(episode_id: str) -> Optional[Dict[str, Any]]:
    """Get current progress for episode"""
    return progress_service.get_progress(episode_id)