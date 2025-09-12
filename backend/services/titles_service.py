# backend/services/titles_service.py
# DEPRECATED shim to maintain imports

from .title_service import generate_titles, normalize_platform
__all__ = ["generate_titles", "normalize_platform"]

# Legacy TitlesService class for backward compatibility
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import re
import pathlib
from config.settings import UPLOAD_DIR
import os

logger = logging.getLogger(__name__)

# Regex to extract episode ID from clip ID (supports both UUID and test formats)
CLIP_ID_RE = re.compile(r"^clip_([0-9a-f\-]{36}|[a-z0-9\-]+)_")

def _load_clip(clip_id: str, uploads_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load real clip data from clips.json file"""
    m = CLIP_ID_RE.match(clip_id)
    if not m:
        logger.warning("Invalid clip_id format: %s", clip_id)
        return None
    
    episode_id = m.group(1)
    clips_file = uploads_dir / episode_id / "clips.json"
    
    if not clips_file.exists():
        logger.warning("clips.json missing for episode %s", episode_id)
        return None
    
    try:
        data = json.loads(clips_file.read_text(encoding="utf-8"))
        clips = data.get("clips", [])
        
        # Find the specific clip
        for clip in clips:
            if clip.get("id") == clip_id:
                return clip
        
        logger.warning("Clip %s not found in clips.json", clip_id)
        return None
        
    except Exception as e:
        logger.error("Failed to load clips.json for episode %s: %s", episode_id, e)
        return None

class TitlesService:
    """Service class for managing clip titles - provides backward compatibility"""
    
    def __init__(self):
        self.clips_cache = {}  # Simple in-memory cache
    
    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get clip data by ID - loads real data from clips.json"""
        uploads_dir = pathlib.Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))
        
        # Try to load real clip data
        clip = _load_clip(clip_id, uploads_dir)
        if clip:
            logger.info(f"TitlesService.get_clip({clip_id}) - loaded real clip data")
            return clip
        
        # Fallback to minimal data if clip not found
        logger.warning(f"TitlesService.get_clip({clip_id}) - clip not found, using minimal fallback")
        return {
            "id": clip_id,
            "text": "",
            "transcript": "",
            "start": 0.0,
            "end": 0.0
        }
    
    def generate_variants(self, clip: Dict[str, Any], body: Any) -> Tuple[List[str], str, Dict[str, Any]]:
        """Generate title variants using legacy method"""
        text = clip.get("transcript") or clip.get("text") or ""
        platform = getattr(body, 'platform', 'default')
        
        # Use new generator as fallback
        titles = generate_titles(text, platform=platform, n=6)
        variants = [t["title"] for t in titles]
        chosen = variants[0] if variants else "Most Leaders Solve the Wrong Problem"
        
        meta = {
            "generator": "legacy_fallback",
            "version": 1,
            "generated_at": "2024-01-01T00:00:00Z",
        }
        
        return variants, chosen, meta
    
    def save_titles(self, clip_id: str, platform: str, variants: List[str], chosen: str, meta: Dict[str, Any]) -> bool:
        """Save generated titles - simplified implementation"""
        logger.info(f"TitlesService.save_titles({clip_id}, {platform}, {len(variants)} variants)")
        # In a real app, this would save to database
        return True
    
    def set_chosen_title(self, clip_id: str, platform: str, title: str) -> bool:
        """Set the chosen title for a clip"""
        logger.info(f"TitlesService.set_chosen_title({clip_id}, {platform}, {title})")
        # In a real app, this would update the database
        return True