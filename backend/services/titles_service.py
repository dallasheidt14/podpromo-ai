# backend/services/titles_service.py
# DEPRECATED shim to maintain imports

from .title_service import generate_titles, normalize_platform
__all__ = ["generate_titles", "normalize_platform"]

# Legacy TitlesService class for backward compatibility
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class TitlesService:
    """Service class for managing clip titles - provides backward compatibility"""
    
    def __init__(self):
        self.clips_cache = {}  # Simple in-memory cache
    
    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get clip data by ID - simplified implementation"""
        # This is a simplified implementation
        # In a real app, this would query the database
        logger.warning(f"TitlesService.get_clip({clip_id}) - using mock data")
        
        # Return mock clip data for testing
        return {
            "id": clip_id,
            "text": "Sample clip text for title generation",
            "transcript": "Sample clip text for title generation",
            "hook_score": 0.5,
            "arousal_score": 0.3,
            "score": 0.7
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