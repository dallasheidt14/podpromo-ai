"""
Canonical clip ID generation utilities.
Ensures stable, collision-safe clip identifiers.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

def generate_clip_id(episode_id: str, start_s: float, end_s: float) -> str:
    """
    Generate a deterministic, collision-safe clip ID.
    
    The ID is based on episode_id + start_ms + end_ms, ensuring:
    - Deterministic: same inputs always produce same ID
    - Collision-safe: different clips have different IDs
    - Stable: text changes don't affect the ID
    
    Args:
        episode_id: Unique episode identifier
        start_s: Clip start time in seconds
        end_s: Clip end time in seconds
        
    Returns:
        16-character hexadecimal clip ID
    """
    # Convert to milliseconds for precision
    start_ms = int(round(start_s * 1000))
    end_ms = int(round(end_s * 1000))
    
    # Create deterministic string
    raw = f"{episode_id}:{start_ms}:{end_ms}"
    
    # Generate SHA1 hash and take first 16 characters
    clip_id = hashlib.sha1(raw.encode()).hexdigest()[:16]
    
    logger.debug(f"CLIP_ID: Generated {clip_id} for {episode_id} [{start_s:.3f}s-{end_s:.3f}s]")
    
    return clip_id

def assign_clip_ids(clips: list, episode_id: str) -> list:
    """
    Assign canonical IDs to all clips in a list.
    
    Args:
        clips: List of clip dictionaries
        episode_id: Episode identifier
        
    Returns:
        List of clips with assigned IDs
    """
    for clip in clips:
        if "id" not in clip or not clip["id"]:
            start = clip.get("start", 0.0)
            end = clip.get("end", start + 10.0)
            clip["id"] = generate_clip_id(episode_id, start, end)
    
    return clips

def validate_clip_id(clip_id: str) -> bool:
    """
    Validate that a clip ID has the expected format.
    
    Args:
        clip_id: Clip ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not clip_id:
        return False
    
    # Check length and character set
    if len(clip_id) != 16:
        return False
    
    # Should be hexadecimal
    try:
        int(clip_id, 16)
        return True
    except ValueError:
        return False
