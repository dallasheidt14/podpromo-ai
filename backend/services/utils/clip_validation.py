"""
Clip time validation and clamping utilities.
Prevents invalid clip boundaries and ensures safe audio processing.
"""

import logging

logger = logging.getLogger(__name__)

def clamp_span(start, end, episode_dur: float) -> tuple[float, float]:
    """
    Clamp clip boundaries to valid ranges and enforce minimum duration.
    
    Args:
        start: Clip start time (seconds)
        end: Clip end time (seconds)  
        episode_dur: Total episode duration (seconds)
        
    Returns:
        Tuple of (clamped_start, clamped_end) in seconds
    """
    # Convert to float and handle None values
    s = max(0.0, float(start or 0.0))
    e = min(float(end or s), episode_dur)
    
    # Ensure end is not before start
    if e <= s:
        e = min(episode_dur, s + 0.5)
    
    # Enforce minimum duration of 0.5 seconds
    if e - s < 0.5:
        e = min(episode_dur, s + 0.5)
    
    # Round to 3 decimal places for consistency
    return round(s, 3), round(e, 3)

def validate_clip_times(clip: dict, episode_duration_s: float) -> dict:
    """
    Validate and clamp clip timing data.
    
    Args:
        clip: Clip dictionary with start/end/duration fields
        episode_duration_s: Total episode duration
        
    Returns:
        Updated clip dictionary with clamped times
    """
    start = clip.get("start", 0.0)
    end = clip.get("end", start + 10.0)  # Default 10s if no end
    
    # Clamp the times
    clamped_start, clamped_end = clamp_span(start, end, episode_duration_s)
    
    # Update the clip
    clip["start"] = clamped_start
    clip["end"] = clamped_end
    clip["duration"] = round(clamped_end - clamped_start, 3)
    
    # Log if we had to make adjustments
    if abs(start - clamped_start) > 0.001 or abs(end - clamped_end) > 0.001:
        logger.debug(f"CLIP_CLAMP: {clip.get('id', 'unknown')} start={start:.3f}→{clamped_start:.3f}, end={end:.3f}→{clamped_end:.3f}")
    
    return clip

def validate_all_clips(clips: list, episode_duration_s: float) -> list:
    """
    Validate and clamp timing for all clips in a list.
    
    Args:
        clips: List of clip dictionaries
        episode_duration_s: Total episode duration
        
    Returns:
        List of clips with validated timing
    """
    validated_clips = []
    
    for clip in clips:
        try:
            validated_clip = validate_clip_times(clip.copy(), episode_duration_s)
            validated_clips.append(validated_clip)
        except Exception as e:
            logger.warning(f"CLIP_VALIDATION_ERROR: Failed to validate clip {clip.get('id', 'unknown')}: {e}")
            # Skip invalid clips rather than crashing
            continue
    
    return validated_clips
