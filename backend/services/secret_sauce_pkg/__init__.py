"""Secret sauce package providing feature extraction and scoring utilities."""

# Import genre-related classes
from .genres import (
    GenreProfile,
    GenreAwareScorer,
)

from .genre_profiles import (
    FantasySportsGenreProfile,
    ComedyGenreProfile,
)

# Import main functions from the original secret_sauce.py
# This is a temporary solution to fix import errors
# We'll gradually move these to their proper modules

# Import all the main functions that are being used
# We'll import them individually to handle missing functions gracefully
try:
    from services.secret_sauce import (
        # Main scoring functions
        compute_features_v4,
        score_segment_v4,
        explain_segment_v4,
        viral_potential_v4,
        get_clip_weights,
        
        # Legacy functions for backward compatibility
        compute_features,
        score_segment,
        explain_segment,
        viral_potential,
        
        # Utility functions
        _hook_score,
        _payoff_presence,
        _info_density,
        _ad_penalty,
        _audio_prosody_score,
        _emotion_score,
        _question_or_list,
        _loopability_heuristic,
        
        # V4 specific functions
        _hook_score_v4,
        _hook_score_v5,
        _emotion_score_v4,
        _info_density_v4,
        _detect_payoff,
        _detect_insight_content,
        _detect_insight_content_v2,
        _calculate_niche_penalty,
        _arousal_score_text,
        _platform_length_match,
        calculate_dynamic_length_score,
        
        # Other utility functions that exist
        find_viral_clips,
        resolve_platform,
        PLATFORM_GENRE_MULTIPLIERS,
        compute_audio_hook_modifier,
        _apply_insight_confidence_multiplier,
        compute_audio_energy,
        _heuristic_title,
        _grade_breakdown,
        detect_podcast_genre,
    )
except ImportError as e:
    print(f"Warning: Some functions could not be imported from secret_sauce: {e}")

# Handle functions that don't exist by creating placeholder functions
def _calibrate_hook_v5(*args, **kwargs):
    """Placeholder for missing function"""
    raise NotImplementedError("_calibrate_hook_v5 function not found in secret_sauce.py")

def _sigmoid01(*args, **kwargs):
    """Placeholder for missing function"""
    raise NotImplementedError("_sigmoid01 function not found in secret_sauce.py")

def attach_hook_scores(*args, **kwargs):
    """Placeholder for missing function"""
    raise NotImplementedError("attach_hook_scores function not found in secret_sauce.py")

__all__ = [
    # Genre classes
    "GenreProfile",
    "FantasySportsGenreProfile", 
    "ComedyGenreProfile",
    "GenreAwareScorer",
    
    # Main scoring functions
    "compute_features_v4",
    "score_segment_v4",
    "explain_segment_v4",
    "viral_potential_v4",
    "get_clip_weights",
    
    # Legacy functions
    "compute_features",
    "score_segment",
    "explain_segment",
    "viral_potential",
    
    # Utility functions
    "_hook_score",
    "_payoff_presence",
    "_info_density",
    "_ad_penalty",
    "_audio_prosody_score",
    "_emotion_score",
    "_question_or_list",
    "_loopability_heuristic",
    
    # V4 specific functions
    "_hook_score_v4",
    "_hook_score_v5",
    "_emotion_score_v4",
    "_info_density_v4",
    "_detect_payoff",
    "_detect_insight_content",
    "_detect_insight_content_v2",
    "_calculate_niche_penalty",
    "_arousal_score_text",
    "_platform_length_match",
    "calculate_dynamic_length_score",
    
    # Other utility functions
    "find_viral_clips",
    "resolve_platform",
    "PLATFORM_GENRE_MULTIPLIERS",
    "compute_audio_hook_modifier",
    "_calibrate_hook_v5",
    "_sigmoid01",
    "attach_hook_scores",
    "_apply_insight_confidence_multiplier",
    "compute_audio_energy",
    "_heuristic_title",
    "_grade_breakdown",
]
