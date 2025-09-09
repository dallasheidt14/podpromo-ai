"""
secret_sauce_pkg - V4 Enhanced Viral Detection System
Modular package for viral clip detection and scoring.
"""

# Import from the modular components
from .genres import GenreProfile, GenreAwareScorer
from .genre_profiles import FantasySportsGenreProfile, ComedyGenreProfile
from .scoring import (
    get_clip_weights,
    score_segment_v4,
    explain_segment_v4,
    viral_potential_v4,
    score_segment,
    viral_potential,
    explain_segment,
    viral_potential_from_segment,
    explain_segment_from_segment
)
from .features import (
    # Main feature computation functions
    compute_features_v4,
    compute_features_v4_batch,
    compute_features,
    compute_features_cached,
    
    # Individual feature functions
    _hook_score,
    _hook_score_v4,
    _hook_score_v5,
    _emotion_score,
    _emotion_score_v4,
    _payoff_presence,
    _payoff_presence_v4,
    _detect_payoff,
    _info_density,
    _info_density_v4,
    _question_or_list,
    _loopability_heuristic,
    _arousal_score_text,
    _audio_prosody_score,
    _detect_insight_content,
    _detect_insight_content_v2,
    _calculate_niche_penalty,
    _ad_penalty,
    _platform_length_match,
    calculate_dynamic_length_score,
    
    # Utility functions
    create_segment_hash,
    debug_segment_scoring,
)

# Import remaining functions from monolithic file temporarily
from .__init__monolithic import (
    _grade_breakdown,
    _heuristic_title,
    _sigmoid,
    resolve_platform,
)

# Export all public functions and classes
__all__ = [
    # Genre system
    "GenreProfile",
    "GenreAwareScorer", 
    "FantasySportsGenreProfile",
    "ComedyGenreProfile",
    
    # Scoring functions
    "get_clip_weights",
    "score_segment_v4",
    "explain_segment_v4", 
    "viral_potential_v4",
    "score_segment",
    "viral_potential",
    "explain_segment",
    "viral_potential_from_segment",
    "explain_segment_from_segment",
    
    # Feature computation functions
    "compute_features_v4",
    "compute_features_v4_batch",
    "compute_features", 
    "compute_features_cached",
    
    # Individual feature functions
    "_hook_score",
    "_hook_score_v4",
    "_hook_score_v5",
    "_emotion_score",
    "_emotion_score_v4",
    "_payoff_presence",
    "_payoff_presence_v4",
    "_detect_payoff",
    "_info_density",
    "_info_density_v4",
    "_question_or_list",
    "_loopability_heuristic",
    "_arousal_score_text",
    "_audio_prosody_score",
    "_detect_insight_content",
    "_detect_insight_content_v2",
    "_calculate_niche_penalty",
    "_ad_penalty",
    "_platform_length_match",
    "calculate_dynamic_length_score",
    
    # Utility functions
    "create_segment_hash",
    "debug_segment_scoring",
    
    # Additional functions from monolithic file
    "_grade_breakdown",
    "_heuristic_title",
    "_sigmoid",
    "resolve_platform",
]
