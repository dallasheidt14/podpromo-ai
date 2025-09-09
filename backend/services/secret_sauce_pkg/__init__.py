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
    
    # Dynamic segmentation functions
    find_natural_boundaries,
    create_dynamic_segments,
    
    # Pipeline functions
    filter_ads_from_features,
    filter_intro_content_from_features,
    split_mixed_segments,
    find_viral_clips,
    
    # Utility functions
    create_segment_hash,
    debug_segment_scoring,
    detect_podcast_genre,
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
    
    # Dynamic segmentation functions
    "find_natural_boundaries",
    "create_dynamic_segments",
    
    # Pipeline functions
    "filter_ads_from_features",
    "filter_intro_content_from_features",
    "split_mixed_segments", 
    "find_viral_clips",
    
    # Explanation and analysis functions
    "_explain_viral_potential_v4",
    "_grade_breakdown",
    "_score_to_grade",
    "_heuristic_title",
    
    # Platform and tone mapping
    "PLATFORM_GENRE_MULTIPLIERS",
    "TONE_TO_GENRE_MAP",
    "PLATFORM_MAP",
    "resolve_platform",
    "resolve_genre_from_tone",
    "interpret_synergy",
    "get_genre_detection_debug",
    
    # Advanced API functions
    "find_viral_clips_with_tone",
    "find_viral_clips_with_genre",
    "find_candidates",
    
    # Question/List scoring V2
    "_question_list_raw_v2",
    "question_list_score_v2",
    "attach_question_list_scores_v2",
    
    # Utility functions
    "create_segment_hash",
    "debug_segment_scoring",
    "detect_podcast_genre",
    
    # Additional functions from monolithic file
    "_grade_breakdown",
    "_heuristic_title",
    "_sigmoid",
    "resolve_platform",
]
