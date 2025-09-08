"""
Feature computation module for viral detection system.
Contains feature extraction and computation functions.
"""

from typing import Dict, List, Tuple, Any
import logging
from config_loader import get_config

logger = logging.getLogger(__name__)

# Import all feature computation functions from the original secret_sauce.py
# This is a temporary solution - we'll gradually move these functions here
from services.secret_sauce import (
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

# Re-export all the imported functions
__all__ = [
    # Main feature computation functions
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
]
