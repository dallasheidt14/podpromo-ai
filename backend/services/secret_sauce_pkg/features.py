"""
Feature computation module for viral detection system.
Contains feature extraction and computation functions.
"""

from typing import Dict, List, Tuple, Any
import logging
import numpy as np
import librosa
import re
import hashlib
from functools import lru_cache
from scipy import signal
from scipy.stats import skew, kurtosis
from config_loader import get_config

logger = logging.getLogger(__name__)

# Import dependencies from other modules
from .scoring import get_clip_weights
from .genres import GenreAwareScorer

# For now, import from the monolithic file to maintain functionality
# We'll gradually move functions here
from services.secret_sauce_pkg.__init__monolithic import (
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

# Export all functions
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
