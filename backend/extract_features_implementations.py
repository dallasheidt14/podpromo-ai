#!/usr/bin/env python3
"""
Script to extract feature computation functions from the monolithic file
and move them to features.py with proper implementations.
"""

import re
import os

def extract_functions():
    # Read the monolithic file
    with open('services/secret_sauce_pkg/__init__monolithic.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # List of feature functions to extract (in order of dependency)
    feature_functions = [
        # Core feature computation functions
        'compute_features_v4',
        'compute_features_v4_batch', 
        'compute_features',
        'compute_features_cached',
        
        # Individual feature functions
        '_hook_score',
        '_hook_score_v4',
        '_hook_score_v5',
        '_emotion_score',
        '_emotion_score_v4',
        '_payoff_presence',
        '_payoff_presence_v4',
        '_detect_payoff',
        '_info_density',
        '_info_density_v4',
        '_info_density_raw_v2',
        '_question_or_list',
        '_loopability_heuristic',
        '_arousal_score_text',
        '_audio_prosody_score',
        '_detect_insight_content',
        '_detect_insight_content_v2',
        '_apply_insight_confidence_multiplier',
        '_calculate_niche_penalty',
        '_ad_penalty',
        '_platform_length_match',
        'calculate_dynamic_length_score',
        
        # Utility functions
        'create_segment_hash',
        'debug_segment_scoring',
        
        # Hook V5 functions
        'compute_audio_hook_modifier',
        'detect_laughter_exclamations',
        'calculate_hook_components',
        'calculate_time_weighted_hook_score',
        'score_patterns_in_text',
        '_saturating_sum',
        '_proximity_bonus',
        '_normalize_quotes_lower',
        '_first_clause',
        '_get_hook_cues_from_config',
        '_family_score',
        '_evidence_guard',
        '_anti_intro_outro_penalties',
        '_audio_micro_for_hook',
        '_sigmoid',
        '_sigmoid01',
        'attach_hook_scores',
        
        # Helper functions
        '_text_based_audio_estimation',
        '_calibrate_hook_v5',
        '_calibrate_info_density_stats',
        '_ql_calibrate_stats',
        '_calibrate_emotion_stats',
        'build_emotion_audio_sidecar',
        'compute_audio_energy',
        'info_density_score_v2',
        'question_list_score_v2',
        'emotion_score_v2',
    ]
    
    extracted_functions = []
    missing_functions = []
    
    for func_name in feature_functions:
        # Find function definition with more flexible pattern
        # Look for function definition and capture until next function/class or end of file
        pattern = rf'^def {re.escape(func_name)}\(.*?(?=^def |^class |^# |^from |^import |^$|^[A-Z])'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        
        if match:
            func_code = match.group(0).rstrip()
            # Clean up trailing keywords
            for keyword in ['def ', 'class ', '# ', 'from ', 'import ']:
                if func_code.endswith(keyword):
                    func_code = func_code[:-len(keyword)]
                    break
            extracted_functions.append((func_name, func_code))
            print(f"✅ Extracted {func_name}")
        else:
            missing_functions.append(func_name)
            print(f"❌ Could not find {func_name}")
    
    # Create the new features.py content
    features_content = '''"""
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

'''

    # Add all extracted functions
    for func_name, func_code in extracted_functions:
        features_content += func_code + '\n\n'
    
    # Add __all__ export list
    features_content += '''
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
    "_info_density_raw_v2",
    "_question_or_list",
    "_loopability_heuristic",
    "_arousal_score_text",
    "_audio_prosody_score",
    "_detect_insight_content",
    "_detect_insight_content_v2",
    "_apply_insight_confidence_multiplier",
    "_calculate_niche_penalty",
    "_ad_penalty",
    "_platform_length_match",
    "calculate_dynamic_length_score",
    
    # Utility functions
    "create_segment_hash",
    "debug_segment_scoring",
    
    # Hook V5 functions
    "compute_audio_hook_modifier",
    "detect_laughter_exclamations",
    "calculate_hook_components",
    "calculate_time_weighted_hook_score",
    "score_patterns_in_text",
    "_saturating_sum",
    "_proximity_bonus",
    "_normalize_quotes_lower",
    "_first_clause",
    "_get_hook_cues_from_config",
    "_family_score",
    "_evidence_guard",
    "_anti_intro_outro_penalties",
    "_audio_micro_for_hook",
    "_sigmoid",
    "_sigmoid01",
    "attach_hook_scores",
]
'''
    
    # Write the new features.py
    with open('services/secret_sauce_pkg/features.py', 'w', encoding='utf-8') as f:
        f.write(features_content)
    
    print(f"\n✅ Extracted {len(extracted_functions)} functions to features.py")
    print(f"❌ Missing {len(missing_functions)} functions: {missing_functions}")
    return extracted_functions, missing_functions

if __name__ == "__main__":
    extract_functions()
