#!/usr/bin/env python3
"""
Script to extract feature computation functions from secret_sauce.py
and move them to features.py with proper implementations.
"""

import re

def extract_feature_functions():
    # Read the original file
    with open('services/secret_sauce.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # List of functions to extract
    functions_to_extract = [
        'compute_features_v4',
        'compute_features_v4_batch', 
        'compute_features',
        'compute_features_cached',
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
        '_question_or_list',
        '_loopability_heuristic',
        '_arousal_score_text',
        '_audio_prosody_score',
        '_detect_insight_content',
        '_detect_insight_content_v2',
        '_calculate_niche_penalty',
        '_ad_penalty',
        '_platform_length_match',
        'calculate_dynamic_length_score',
        'create_segment_hash',
        'debug_segment_scoring',
        '_info_density_raw_v2'
    ]
    
    # Extract each function
    extracted_functions = []
    remaining_content = content
    
    for func_name in functions_to_extract:
        # Find function definition
        pattern = rf'^def {re.escape(func_name)}\(.*?^def |^class |^# |^from |^import |^$'
        match = re.search(pattern, remaining_content, re.MULTILINE | re.DOTALL)
        
        if match:
            func_code = match.group(0).rstrip()
            # Remove the trailing 'def ' or 'class ' from next item
            if func_code.endswith('def '):
                func_code = func_code[:-4]
            elif func_code.endswith('class '):
                func_code = func_code[:-6]
            
            extracted_functions.append(func_code)
            # Remove this function from remaining content
            remaining_content = remaining_content.replace(func_code, '', 1)
            print(f"Extracted {func_name}")
        else:
            print(f"Could not find {func_name}")
    
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
    for func_code in extracted_functions:
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
    "_calculate_niche_penalty",
    "_ad_penalty",
    "_platform_length_match",
    "calculate_dynamic_length_score",
    
    # Utility functions
    "create_segment_hash",
    "debug_segment_scoring",
]
'''
    
    # Write the new features.py
    with open('services/secret_sauce_pkg/features.py', 'w', encoding='utf-8') as f:
        f.write(features_content)
    
    print(f"Extracted {len(extracted_functions)} functions to features.py")
    
    # Now remove these functions from the original file
    # This is complex, so let's do it step by step
    print("Functions extracted. Next step: remove from original file.")

if __name__ == "__main__":
    extract_feature_functions()
