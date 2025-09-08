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

# Placeholder for now - we'll add functions one by one