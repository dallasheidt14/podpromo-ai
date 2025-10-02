"""
Deterministic random number generation utilities.
Ensures reproducible results across all random number generators.
"""

import hashlib
import os
import random
import logging

logger = logging.getLogger(__name__)

# Optional imports for different RNG libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

def seed_everything(episode_id: str) -> int:
    """
    Seed all random number generators with a deterministic value derived from episode_id.
    
    This ensures reproducible results across:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator
    
    Args:
        episode_id: Unique identifier for the episode
        
    Returns:
        The seed value used
    """
    # Generate deterministic seed from episode_id
    h = int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16)
    
    # Seed Python's random module
    random.seed(h)
    
    # Seed NumPy if available
    if HAS_NUMPY:
        np.random.seed(h)
        logger.debug(f"RNG_SEED: NumPy seeded with {h}")
    
    # Seed PyTorch if available
    if HAS_TORCH:
        torch.manual_seed(h)
        torch.cuda.manual_seed_all(h)
        # Keep performance by not using deterministic algorithms by default
        # Set to True only for strict reproducibility requirements
        torch.use_deterministic_algorithms(False)
        
        # Set CUBLAS workspace config for deterministic behavior when needed
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        
        logger.debug(f"RNG_SEED: PyTorch seeded with {h}")
    
    logger.info(f"RNG_SEED: All generators seeded with {h} for episode {episode_id}")
    return h

def get_deterministic_seed(episode_id: str) -> int:
    """
    Get a deterministic seed value for an episode without seeding generators.
    
    Useful when you need the seed value but don't want to modify global state.
    
    Args:
        episode_id: Unique identifier for the episode
        
    Returns:
        Deterministic seed value
    """
    return int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16)

def reset_random_state():
    """
    Reset random number generators to unseeded state.
    
    Useful for testing or when you want to return to non-deterministic behavior.
    """
    random.seed()
    
    if HAS_NUMPY:
        np.random.seed()
    
    if HAS_TORCH:
        torch.manual_seed(torch.initial_seed())
    
    logger.debug("RNG_SEED: Reset all generators to unseeded state")
