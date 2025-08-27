"""
Tests for the config loader functionality.
"""

import json
import os
import sys
import tempfile
import shutil

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from config_loader import load_config, get_config, BASE_DIR


def test_load_defaults_when_missing():
    """Test that defaults are loaded when config file is missing"""
    # Save original config path
    original_path = os.path.join(BASE_DIR, "config", "secret_config.json")
    backup_path = original_path + ".backup"
    
    try:
        # Move config file temporarily
        if os.path.exists(original_path):
            shutil.move(original_path, backup_path)
        
        # Test loading
        cfg = load_config()
        assert "weights" in cfg and "lexicons" in cfg
        assert cfg["weights"]["hook"] > 0
        assert "HOOK_CUES" in cfg["lexicons"]
        
    finally:
        # Restore config file
        if os.path.exists(backup_path):
            shutil.move(backup_path, original_path)


def test_merge_user_config():
    """Test that user config is properly merged with defaults"""
    cfg_dir = os.path.join(BASE_DIR, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    user_path = os.path.join(cfg_dir, "secret_config.json")
    
    # Create test user config
    user = {
        "weights": {
            "hook": 0.4, "prosody": 0.2, "emotion": 0.1, 
            "q_or_list": 0.1, "payoff": 0.1, "info": 0.05, "loop": 0.05
        },
        "lexicons": {
            "EMO_WORDS": ["hilarious", "wild"]
        }
    }
    
    # Save test config
    with open(user_path, "w", encoding="utf-8") as f:
        json.dump(user, f)
    
    try:
        # Test loading
        cfg = load_config()
        assert abs(cfg["weights"]["hook"] - 0.4) < 1e-6
        assert "hilarious" in cfg["lexicons"]["EMO_WORDS"]
        
        # Test that other defaults are preserved
        assert "HOOK_CUES" in cfg["lexicons"]
        assert cfg["weights"]["prosody"] == 0.2
        
    finally:
        # Clean up test config
        if os.path.exists(user_path):
            os.remove(user_path)


def test_get_config_caching():
    """Test that get_config uses caching"""
    cfg1 = get_config()
    cfg2 = get_config()
    
    # Should be the same object (cached)
    assert cfg1 is cfg2


def test_config_structure():
    """Test that config has expected structure"""
    cfg = get_config()
    
    # Check weights
    assert "weights" in cfg
    weight_keys = ["hook", "prosody", "emotion", "q_or_list", "payoff", "info", "loop"]
    for key in weight_keys:
        assert key in cfg["weights"]
        assert isinstance(cfg["weights"][key], (int, float))
    
    # Check lexicons
    assert "lexicons" in cfg
    lexicon_keys = ["HOOK_CUES", "EMO_WORDS", "FILLERS", "PAYOFF_MARKERS", "QUESTION_STARTS", "LIST_MARKERS"]
    for key in lexicon_keys:
        assert key in cfg["lexicons"]
        assert isinstance(cfg["lexicons"][key], list)
    
    # Check weights sum to approximately 1.0
    weight_sum = sum(cfg["weights"].values())
    assert abs(weight_sum - 1.0) < 0.01, f"Weights sum to {weight_sum}, expected ~1.0"
