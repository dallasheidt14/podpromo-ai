"""
Config loader/merger for secret_sauce.
- Loads JSON at startup.
- Supports live reload via /config/reload (see api.py snippet below).
- Safe defaults if files missing.
"""
import json
import os

DEFAULT = {
    "weights": {
        "hook": 0.35, "arousal": 0.20, "emotion": 0.15,
        "q_or_list": 0.10, "payoff": 0.10, "info": 0.05, "loop": 0.05, "platform_len": 0.05
    },
    "lexicons": {
        "HOOK_CUES": [
            "wrong","myth","everyone thinks","but actually","the mistake",
            "3 rules","5 rules","step 1","step one","top 3","top three",
            "you absolutely need to","you need to","the key is","whatever you do",
            "i was wrong","i changed my mind","what if","how do you","why do we","should you"
        ],
        "EMO_WORDS": [
            "unbelievable","crazy","wild","insane","shocked","shock","hate","love",
            "fear","nightmare","wow","hilarious","funny"
        ],
        "FILLERS": ["um","uh","like","you know","sort of","kinda"],
        "PAYOFF_MARKERS": [
            "because","so ","so,","therefore","here's why","the point is",
            "which means","in short","bottom line","tl;dr","the key is"
        ],
        "QUESTION_STARTS": ["what","how","why","when","where","who","should","would","could"],
        "LIST_MARKERS": [" 2 "," 3 "," 4 "," step "," rule "," rules "," tips "," tricks "]
    }
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "secret_config.json")
PRESETS_DIR = os.path.join(BASE_DIR, "config", "presets")

_cache = None

def load_config():
    global _cache
    cfg = DEFAULT.copy()
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user = json.load(f)
            # shallow merge
            if "weights" in user: 
                cfg["weights"].update(user["weights"])
            if "lexicons" in user:
                for k, v in user["lexicons"].items():
                    if k in cfg["lexicons"] and isinstance(v, list):
                        cfg["lexicons"][k] = v
            # Load V2 configuration sections
            if "features" in user:
                cfg["features"] = user["features"]
            if "length_targets" in user:
                cfg["length_targets"] = user["length_targets"]
            if "diversity" in user:
                cfg["diversity"] = user["diversity"]
            if "pitch" in user:
                cfg["pitch"] = user["pitch"]
            if "arousal" in user:
                cfg["arousal"] = user["arousal"]
            if "presets" in user:
                cfg["presets"] = user["presets"]
            
            # Debug: Log what was loaded
            print(f"DEBUG: Loaded config features: {user.get('features', {})}")
            print(f"DEBUG: Loaded config weights: {user.get('weights', {})}")
    except Exception as e:
        print(f"DEBUG: Config load error: {e}")
        pass
    _cache = cfg
    return cfg

def get_config():
    cfg = _cache or load_config()
    return cfg_defaults(cfg)

def reload_config():
    """Reload configuration from files"""
    global _cache
    _cache = None
    return load_config()

def set_weights(weights: dict):
    """Set custom weights for scoring"""
    try:
        # Validate weights - updated to use new key names
        required_keys = ["hook", "arousal", "payoff", "info", "q_or_list", "loop", "platform_len"]
        if not all(key in weights for key in required_keys):
            return {"ok": False, "error": f"Missing required weight keys: {required_keys}"}
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total == 0:
            return {"ok": False, "error": "Weights cannot all be zero"}
        
        normalized_weights = {k: v/total for k, v in weights.items()}
        
        # Load current config
        current_config = get_config()
        
        # Update weights
        current_config["weights"] = normalized_weights
        
        # Save updated config
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=2)
        
        # Reload configuration
        reload_config()
        
        return normalized_weights
        
    except Exception as e:
        return {"ok": False, "error": str(e)}

def load_preset(preset_name: str):
    """Load a preset configuration"""
    try:
        preset_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        if not os.path.exists(preset_path):
            return {"ok": False, "error": "Preset not found"}
        
        # Read preset
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        
        # Overwrite main config with preset
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2)
        
        # Reload configuration
        reload_config()
        
        return preset.get("weights", {})
        
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ============================================================================
# V2 Config Helpers
# ============================================================================

import copy
from typing import Dict

def cfg_defaults(cfg: Dict) -> Dict:
    """Set default values for v2 configuration structure"""
    cfg.setdefault("features", {})
    cfg.setdefault("weights", {
        "hook":0.35,"arousal":0.20,"payoff":0.15,"info":0.10,"loop":0.08,"q_or_list":0.07,"platform_len":0.05
    })
    cfg.setdefault("length_targets", {})
    cfg.setdefault("diversity", {"min_time_gap_s":30, "min_cosine_distance":0.25})
    cfg.setdefault("pitch", {"fmin":75, "fmax":400, "frame_length":2048, "hop_length":512, "enabled":True})
    cfg.setdefault("arousal", {
        "alpha_synergy": 0.25,
        "weights": {
            "rms_var": 0.20,
            "rms_delta": 0.12,
            "lift": 0.18,
            "centroid_var": 0.10,
            "anti_pause": 0.12,
            "laugh": 0.06,
            "f0_var": 0.14,
            "f0_range": 0.06,
            "voiced_frac": 0.02,
            "exclam_boost": 0.04
        }
    })
    cfg.setdefault("presets", {"platform":{}, "tone_mods":{}, "aliases":{}})
    return cfg

def is_on(key: str, cfg: Dict = None) -> bool:
    """Check if a v2 feature flag is enabled"""
    if cfg is None:
        cfg = get_config()
    return bool(cfg.get("features", {}).get(key, False))

def compose_preset(base_weights: Dict, platform_key: str, tone_key: str|None, cfg: Dict) -> Dict:
    """Compose weights from platform + tone combination"""
    # start from platform weights
    plat = cfg.get("presets",{}).get("platform",{}).get(platform_key, {})
    w = copy.deepcopy(plat.get("weights", base_weights))
    if tone_key:
        delta = cfg.get("presets",{}).get("tone_mods",{}).get(tone_key, {})
        for k, dv in delta.items():
            w[k] = max(0.0, w.get(k, 0.0) + dv)
    # renormalize ~1.0
    s = sum(w.values()) or 1.0
    for k in w:
        w[k] = round(w[k]/s, 4)
    return w

def resolve_platform_lengths(platform_key: str, cfg: Dict):
    """Get length targets for a specific platform"""
    lk = cfg.get("presets",{}).get("platform",{}).get(platform_key, {}).get("length_key", platform_key)
    lt = cfg.get("length_targets", {}).get(lk)
    return lt or {"sweet_spot":[15,25],"okay":[12,30]}

def resolve_legacy_preset(name: str, cfg: Dict):
    """Resolve legacy preset names to platform+tone combinations"""
    alias = cfg.get("presets",{}).get("aliases",{}).get(name)
    return alias  # {"platform": "...", "tone": "..."}
