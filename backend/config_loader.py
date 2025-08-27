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
        "hook": 0.35, "prosody": 0.20, "emotion": 0.15,
        "q_or_list": 0.10, "payoff": 0.10, "info": 0.05, "loop": 0.05
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
    except Exception:
        pass
    _cache = cfg
    return cfg

def get_config():
    return _cache or load_config()
