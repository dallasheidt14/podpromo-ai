"""
secret_sauce.py
All heuristics + weights for picking 'winning' clips.
Edit here to evolve your edge (cues, weights, thresholds).
"""

from typing import Dict, List, Tuple
import numpy as np, librosa
from config_loader import get_config

_cfg = get_config()
CLIP_WEIGHTS = _cfg["weights"]

HOOK_CUES        = tuple(_cfg["lexicons"]["HOOK_CUES"])
EMO_WORDS        = tuple(_cfg["lexicons"]["EMO_WORDS"])
FILLERS          = tuple(_cfg["lexicons"]["FILLERS"])
PAYOFF_MARKERS   = tuple(_cfg["lexicons"]["PAYOFF_MARKERS"])
QUESTION_STARTS  = tuple(_cfg["lexicons"]["QUESTION_STARTS"])
LIST_MARKERS     = tuple(_cfg["lexicons"]["LIST_MARKERS"])

# -----------------------------
# Text feature helpers
# -----------------------------
def _hook_score(text: str) -> float:
    t = text.lower()[:200]
    hits = sum(1 for cue in HOOK_CUES if cue in t)
    return float(min(hits / 3.0, 1.0))

def _emotion_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for w in EMO_WORDS if w in t)
    return float(min(hits / 3.0, 1.0))

def _question_or_list(text: str) -> float:
    t = text.strip().lower()
    is_q = 1.0 if "?" in t or any(t.startswith(s) for s in QUESTION_STARTS) else 0.0
    has_list = 1.0 if any(x in t for x in LIST_MARKERS) else 0.0
    return float(min(is_q + 0.5 * has_list, 1.0))

def _info_density(text: str) -> float:
    words = text.split()
    if not words: return 0.0
    fillers = sum(1 for w in words if w.lower().strip(".,?!") in FILLERS)
    ratio = 1.0 - min(fillers / max(len(words),1), 1.0)
    # bonus if avg sentence length is short (snappy)
    avg_len_bonus = 0.1 if len(words) / max(text.count(".")+text.count("!")+text.count("?"), 1) <= 16 else 0.0
    return float(np.clip(ratio + avg_len_bonus, 0.0, 1.0))

def _payoff_presence(text: str) -> float:
    t = text.lower()
    return 1.0 if any(p in t for p in PAYOFF_MARKERS) else (0.4 if " is " in t or " are " in t else 0.0)

def _loopability_heuristic(text: str) -> float:
    # simple default; refine later by visual match or last->first phoneme similarity
    ends_clean = text.strip().endswith((".", "!", "?"))
    return 0.7 if ends_clean else 0.4

# -----------------------------
# Audio feature helpers
# -----------------------------
def _audio_prosody_score(audio_path: str, start: float, end: float) -> float:
    """
    Combine RMS dynamics + rough pitch variance near the start (hook).
    """
    y, sr = librosa.load(audio_path, sr=None, offset=max(0, start-0.2), duration=(end-start+0.4))
    if y.size == 0:
        return 0.0

    # RMS dynamics
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
    dyn = 0.0
    if rms.size > 10:
        p95 = float(np.percentile(rms, 95))
        p50 = float(np.percentile(rms, 50))
        if p50 > 0:
            dyn = np.clip((p95 / p50) - 1.0, 0.0, 2.0) / 2.0

    # Pitch variance (rough)
    try:
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[mags > np.median(mags)]  # keep voiced-ish
        pitch_std = float(np.std(pitch_vals[pitch_vals > 0])) if pitch_vals.size else 0.0
        # normalize pitch_std by a rough scalar to ~0..1 band
        pit = float(np.clip(pitch_std / 100.0, 0.0, 1.0))
    except Exception:
        pit = 0.0

    return float(np.clip(0.7 * dyn + 0.3 * pit, 0.0, 1.0))

# -----------------------------
# Public API used by clip_engine
# -----------------------------
def compute_features(segment: Dict, audio_file: str) -> Dict:
    text = segment.get("text","")
    feats = {
        "hook_score":     _hook_score(text),
        "prosody_score":  _audio_prosody_score(audio_file, segment["start"], segment["end"]),
        "emotion_score":  _emotion_score(text),
        "question_score": _question_or_list(text),
        "payoff_score":   _payoff_presence(text),
        "info_density":   _info_density(text),
        "loopability":    _loopability_heuristic(text),
    }
    return feats

def score_segment(features: Dict, weights: Dict = CLIP_WEIGHTS) -> float:
    return float(
        weights["hook"]   * features["hook_score"] +
        weights["prosody"]* features["prosody_score"] +
        weights["emotion"]* features["emotion_score"] +
        weights["q_or_list"]* features["question_score"] +
        weights["payoff"] * features["payoff_score"] +
        weights["info"]   * features["info_density"] +
        weights["loop"]   * features["loopability"]
    )

# Optional: nightly adjustment hook you can call with user choices
def adjust_weights_from_feedback(weights: Dict, accepted: List[Dict], rejected: List[Dict]) -> Dict:
    """
    Super-simple online tuner: if accepted clips skew high on a feature, nudge weight up.
    Use with caution; keep nudges tiny (e.g., ±0.01) and clamp to reasonable bounds.
    """
    import statistics as stats

    def avg(name, items):
        vals = [c["features"][name] for c in items if "features" in c and name in c["features"]]
        return stats.mean(vals) if vals else 0.0

    names = ["hook_score","prosody_score","emotion_score","question_score","payoff_score","info_density","loopability"]
    new = dict(weights)
    for n in names:
        diff = avg(n, accepted) - avg(n, rejected)
        if abs(diff) > 0.05:
            key = {
                "hook_score": "hook", "prosody_score": "prosody", "emotion_score": "emotion",
                "question_score": "q_or_list", "payoff_score":"payoff", "info_density":"info", "loopability":"loop"
            }[n]
            new[key] = float(np.clip(new[key] + 0.01*np.sign(diff), 0.02, 0.6))
    # normalize to sum≈1
    s = sum(new.values())
    for k in new: new[k] = float(new[k]/s)
    return new
