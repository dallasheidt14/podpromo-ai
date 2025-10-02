# backend/services/audio_features.py
# -------------------------------------------------------------------
# Lightweight audio prosody features for speech-driven virality scoring.
# - Keeps dependencies optional (librosa only if available)
# - Computes robust per-segment prosody features (RMS dynamics, pauses,
#   spectral centroid variance, laugh proxy, pitch variance/range)
# - Provides episode-level normalization stats
# - Provides arousal_score_v2: text_arousal ⨉ prosody (env-configurable)
# -------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# -------- Optional dependency (only used for pitch via YIN) --------
try:
    import librosa  # optional, only for F0
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# -------- Settings (feature flags & weights) with safe fallbacks ----
try:
    # Preferred path
    from backend.config.settings import (
        ENABLE_PROSODY_V2,
        AROUSAL_V2_BLEND_TEXT,
        AROUSAL_V2_BLEND_AUDIO,
        ENABLE_SPECTRAL_FLUX,
        RMS_AUDIO_WEIGHT,
        F0_AUDIO_WEIGHT,
        FLUX_AUDIO_WEIGHT,
        VOICED_FRAC_MIN,
        PAUSE_FRAC_MAX,
        FLUX_AB_PCT,
    )
except Exception:
    # Safe defaults if settings not wired yet
    ENABLE_PROSODY_V2 = True
    AROUSAL_V2_BLEND_TEXT = 0.60
    AROUSAL_V2_BLEND_AUDIO = 0.40
    ENABLE_SPECTRAL_FLUX = False
    RMS_AUDIO_WEIGHT = 0.6
    F0_AUDIO_WEIGHT = 0.3
    FLUX_AUDIO_WEIGHT = 0.1
    VOICED_FRAC_MIN = 0.30
    PAUSE_FRAC_MAX = 0.85
    FLUX_AB_PCT = 0.0


# ------------------------------ Helpers -----------------------------

def _slice(y: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    return y[i0:i1]


def _rms(x: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    if x.size == 0:
        return np.zeros(1, dtype=np.float32)
    pad = frame_length // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    n = max(1, 1 + (len(xpad) - frame_length) // hop_length)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_length
        out[i] = np.sqrt(np.mean(xpad[s:s + frame_length] ** 2) + 1e-12)
    return out


def _zero_cross_rate(x: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    if x.size == 0:
        return np.zeros(1, dtype=np.float32)
    pad = frame_length // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    n = max(1, 1 + (len(xpad) - frame_length) // hop_length)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_length
        frame = xpad[s:s + frame_length]
        out[i] = np.mean(np.abs(np.diff(np.sign(frame)))) * 0.5
    return out


def _spectral_centroid(x: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Cheap centroid: FFT mag * freq / sum mag."""
    if x.size == 0:
        return np.zeros(1, dtype=np.float32)
    pad = frame_length // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    n = max(1, 1 + (len(xpad) - frame_length) // hop_length)
    win = np.hanning(frame_length).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_length
        frame = xpad[s:s + frame_length] * win
        spec = np.abs(np.fft.rfft(frame))
        denom = float(np.sum(spec) + 1e-9)
        out[i] = float(np.sum(spec * freqs) / denom)
    return out


def _f0_features(
    x: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: int = 75,
    fmax: int = 400,
) -> Tuple[float, float, float]:
    """
    Return (f0_var, f0_range, voiced_frac) in ~[0..1].
    Uses librosa.yin if available; otherwise returns zeros.
    """
    if not _HAS_LIBROSA or x.size == 0:
        return 0.0, 0.0, 0.0
    try:
        f0 = librosa.yin(
            x, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length
        )
        voiced = (f0 > 0) & (~np.isinf(f0)) & (~np.isnan(f0))
        if not np.any(voiced):
            return 0.0, 0.0, 0.0
        f0v = f0[voiced]
        # variance (log space) → expressive variation
        v = float(np.var(np.log1p(f0v)))
        v01 = float(np.clip(v / (v + 0.5), 0.0, 1.0))
        # range P10–P90 → melodic span
        p10, p90 = np.percentile(f0v, 10), np.percentile(f0v, 90)
        rng = float(np.clip((p90 - p10) / (p90 + 1e-9), 0.0, 1.0))
        voiced_frac = float(np.clip(np.mean(voiced), 0.0, 1.0))
        return v01, rng, voiced_frac
    except Exception:
        return 0.0, 0.0, 0.0


def _normalize_01(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    lo, hi = np.percentile(arr, 10), np.percentile(arr, 90)
    rng = hi - lo
    if rng <= 1e-9:
        return 0.5  # Return neutral value instead of 0.0
    x = (np.mean(arr) - lo) / rng
    return 0.0 if x != x or x == float("inf") or x == float("-inf") else float(np.clip(x, 0.0, 1.0))


def _var_01(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    v = float(np.var(arr))
    return float(np.clip(v / (v + 1.0), 0.0, 1.0))


def _delta_01(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    p10, p90 = np.percentile(arr, 10), np.percentile(arr, 90)
    return float(np.clip((p90 - p10) / (p90 + 1e-9), 0.0, 1.0))


def _spectral_flux_scalar(x: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> float:
    """Compute spectral flux as a scalar value"""
    if x.size == 0:
        return 0.0
    win = np.hanning(n_fft).astype(np.float32)
    frames = max(1, 1 + (len(x) - n_fft) // hop_length)
    if frames < 2:
        return 0.0
    prev = None
    acc, cnt = 0.0, 0
    for i in range(frames):
        s = i * hop_length
        frame = x[s:s+n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag = np.abs(np.fft.rfft(frame * win))
        if prev is not None:
            diff = mag - prev
            acc += float(np.sqrt(np.sum(diff * diff)))
            cnt += 1
        prev = mag
    return acc / max(1, cnt)


def _estimate_voiced_frac_no_pitch(x: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Combine energy and ZCR to approximate voiced frames when no pitch available"""
    if x.size == 0:
        return 0.0
    rms = _rms(x, frame_length=frame_length, hop_length=hop_length)
    zcr = _zero_cross_rate(x, frame_length=frame_length, hop_length=hop_length)
    peak = float(np.max(rms) + 1e-9)
    # -25 dB below peak as energy gate
    e_gate = peak * (10.0 ** (-25.0 / 20.0))
    voiced_like = (rms >= e_gate) & (zcr <= 0.25)  # speech tends to have lower ZCR than unvoiced/noise
    return float(np.mean(voiced_like))


# ------------------------ Segment-level Features --------------------

def compute_prosody(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute per-segment prosody features. Returns a dict with:
      - rms_var, rms_delta, rms_lift_0_3s
      - pause_frac
      - centroid_var
      - laugh_flag
      - f0_var, f0_range, voiced_frac
    """
    seg = _slice(y, sr, start_s, end_s)
    if seg.size == 0:
        return {
            "rms_var": 0.0,
            "rms_delta": 0.0,
            "rms_lift_0_3s": 0.0,
            "pause_frac": 0.0,
            "centroid_var": 0.0,
            "laugh_flag": False,
            "f0_var": 0.0,
            "f0_range": 0.0,
            "voiced_frac": 0.0,
        }

    # RMS dynamics
    rms = _rms(seg)
    rms_norm = rms / (float(np.max(rms)) + 1e-9)
    rms_var = _var_01(rms_norm)
    rms_delta = _delta_01(rms_norm)

    # First 0–3s lift vs window mean
    first = _slice(seg, sr, 0.0, min(3.0, (end_s - start_s)))
    first_rms = float(np.sqrt(np.mean(first ** 2) + 1e-12)) if first.size else 0.0
    win_rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    rms_lift = float(np.clip((first_rms - win_rms) / (win_rms + 1e-9), -0.5, 1.0))
    rms_lift_0_3s = float(np.interp(rms_lift, [-0.5, 1.0], [0.0, 1.0]))

    # Pauses: fraction of samples below -35 dB of peak
    peak = float(np.max(np.abs(seg)) + 1e-12)
    thresh = peak * (10.0 ** (-35 / 20.0))
    pause_frac = float(np.clip(np.mean(np.abs(seg) < thresh), 0.0, 1.0)) if peak > 0 else 0.0

    # Spectral centroid dynamics (proxy for brightness/expressivity)
    cent = _spectral_centroid(seg, sr)
    cent_var = _var_01(cent)

    # Laughter proxy: sudden broadband spikes (centroid + RMS jump + ZCR)
    zcr = _zero_cross_rate(seg)
    spikes = (rms_norm[1:] - rms_norm[:-1] > 0.25).astype(np.float32)
    laugh_like = (spikes.mean() > 0.08) and (_delta_01(cent) > 0.25) and (_normalize_01(zcr) > 0.5)
    laugh_flag = bool(laugh_like)

    # Pitch features (if enabled and librosa available)
    f0_var = f0_range = voiced_frac = 0.0
    use_pitch = True
    if config and "pitch" in config:
        use_pitch = bool(config["pitch"].get("enabled", True))
        pf = config["pitch"]
        fl = int(pf.get("frame_length", 2048))
        hl = int(pf.get("hop_length", 512))
        fmin = int(pf.get("fmin", 75))
        fmax = int(pf.get("fmax", 400))
    else:
        fl, hl, fmin, fmax = 2048, 512, 75, 400

    if use_pitch:
        f0_var, f0_range, voiced_frac = _f0_features(seg, sr, frame_length=fl, hop_length=hl, fmin=fmin, fmax=fmax)

    return {
        "rms_var": float(rms_var),
        "rms_delta": float(rms_delta),
        "rms_lift_0_3s": float(rms_lift_0_3s),
        "pause_frac": float(pause_frac),
        "centroid_var": float(cent_var),
        "laugh_flag": bool(laugh_flag),
        "f0_var": float(f0_var),
        "f0_range": float(f0_range),
        "voiced_frac": float(voiced_frac),
    }


# ---------------------- Episode-level Normalization -----------------

def episode_prosody_stats(segments_prosody: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Build per-episode means/stds for normalization. Pass a list of dicts
    returned by compute_prosody (for each segment).
    Keys computed: rms_var_mean/std, f0_var_mean/std
    """
    keys = ("rms_var", "f0_var")
    stats: Dict[str, float] = {}
    for key in keys:
        vals = [float(s.get(key, 0.0)) for s in segments_prosody if key in s]
        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            stats[f"{key}_mean"] = float(np.mean(arr))
            stats[f"{key}_std"] = float(np.std(arr) + 1e-6)
        else:
            stats[f"{key}_mean"] = 0.0
            stats[f"{key}_std"] = 1.0
    return stats


# -------------------------- Arousal v2 Blend ------------------------

def blend_arousal_v2(
    text_arousal: float,
    prosody: Dict[str, float],
    episode_stats: Optional[Dict[str, float]] = None
) -> float:
    """
    Combine text-based arousal with prosody-driven energy into [0..1].
    Uses episode-level z-normalization on rms_var & f0_var to be mic/loudness agnostic.
    Controlled by ENABLE_PROSODY_V2 and AROUSAL_V2_BLEND_* weights.
    """
    if not ENABLE_PROSODY_V2:
        return float(text_arousal or 0.0)

    episode_stats = episode_stats or {}
    def z(val: float, mean_key: str, std_key: str) -> float:
        mean = float(episode_stats.get(mean_key, 0.0))
        std = float(episode_stats.get(std_key, 1.0))
        return (float(val) - mean) / max(1e-6, std)

    z_rms = z(prosody.get("rms_var", 0.0), "rms_var_mean", "rms_var_std")
    z_f0  = z(prosody.get("f0_var", 0.0),  "f0_var_mean",  "f0_var_std")

    # squash to [-1,1], average, then map to [0,1]
    p = 0.5 * (np.tanh(0.75 * z_rms) + np.tanh(0.75 * z_f0))
    p01 = float(0.5 * (p + 1.0))

    blended = (
        float(AROUSAL_V2_BLEND_TEXT) * float(text_arousal or 0.0) +
        float(AROUSAL_V2_BLEND_AUDIO) * p01
    )
    return float(np.clip(blended, 0.0, 1.0))


# --------------- Convenience: one-shot per-segment compute ----------

def segment_prosody(y, sr, t0, t1):
    """
    Return raw prosody metrics for a segment: dict(prosody_rms, prosody_flux).
    Compatible interface for the scoring pipeline.
    """
    feats = compute_prosody(y, sr, t0, t1)
    return {
        "prosody_rms": feats.get("rms_mean", 0.0),
        "prosody_flux": feats.get("spectral_flux", 0.0)
    }

def arousal_score_v2(text_arousal: float, prosody: dict, stats: dict) -> float:
    """
    Blend text arousal with episode-normalized prosody → [0,1].
    Compatible interface for the scoring pipeline.
    """
    return blend_arousal_v2(text_arousal, prosody, stats)

def prosody_with_arousal_v2(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    text_arousal: float,
    episode_stats: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Convenience helper that returns the prosody dict PLUS arousal_score_v2 blended in.
    Useful if you want a single call in your segment loop.
    """
    feats = compute_prosody(y, sr, start_s, end_s, config=config)
    feats["arousal_score_v2"] = blend_arousal_v2(text_arousal, feats, episode_stats or {})
    return feats
