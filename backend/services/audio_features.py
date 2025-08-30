# backend/services/audio_features.py
import numpy as np

# --- add near top with other imports ---
try:
    import librosa  # optional, only for F0
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

def _slice(y, sr, t0, t1):
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    return y[i0:i1]

def _rms(x, frame_length=2048, hop_length=512):
    if x.size == 0:
        return np.zeros(1)
    # simple RMS without librosa dependency
    pad = frame_length // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    n = 1 + (len(xpad) - frame_length) // hop_length
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_length
        out[i] = np.sqrt(np.mean(xpad[s:s+frame_length]**2) + 1e-12)
    return out

def _zero_cross_rate(x, frame_length=2048, hop_length=512):
    if x.size == 0:
        return np.zeros(1)
    pad = frame_length // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    n = 1 + (len(xpad) - frame_length) // hop_length
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_length
        frame = xpad[s:s+frame_length]
        out[i] = np.mean(np.abs(np.diff(np.sign(frame)))) * 0.5
    return out

def _spectral_centroid(x, sr, frame_length=2048, hop_length=512):
    # cheap centroid: FFT mag * freq / sum mag
    if x.size == 0:
        return np.zeros(1)
    pad = frame_length // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    n = 1 + (len(xpad) - frame_length) // hop_length
    win = np.hanning(frame_length).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0/sr)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop_length
        frame = xpad[s:s+frame_length] * win
        spec = np.abs(np.fft.rfft(frame))
        denom = np.sum(spec) + 1e-9
        out[i] = float(np.sum(spec * freqs) / denom)
    return out

def _f0_features(x, sr, frame_length=2048, hop_length=512, fmin=75, fmax=400):
    """
    Return f0_var (0..1), f0_range (0..1), voiced_frac (0..1).
    Uses librosa.yin if available; otherwise returns zeros.
    """
    if not _HAS_LIBROSA or x.size == 0:
        return 0.0, 0.0, 0.0
    try:
        # search range good for most speech (75–400 Hz)
        f0 = librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length)
        # voiced frames filter
        voiced = (f0 > 0) & (~np.isinf(f0)) & (~np.isnan(f0))
        if not np.any(voiced):
            return 0.0, 0.0, 0.0
        f0v = f0[voiced]
        # normalize features
        # variance → expressive variation
        v = float(np.var(np.log1p(f0v)))
        v01 = float(np.clip(v / (v + 0.5), 0.0, 1.0))
        # range P10–P90 → melodic span
        p10, p90 = np.percentile(f0v, 10), np.percentile(f0v, 90)
        rng = float(np.clip((p90 - p10) / (p90 + 1e-9), 0.0, 1.0))
        voiced_frac = float(np.clip(np.mean(voiced), 0.0, 1.0))
        return v01, rng, voiced_frac
    except Exception:
        return 0.0, 0.0, 0.0

def _normalize_01(arr):
    if arr.size == 0:
        return 0.0
    lo, hi = np.percentile(arr, 10), np.percentile(arr, 90)
    if hi - lo < 1e-9:
        return 0.0
    return float(np.clip((np.mean(arr) - lo) / (hi - lo), 0.0, 1.0))

def _var_01(arr):
    if arr.size == 0:
        return 0.0
    return float(np.clip(np.var(arr) / (np.var(arr) + 1.0), 0.0, 1.0))

def _delta_01(arr):
    if arr.size == 0:
        return 0.0
    p10, p90 = np.percentile(arr, 10), np.percentile(arr, 90)
    return float(np.clip((p90 - p10) / (p90 + 1e-9), 0.0, 1.0))

def compute_prosody(y, sr, start_s, end_s, config=None):
    """
    Returns dict with:
      rms_var, rms_delta, rms_lift_0_3s, pause_frac, centroid_var, laugh_flag,
      f0_var, f0_range, voiced_frac
    All scaled ~0..1 where possible.
    
    Args:
        y: audio samples
        sr: sample rate
        start_s: start time in seconds
        end_s: end time in seconds
        config: optional config dict with pitch settings
    """
    seg = _slice(y, sr, start_s, end_s)
    if seg.size == 0:
        return {
            "rms_var": 0.0, "rms_delta": 0.0, "rms_lift_0_3s": 0.0,
            "pause_frac": 0.0, "centroid_var": 0.0, "laugh_flag": False
        }
    # RMS dynamics
    rms = _rms(seg)
    rms_norm = rms / (np.max(rms) + 1e-9)
    rms_var = _var_01(rms_norm)
    rms_delta = _delta_01(rms_norm)

    # first 0-3s lift vs window mean
    first = _slice(seg, sr, 0.0, min(3.0, (end_s - start_s)))
    first_rms = float(np.sqrt(np.mean(first**2) + 1e-12)) if first.size else 0.0
    win_rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
    rms_lift = float(np.clip((first_rms - win_rms) / (win_rms + 1e-9), -0.5, 1.0))
    rms_lift_0_3s = float(np.interp(rms_lift, [-0.5, 1.0], [0.0, 1.0]))

    # pauses: fraction of frames below -35 dB of window peak
    peak = np.max(np.abs(seg)) + 1e-12
    thresh = peak * (10 ** (-35/20.0))
    pauses = np.mean(np.abs(seg) < thresh) if peak > 0 else 0.0
    pause_frac = float(np.clip(pauses, 0.0, 1.0))

    # spectral centroid dynamics (proxy for brightness/expressivity)
    cent = _spectral_centroid(seg, sr)
    cent_var = _var_01(cent)

    # laughter proxy: sudden broadband spikes (centroid + rms jump + ZCR)
    zcr = _zero_cross_rate(seg)
    spikes = (rms_norm[1:] - rms_norm[:-1] > 0.25).astype(np.float32)
    laugh_like = (spikes.mean() > 0.08) and (_delta_01(cent) > 0.25) and (_normalize_01(zcr) > 0.5)
    laugh_flag = bool(laugh_like)

    # NEW: pitch features (with config if available)
    if config and config.get("pitch", {}).get("enabled", True):
        pitch_cfg = config["pitch"]
        f0_var, f0_range, voiced_frac = _f0_features(
            seg, sr,
            frame_length=pitch_cfg.get("frame_length", 2048),
            hop_length=pitch_cfg.get("hop_length", 512),
            fmin=pitch_cfg.get("fmin", 75),
            fmax=pitch_cfg.get("fmax", 400)
        )
    else:
        f0_var, f0_range, voiced_frac = 0.0, 0.0, 0.0

    return {
        "rms_var": rms_var,
        "rms_delta": rms_delta,
        "rms_lift_0_3s": rms_lift_0_3s,
        "pause_frac": float(pause_frac),
        "centroid_var": cent_var,
        "laugh_flag": laugh_flag,
        "f0_var": f0_var,
        "f0_range": f0_range,
        "voiced_frac": voiced_frac
    }
