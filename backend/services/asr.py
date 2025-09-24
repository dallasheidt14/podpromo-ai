"""
ASR (Automatic Speech Recognition) service with quality-aware two-pass transcription.
"""

from __future__ import annotations

import logging
import re
import inspect
import math
from typing import Tuple, Dict, Any, List, Optional, Union, Sequence, TypedDict
import time
from dataclasses import dataclass, replace

from faster_whisper import WhisperModel
from faster_whisper.transcribe import VadOptions
import ctranslate2

from config.settings import (
    ASR_MODEL, ASR_DEVICE, ASR_COMPUTE_TYPE, ASR_WORD_TS, ASR_VAD,
    ASR_VAD_SILENCE_MS, ASR_VAD_SPEECH_PAD_MS, ASR_COND_PREV,
    ASR_BEAM_SIZE, ASR_TEMPS, ENABLE_QUALITY_RETRY,
    ASR_LOW_QL_LOGPROB, ASR_LOW_QL_COMPRESS, ASR_LOW_QL_MIN_PUNCT,
    ASR_HQ_ON_RETRY_COMPUTE_TYPE, ASR_GPU_MEMORY_FRACTION
)

log = logging.getLogger(__name__)


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Calculate mean of valid numeric values, ignoring NaN and None."""
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return (sum(nums) / len(nums)) if nums else None


def _estimate_asr_conf(segments: List[Dict[str, Any]]) -> float:
    """
    Estimate a [0..1] ASR confidence for a list of normalized segments.

    Heuristic priority:
      1) Mean of word-level probabilities if present (words[].prob|confidence|p)
      2) Mean of segment-level confidence proxies:
          - sigmoid(avg_logprob)
          - 'confidence' if provided by some backends
      3) 1 - mean(no_speech_prob)
      4) Fallback 0.5 if nothing available
    """
    if not segments:
        return 0.5

    word_probs: List[float] = []
    seg_confs: List[float] = []
    inv_no_speech: List[float] = []

    for s in segments:
        # 1) word-level confidences (preferred when available)
        words = s.get("words") or []
        for w in words:
            # normalized dicts expected; normalize_asr_result should already coerce objects
            p = (w.get("prob") if isinstance(w, dict) else None)  # standard FW field
            if p is None and isinstance(w, dict):
                p = w.get("confidence", w.get("p"))
            if isinstance(p, (int, float)):
                word_probs.append(float(p))

        # 2) segment-level confidence proxies
        #    - avg_logprob is in log space; squash to (0,1) with logistic
        ap = s.get("avg_logprob")
        if isinstance(ap, (int, float)) and not math.isnan(ap):
            seg_confs.append(1.0 / (1.0 + math.exp(-float(ap))))

        cp = s.get("confidence")
        if isinstance(cp, (int, float)) and not math.isnan(cp):
            # already a [0..1] in some implementations
            seg_confs.append(float(cp))

        # 3) 1 - no_speech_prob (higher is better)
        nsp = s.get("no_speech_prob")
        if isinstance(nsp, (int, float)) and not math.isnan(nsp):
            inv_no_speech.append(max(0.0, min(1.0, 1.0 - float(nsp))))

    # choose first available, in order of preference
    for candidate in (_safe_mean(word_probs), _safe_mean(seg_confs), _safe_mean(inv_no_speech)):
        if candidate is not None:
            return max(0.0, min(1.0, float(candidate)))

    # 4) final fallback
    return 0.5


# Stall detection error for watchdog
class AsrStallError(RuntimeError):
    """Raised when Faster-Whisper generator yields no items for too long."""
    pass

# --- Define ASRSettings BEFORE any function annotations that use it ---
@dataclass
class ASRSettings:
    """ASR configuration settings with CPU fallback support"""
    # NEW core fields
    model_name: str = ASR_MODEL
    device: str = ASR_DEVICE
    compute_type: str = ASR_COMPUTE_TYPE
    vad_filter: bool = ASR_VAD
    model_cache_dir: Optional[str] = None

    # KEEP existing knobs
    word_timestamps: bool = ASR_WORD_TS
    vad_silence_ms: int = ASR_VAD_SILENCE_MS
    vad_speech_pad_ms: int = ASR_VAD_SPEECH_PAD_MS
    condition_previous: bool = ASR_COND_PREV
    beam_size: Optional[int] = ASR_BEAM_SIZE
    best_of: Optional[int] = None
    patience: Optional[float] = None
    temperature: Union[float, Sequence[float]] = 0.0
    temperatures: str = ASR_TEMPS  # Keep for backward compatibility

    # Quality retry flags
    enable_quality_retry: bool = ENABLE_QUALITY_RETRY
    retry_temperature: float = 0.2
    retry_beam_size: int = 5
    low_ql_logprob: float = ASR_LOW_QL_LOGPROB
    low_ql_compress: float = ASR_LOW_QL_COMPRESS
    low_ql_min_punct: float = ASR_LOW_QL_MIN_PUNCT
    hq_retry_compute_type: str = ASR_HQ_ON_RETRY_COMPUTE_TYPE
    gpu_memory_fraction: float = ASR_GPU_MEMORY_FRACTION

    # ADD these (safe defaults, no None gets into ct2)
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    max_initial_timestamp: float = 1.0
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = None  # allow FW default

    # Legacy alias (back-compat)
    @property
    def model(self) -> str:
        return self.model_name
    
    @model.setter
    def model(self, v: str) -> None:
        self.model_name = v

    def with_cpu_fallback(self) -> 'ASRSettings':
        """Create a copy with CPU fallback settings"""
        import os
        
        # Check if forced to CPU
        force_cpu = os.getenv("FORCE_CPU_WHISPER") == "1"
        
        return replace(
            self,
            device="cpu",
            compute_type="int8" if force_cpu else "int8"  # Always use int8 for CPU fallback
        )
    
    def clone(self, **updates):
        """Clone settings with updates - works regardless of object type"""
        new = ASRSettings()
        new.__dict__.update(self.__dict__)
        new.__dict__.update(updates)
        return new


# Strongly-typed alias for ASR results
class ASRResult(TypedDict, total=False):
    segments: List[Dict[str, Any]]  # ALWAYS present; list of dicts
    info: Dict[str, Any]            # ALWAYS present
    words: List[Dict[str, Any]]     # ALWAYS present (possibly synthesized)
    asr_confidence: float


def _resolve_device(device: str) -> str:
    """Resolve device with proper CUDA detection using ctranslate2"""
    if device in ("cuda", "cpu"):
        return device
    
    # Check for explicit CPU override
    import os
    if os.getenv("FORCE_CPU_WHISPER", "0") == "1":
        return "cpu"
    
    # auto → prefer CUDA only if ctranslate2 sees a device
    try:
        if getattr(ctranslate2, "get_cuda_device_count", lambda: 0)() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _sanitize_fw_options(s: ASRSettings) -> Dict[str, Any]:
    """
    Build sanitized kwargs for Faster-Whisper with no None values, and keep
    Windows-friendly defaults (single worker, simple decode) to avoid deadlocks.
    """
    def _list(x):
        if x is None:
            return [0.0]
        return x if isinstance(x, list) else [x]

    opts = {
        # decoding knobs (keep simple for stability)
        "beam_size": int(getattr(s, "beam_size", 1) or 1),
        "best_of": int(getattr(s, "best_of", 1) or 1),
        "patience": float(getattr(s, "patience", 1.0) or 1.0),
        "length_penalty": float(getattr(s, "length_penalty", 1.0) or 1.0),
        "repetition_penalty": float(getattr(s, "repetition_penalty", 1.0) or 1.0),
        "no_repeat_ngram_size": int(getattr(s, "no_repeat_ngram_size", 0) or 0),

        # stability knobs
        "suppress_blank": bool(getattr(s, "suppress_blank", True)),
        "suppress_tokens": getattr(s, "suppress_tokens", None) or None,
        "word_timestamps": bool(getattr(s, "word_timestamps", True)),
        "max_initial_timestamp": float(getattr(s, "max_initial_timestamp", 1.0) or 1.0),

        # decoding/robustness
        "temperature": _list(getattr(s, "temperature", [0.0])),
        "condition_on_previous_text": bool(getattr(s, "condition_on_previous_text", False)),
        "vad_filter": bool(getattr(s, "vad_filter", True)),

        # optional pass-throughs
        "without_timestamps": bool(getattr(s, "without_timestamps", False)),
        "prepend_punctuations": getattr(s, "prepend_punctuations", None) or None,
        "append_punctuations": getattr(s, "append_punctuations", None) or None,
        "language": getattr(s, "language", None) or None,
        # NOTE: task handled below
    }

    # Only include `task` if valid; else omit so FW defaults to "transcribe"
    raw_task = getattr(s, "task", None)
    task = (raw_task or "").strip().lower() if isinstance(raw_task, str) else raw_task
    if task in ("transcribe", "translate"):
        opts["task"] = task

    # Ensure word timestamps are requested for confidence estimation
    opts.setdefault("word_timestamps", True)
    
    # Drop any keys that are None (FW is sensitive to some Nones)
    return {k: v for k, v in opts.items() if v is not None}


def _consume_with_watchdog(gen, stall_seconds: float = 30.0, log_every: int = 20):
    """
    Consume Faster-Whisper segments generator with stall detection and progress logs.
    Raises AsrStallError if no progress for `stall_seconds`.
    """
    out: List[Any] = []
    last = time.monotonic()
    for i, seg in enumerate(gen, 1):
        if log.isEnabledFor(logging.DEBUG):
            try:
                start_ts = getattr(seg, "start", -1.0)
                end_ts = getattr(seg, "end", -1.0)
                text_preview = (getattr(seg, "text", "") or "")[:60]
                log.debug("ASR_SEG: #%d (%.2f→%.2f) '%s'", i, start_ts, end_ts, text_preview)
            except Exception:
                log.debug("ASR_SEG: #%d (unprintable segment)", i)

        out.append(seg)
        if (i % log_every) == 0:
            log.info("ASR_GEN: consumed %d segments...", i)
        last = time.monotonic()
        # small cooperative yield for Windows
        if (i % 50) == 0:
            time.sleep(0.001)
        # stall check
        if time.monotonic() - last > stall_seconds:
            raise AsrStallError(f"ASR generator stalled > {stall_seconds}s after {i} segments")

    if len(out) == 0:
        log.info("ASR_GEN: no segments yielded (empty audio or stall before first segment)")
    return out


def _settings_to_cpu(s: "ASRSettings") -> "ASRSettings":
    return s.clone(device="cpu", compute_type="int8")


def _run_once(asr_input, s: "ASRSettings"):
    """Run a single ASR pass with watchdog-based generator consumption and failover."""
    device = _resolve_device(s.device)
    cpu_threads = int(getattr(s, "cpu_threads", 4) or 4)
    model_workers = int(getattr(s, "num_workers", 1) or 1)
    model = WhisperModel(
        s.model_name,
        device=device,
        compute_type=(s.compute_type if device == "cuda" else "int8"),
        download_root=s.model_cache_dir,
        cpu_threads=cpu_threads,
        num_workers=model_workers,
    )
    try:
        # Build sanitized options so ct2.generate never sees None/incorrect types
        options = _sanitize_fw_options(s)
        
        # Guard: if someone set an invalid task later in the pipeline, remove it
        if "task" in options and options["task"] not in ("transcribe", "translate"):
            del options["task"]
        
        log.info(
            "ASR_RUN: starting transcribe (device=%s, compute=%s, opts=%s)",
            device,
            (s.compute_type if device == "cuda" else "int8"),
            {k: options[k] for k in ("beam_size", "temperature", "vad_filter", "word_timestamps") if k in options},
        )
        gen, info = model.transcribe(asr_input, **options)
        # Consume with watchdog
        segments = _consume_with_watchdog(
            gen,
            stall_seconds=float(getattr(s, "stall_seconds", 30.0)),
            log_every=int(getattr(s, "log_every", 20)),
        )
        log.info("ASR_RUN: %s OK (segments=%d)", device.upper(), len(segments))
        return segments, info
    except AsrStallError:
        log.error("ASR stall detected; retrying with CPU+minimal options", exc_info=True)
        cpu = s.clone(device="cpu", compute_type="int8") if hasattr(s, "clone") else s
        # Fail-safe options: simplest temperature, avoid VAD edge-cases, single worker
        fail_safe_opts = {
            **_sanitize_fw_options(cpu),
            "temperature": [0.0],
            "vad_filter": False,
            "num_workers": 1,
        }
        cpu_model = WhisperModel(
            getattr(cpu, "model_name", s.model_name),
            device="cpu",
            compute_type="int8",
            download_root=getattr(cpu, "model_cache_dir", s.model_cache_dir),
            cpu_threads=cpu_threads,
        )
        try:
            gen2, info2 = cpu_model.transcribe(asr_input, **fail_safe_opts)
            segments2 = _consume_with_watchdog(
                gen2,
                stall_seconds=float(getattr(cpu, "stall_seconds", 30.0)),
                log_every=int(getattr(cpu, "log_every", 40)),
            )
            log.info("ASR_RUN: CPU OK (segments=%d)", len(segments2))
            return segments2, info2
        finally:
            del cpu_model
    finally:
        del model  # helps on Windows


def _needs_retry(segments: List[Any]) -> bool:
    """Check if segments need quality retry"""
    # Simple heuristic - you can enhance this
    if not segments:
        return True
    
    # Check for very short segments (might indicate poor quality)
    total_duration = sum(getattr(s, 'end', 0) - getattr(s, 'start', 0) for s in segments)
    if total_duration < 1.0:  # Less than 1 second total
        return True
    
    return False


def _is_better(new_segments: List[Any], old_segments: List[Any]) -> bool:
    """Check if new segments are better than old ones"""
    # Simple heuristic - you can enhance this
    if not new_segments:
        return False
    if not old_segments:
        return True
    
    # Prefer longer total duration
    new_duration = sum(getattr(s, 'end', 0) - getattr(s, 'start', 0) for s in new_segments)
    old_duration = sum(getattr(s, 'end', 0) - getattr(s, 'start', 0) for s in old_segments)
    
    return new_duration > old_duration


def _extract_words_safe(segments: List[Any], words: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Extract word timestamps safely - ALWAYS returns a list, never None"""
    out = []
    
    # If words provided, normalize them first
    if words:
        from services.util import _normalize_schema
        normalized = _normalize_schema(words)
        if normalized:
            return normalized
    
    # Otherwise synthesize from segments
    for s in segments or []:
        if getattr(s, "words", None):
            for w in s.words:
                # normalize both schema variants
                if hasattr(w, "start"):
                    out.append({"start": w.start, "end": w.end, "text": w.word})
                else:
                    out.append({
                        "start": w.get("t", 0.0), 
                        "end": w.get("t", 0.0) + w.get("d", 0.0), 
                        "text": w.get("w", "")
                    })
        else:
            # synthesize from segment text if needed
            text = getattr(s, "text", "").strip()
            if text:
                out.append({
                    "start": getattr(s, "start", 0.0), 
                    "end": getattr(s, "end", 0.0), 
                    "text": text
                })
    return out


def _punct_quality(texts: List[str]) -> float:
    """% of sentences that end with .!? (crude but useful signal)."""
    if not texts:
        return 0.0
    sents, enders = 0, 0
    for t in texts:
        parts = re.split(r'(?<=[\.\!\?])\s+', (t or "").strip())
        for p in parts:
            if not p:
                continue
            sents += 1
            if re.search(r'[\.!\?]\s*$', p):
                enders += 1
    return enders / max(1, sents)


def _mk_vad_params(min_sil_ms: int, speech_pad_ms: int) -> Dict[str, int]:
    """
    Build VAD parameters with correct naming for Faster-Whisper.
    Always use 'min_silence_duration_ms' (the correct parameter name).
    """
    return {
        "min_silence_duration_ms": int(min_sil_ms),
        "speech_pad_ms": int(speech_pad_ms)
    }


def _transcribe_once(
    model: WhisperModel,
    audio_path: str,
    *,
    language: str = None,
    word_ts: bool,
    vad_on: bool,
    min_sil_ms: int,
    pad_ms: int,
    beam_size: int,
    temperature: Any
):
    kwargs = dict(
        language=language,
        word_timestamps=word_ts,
        condition_on_previous_text=ASR_COND_PREV,
        beam_size=beam_size,
        temperature=temperature,
    )
    if vad_on:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = _mk_vad_params(min_sil_ms, pad_ms)
    else:
        kwargs["vad_filter"] = False

    return model.transcribe(audio_path, **kwargs)


def transcribe_with_quality(asr_input, settings: "ASRSettings", *, prefer_preprocessed: bool = True) -> ASRResult:
    """
    Two-pass ASR with quality-aware retry and robust GPU→CPU fallback.
    Returns: ASRResult with normalized shapes
    """
    if settings is None:
        settings = ASRSettings()
    
    # Pass-1 with provided settings
    segments, info = _run_once(asr_input, settings)

    # Optional Pass-2 (quality retry) — still materialized
    if settings.enable_quality_retry and _needs_retry(segments):
        retry = replace(settings, temperature=settings.retry_temperature, beam_size=settings.retry_beam_size)
        seg2, info2 = _run_once(asr_input, retry)
        if _is_better(seg2, segments):
            segments, info = seg2, info2

    # Hard normalize shapes right here so downstream never sees raw ct2 types
    from services.util import normalize_asr_result, AttrDict, to_attrdict_list, coerce_word_schema
    from services.asr_helpers import ensure_words
    
    norm_segments, norm_info, words = normalize_asr_result({"segments": segments, "info": info})
    
    # Coerce every word into the canonical schema to prevent KeyError: 'word'
    if words:
        words = [coerce_word_schema(w) for w in words]
        words = to_attrdict_list(words)
    
    # Also normalize per-segment .words if present
    for s in norm_segments:
        if "words" in s and s["words"]:
            s["words"] = [coerce_word_schema(w) for w in s["words"]]
            s["words"] = to_attrdict_list(s["words"])
    
    # ensure both segments and words support attribute access
    segments_ad = to_attrdict_list(norm_segments)
    
    # also ensure words list has AttrDict items
    words_result = ensure_words(norm_segments, words)
    if isinstance(words_result, list):
        words_result = [coerce_word_schema(w) for w in words_result]
        words_result = to_attrdict_list(words_result)
    
    return {
        "segments": segments_ad,   # AttrDict list so seg.start / seg.text works
        "info": norm_info or {},     # dict
        "words": words_result,  # AttrDict list - synthesized if needed
        "asr_confidence": _estimate_asr_conf(norm_segments),
    }


# Legacy function for backward compatibility
def _to_words(segs):
    """Legacy function - now calls the safe extractor"""
    return _extract_words_safe(segs)
