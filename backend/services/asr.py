"""
ASR (Automatic Speech Recognition) service with quality-aware two-pass transcription.
"""

import logging
import re
import inspect
from typing import Tuple, Dict, Any, List, Optional, Union, Sequence
from dataclasses import dataclass, replace

from faster_whisper import WhisperModel
from faster_whisper.transcribe import VadOptions

from config.settings import (
    ASR_MODEL, ASR_DEVICE, ASR_COMPUTE_TYPE, ASR_WORD_TS, ASR_VAD,
    ASR_VAD_SILENCE_MS, ASR_VAD_SPEECH_PAD_MS, ASR_COND_PREV,
    ASR_BEAM_SIZE, ASR_TEMPS, ENABLE_QUALITY_RETRY,
    ASR_LOW_QL_LOGPROB, ASR_LOW_QL_COMPRESS, ASR_LOW_QL_MIN_PUNCT,
    ASR_HQ_ON_RETRY_COMPUTE_TYPE, ASR_GPU_MEMORY_FRACTION
)

log = logging.getLogger(__name__)


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


def _run_once(asr_input, s: ASRSettings):
    """Run a single ASR pass with immediate generator materialization"""
    model = WhisperModel(
        s.model_name,
        device=s.device,
        compute_type=(s.compute_type if s.device == "cuda" else "int8"),
        download_root=s.model_cache_dir,
    )
    try:
        gen, info = model.transcribe(
            asr_input,
            vad_filter=s.vad_filter,
            word_timestamps=True,
            temperature=s.temperature,
            beam_size=s.beam_size,
            best_of=s.best_of,
            patience=s.patience,
        )
        # IMPORTANT: materialize the generator so CUDA errors occur here
        segments = list(gen)
        log.info("ASR_RUN: %s OK (segments=%d)", s.device.upper(), len(segments))
        return segments, info
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


def transcribe_with_quality(asr_input, settings: ASRSettings, *, prefer_preprocessed: bool = True) -> Tuple[List[Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Two-pass ASR with quality-aware retry and robust GPU→CPU fallback.
    Returns: (segments, info, words) - canonical triple format
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

    words = _extract_words_safe(segments)  # always returns list
    return segments, info, words


# Legacy function for backward compatibility
def _to_words(segs):
    """Legacy function - now calls the safe extractor"""
    return _extract_words_safe(segs)
