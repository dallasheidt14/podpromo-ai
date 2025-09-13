"""
ClipScore Service - Glue layer between secret_sauce and the rest of the system.
This service orchestrates the clip scoring pipeline using the proprietary algorithms.
"""

import numpy as np
import logging
import re
import random
import math
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from models import AudioFeatures, TranscriptSegment, MomentScore
from config.settings import (
    UPLOAD_DIR, OUTPUT_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    PRERANK_ENABLED, TOP_K_RATIO, TOP_K_MIN, TOP_K_MAX, STRATIFY_ENABLED, 
    SAFETY_KEEP_ENABLED, COMPARE_SCORING_MODES, PRERANK_WEIGHTS,
    DURATION_TARGET_MIN, DURATION_TARGET_MAX
)

# Quality gate thresholds
THRESHOLDS = {
    "strict": {
        "payoff_min": 0.12,   # was ~0.20; allow more clips with moderate payoff
        "ql_max": 0.78,       # allow a bit more "QL" penalty on interviews
        "hook_min": 0.08,
        "arousal_min": 0.20,
    },
    "balanced": {
        "payoff_min": 0.08,
        "ql_max": 0.85,
        "hook_min": 0.06,
        "arousal_min": 0.15,
    },
    "relaxed": {
        "payoff_min": 0.05,
        "ql_max": 0.90,
        "hook_min": 0.04,
        "arousal_min": 0.10,
    }
}

def _gate(candidates: List[Dict], mode: str) -> List[Dict]:
    """Apply quality gate with specified mode"""
    if not candidates:
        return []
    
    thresholds = THRESHOLDS.get(mode, THRESHOLDS["balanced"])
    passed = []
    
    for c in candidates:
        # Check if clip passes quality gates
        payoff = c.get("payoff_score", 0.0)
        hook = c.get("hook_score", 0.0)
        arousal = c.get("arousal_score", 0.0)
        info_density = c.get("info_density", 0.0)
        
        # Calculate quality penalty (QL)
        ql_penalty = 0.0
        if info_density > 0:
            ql_penalty = min(1.0, (1.0 - info_density) * 0.5)
        
        # Apply thresholds
        if (payoff >= thresholds["payoff_min"] and 
            hook >= thresholds["hook_min"] and 
            arousal >= thresholds["arousal_min"] and
            ql_penalty <= thresholds["ql_max"]):
            passed.append(c)
    
    logger.info(f"QUALITY_GATE[{mode}]: {len(candidates)} -> {len(passed)}")
    return passed

def apply_quality_gate(candidates: List[Dict], mode: str = "strict") -> List[Dict]:
    """Apply quality gate with auto-relaxation when too few clips remain"""
    passed = _gate(candidates, mode)
    
    if len(passed) < 3 and mode == "strict":
        # Auto relax to get usable output
        logger.info(f"QUALITY_GATE: strict yielded {len(passed)} < 3, auto-relaxing to balanced")
        relaxed = _gate(candidates, "balanced")
        
        # Keep order stable: take what strict allowed, then add best relaxed
        seen = {id(c) for c in passed}
        for c in relaxed:
            if id(c) not in seen:
                passed.append(c)
            if len(passed) >= 3:
                break
    
    return passed

def adaptive_gate(candidates: List[Dict], min_count: int = 3) -> List[Dict]:
    """Adaptive gate that adds clips from remaining pool by score + platform-fit"""
    base = [c for c in candidates if c.get("quality_ok", False)]
    if len(base) >= min_count:
        return base
    
    # Build pool of non-ads, non-questions
    pool = [
        c for c in candidates
        if c not in base and not c.get("is_ad", False) and not c.get("is_question", False)
    ]
    
    # Sort by score, then platform fit, then info density
    pool.sort(key=lambda c: (
        c.get("final_score", 0.0),
        c.get("platform_length_score_v2", 0.0),
        c.get("info_density", 0.0)
    ), reverse=True)
    
    start_len = len(base)
    for c in pool:
        base.append(c)
        if len(base) >= min_count:
            break
    
    logger.info(f"ADAPTIVE_GATE: start={start_len} target={min_count} added={len(base)-start_len} final={len(base)}")
    return base

def enforce_soft_floor(candidates: List[Dict], min_count: int = 3) -> List[Dict]:
    """Enforce final soft floor after all gates"""
    if len(candidates) >= min_count:
        return candidates
    
    # Get all remaining candidates (non-ads)
    pool = [c for c in candidates if not c.get("is_ad", False)]
    
    # Sort by score
    pool.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
    
    # Add from pool until we reach min_count
    result = list(candidates)
    for c in pool:
        if c not in result:
            result.append(c)
        if len(result) >= min_count:
            break
    
    logger.info(f"SOFT_FLOOR: {len(candidates)} -> {len(result)} (target: {min_count})")
    return result

# --- helpers for authoritative finish-thought tracking ---
def authoritative_fallback_mode(result_meta):
    try:
        return bool(result_meta["ft"]["fallback_mode"])
    except Exception:
        # conservative default if meta missing
        return True

def normalize_ft_status(c):
    # ensure every candidate has an ft_status
    if not c.get("ft_status"):
        c["ft_status"] = "unresolved"
    return c

def _interval_iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0

def _finished_rank(meta):
    s = (meta or {}).get("ft_status")
    return 2 if s == "finished" else 1 if s == "sparse_finished" else 0

def _length_fit(start, end, target=18.0):
    return -abs((end - start) - target)

def _dedup_by_time_iou(cands, iou_thresh=0.85, target_len=18.0):
    if len(cands) <= 1: 
        return cands
    # sort: score desc, finished rank desc, length fit (closer to target) desc
    ordered = sorted(
        cands,
        key=lambda c: (
            c.get("display_score", 0.0),
            _finished_rank((c.get("meta") or {})),
            _length_fit(c["start"], c["end"], target_len),
        ),
        reverse=True,
    )
    kept = []
    for c in ordered:
        s, e = c["start"], c["end"]
        if any(_interval_iou(s, e, k["start"], k["end"]) >= iou_thresh for k in kept):
            continue
        kept.append(c)
    return kept

def _text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using simple word overlap"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize text: lowercase, remove punctuation, split into words
    import re
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0

def _dedup_by_text(cands, text_thresh=0.90):
    """Remove candidates with similar text content"""
    if len(cands) <= 1:
        return cands
    
    # Sort by score (best first)
    ordered = sorted(cands, key=lambda c: c.get("display_score", 0.0), reverse=True)
    
    kept = []
    for c in ordered:
        text = c.get("text", "") or c.get("transcript", "")
        if not text:
            kept.append(c)
            continue
            
        # Check against all kept candidates
        is_duplicate = False
        for k in kept:
            k_text = k.get("text", "") or k.get("transcript", "")
            if k_text and _text_similarity(text, k_text) >= text_thresh:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(c)
    
    return kept

def _format_candidate(seg):
    """Format a single segment into a candidate with ft_status propagation"""
    c = {
        "id": seg.get("id"),
        "start": float(seg["start"]),
        "end": float(seg["end"]),
        "duration": float(seg["end"] - seg["start"]),
        "text": seg.get("text", ""),
        "ft_status": seg.get("ft_status"),   # <-- carry through
        "ft_meta": seg.get("ft_meta", {}),   # optional but useful for logging
    }
    return c

def run_safety_or_shrink(candidate, eos_times, word_end_times, platform, fallback_mode):
    """Run safety/shrink logic on a single candidate and return status metadata"""
    from services.secret_sauce_pkg.features import finish_thought_normalize
    
    # Check if clip already ends on a finished thought
    if candidate.get("finished_thought", 0) == 1:
        return {"status": candidate.get("ft_status", "finished")}
    
    # Try to normalize this clip
    normalized_clip, result = finish_thought_normalize(candidate, eos_times, word_end_times, platform, fallback_mode)
    
    if result.get("status") == "unresolved":
        # Special handling for protected long clips
        if candidate.get("protected_long", False):
            dur = candidate.get('end', 0) - candidate.get('start', 0)
            if dur >= 18.0:  # Only for long clips
                return {"status": "unresolved", "reason": "protected_long"}
        
        # For non-protected clips or sparse EOS, be more lenient
        if fallback_mode:
            return {"status": "sparse_finished", "reason": "fallback_mode"}
        else:
            return {"status": "unresolved", "reason": "no_eos_hit"}
    else:
        # Clip was modified - update it
        candidate.update(normalized_clip)
        return result

from services.secret_sauce_pkg import compute_features_v4, score_segment_v4, explain_segment_v4, viral_potential_v4, get_clip_weights
from services.viral_moment_detector import ViralMomentDetector
from services.progress_writer import write_progress
from services.prerank import pre_rank_candidates, get_safety_candidates, pick_stratified
from services.quality_filters import fails_quality, filter_overlapping_candidates, filter_low_quality
from services.candidate_formatter import format_candidates

# Enhanced compute function with all Phase 1-3 features
from services.secret_sauce_pkg.features import compute_features_v4_enhanced as _compute_features

# Feature coverage logging
REQUIRED_FEATURE_KEYS = [
    "hook_score", "arousal_score", "payoff_score", "info_density",
    "loopability", "insight_score", "platform_len_match"
]
OPTIONAL_FEATURE_KEYS = [
    "insight_conf", "q_list_score", "prosody_arousal", "platform_length_score_v2", "emotion_score"
]

def _log_feature_coverage(logger, feats: dict):
    missing = [k for k in REQUIRED_FEATURE_KEYS if k not in feats]
    if missing:
        logger.warning(f"[features] Missing required: {missing}")
    soft_missing = [k for k in OPTIONAL_FEATURE_KEYS if k not in feats]
    if soft_missing:
        logger.info(f"[features] Optional not present: {soft_missing}")
from config_loader import get_config

# ensure v2 is ON everywhere
def is_on(_): return True  # temporary until flags are unified

logger = logging.getLogger(__name__)

def choose_default_clip(clips: List[Dict[str, Any]]) -> Optional[str]:
    """Choose the default clip for UI display"""
    if not clips:
        return None
    
    longs = [c for c in clips if c.get("protected_long")]
    if longs:
        longs.sort(key=lambda c: (-c.get("pl_v2", 0), -c.get("duration", 0), -c.get("final_score", 0)))
        return longs[0]["id"]
    
    rest = sorted(clips, key=lambda c: (-c.get("final_score", 0), -c.get("finished_thought", 0), -c.get("pl_v2", 0)))
    if rest:
        return rest[0]["id"]
    
    return None

def rank_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank clips with protected_long clips first"""
    longs = sorted([c for c in clips if c.get("protected_long")],
                   key=lambda c: (-c.get("pl_v2", 0), -c.get("duration", 0), -c.get("final_score", 0)))
    rest = sorted([c for c in clips if not c.get("protected_long")],
                  key=lambda c: (-c.get("final_score", 0), -c.get("finished_thought", 0), -c.get("pl_v2", 0)))
    ordered = longs + rest
    for i, c in enumerate(ordered): 
        c["rank_primary"] = i
    return ordered

def final_safety_pass(clips: List[Dict[str, Any]], eos_times: List[float], word_end_times: List[float], platform: str) -> List[Dict[str, Any]]:
    """
    Final safety pass to ensure all clips end on finished thoughts.
    This catches any clips that might have slipped through without proper EOS normalization.
    Always runs, even with sparse EOS data using fallback mode.
    """
    if not eos_times or not word_end_times:
        logger.warning("SAFETY_FALLBACK: No EOS data available, using relaxed mode")
        # In fallback mode, just mark all clips as finished to avoid drops
        for clip in clips:
            clip['finished_thought'] = 1
        return clips
    
    from services.secret_sauce_pkg.features import finish_thought_normalize, near_eos
    
    safety_updated = 0
    safety_dropped = 0
    safety_protected = 0
    
    # Check EOS density
    word_count = len(word_end_times)
    eos_density = len(eos_times) / max(word_count, 1)
    is_sparse = eos_density < 0.02 or word_count < 500
    
    if is_sparse:
        logger.warning(f"EOS_SPARSE: Using relaxed safety mode (density: {eos_density:.3f}, words: {word_count})")
    
    for clip in clips:
        # Check if clip already ends on a finished thought
        if clip.get("finished_thought", 0) == 1:
            continue
        
        # Try to normalize this clip
        # Determine fallback mode based on EOS density
        fallback_mode = len(eos_times) < 50 or len(word_end_times) < 500
        normalized_clip, result = finish_thought_normalize(clip, eos_times, word_end_times, platform, fallback_mode)
        
        if result == "unresolved":
            # Special handling for protected long clips
            if clip.get("protected_long", False):
                # Try last-resort extension with wider window
                dur = clip.get('end', 0) - clip.get('start', 0)
                if dur >= 18.0:  # Only for long clips
                    logger.warning(f"SAFETY_PROTECTED: keeping unresolved long clip {clip.get('id', 'unknown')} dur={dur:.1f}s")
                    clip['finished_thought'] = 0
                    clip['needs_review'] = True
                    safety_protected += 1
                    continue
            
            # For non-protected clips or sparse EOS, be more lenient
            if is_sparse:
                logger.warning(f"SAFETY_KEEP_SPARSE: keeping unresolved clip {clip.get('id', 'unknown')} due to sparse EOS")
                clip['finished_thought'] = 0
                clip['needs_review'] = True  # Mark for sparse EOS mode
                safety_protected += 1
                continue
            else:
                safety_dropped += 1
                logger.warning(f"SAFETY_DROP: clip {clip.get('id', 'unknown')} unresolved after safety pass")
                continue
        elif result != "snap_ok":
            # Clip was modified - update it
            clip.update(normalized_clip)
            safety_updated += 1
            
            # Safety updates must write ft_status back
            if isinstance(result, dict) and result.get("status"):
                clip["ft_status"] = result["status"]  # 'finished' | 'sparse_finished' | 'unresolved' | 'extended'
            
            logger.info(f"SAFETY_UPDATE: clip {clip.get('id', 'unknown')} -> {result}")
    
    # Calculate telemetry
    word_count = len(word_end_times)
    eos_count = len(eos_times)
    eos_density = eos_count / max(word_count, 1)
    fallback_mode = word_count < 500 or eos_count == 0
    
    # Count finish thought results
    finished_count = sum(1 for c in clips if c.get("finished_thought", 0) == 1)
    finished_ratio = finished_count / max(len(clips), 1)
    
    # Log episode-level telemetry
    logger.info(f"TELEMETRY: word_count={word_count}, eos_count={eos_count}, eos_density={eos_density:.3f}")
    logger.info(f"TELEMETRY: fallback_mode={fallback_mode}, finished_ratio={finished_ratio:.2f}")
    logger.info(f"SAFETY_PASS: updated {safety_updated}, dropped {safety_dropped}, protected {safety_protected} clips")
    
    # Episode-level thresholds
    if word_count < 500 or eos_count == 0:
        logger.warning("EOS_SPARSE: word_count < 500 or eos_count == 0")
    if finished_ratio < 0.95:
        logger.warning(f"FINISH_RATIO_LOW: {finished_ratio:.2f}")
    
    # Remove any clips that were marked for dropping (but keep protected ones)
    # In sparse EOS mode, be more lenient with finished_thought requirements
    if fallback_mode:
        return [c for c in clips if c.get("finished_thought", 0) == 1 or c.get("needs_review", False) or c.get("protected", False)]
    else:
        return [c for c in clips if c.get("finished_thought", 0) == 1 or c.get("needs_review", False) or near_eos(c.get("end", 0), eos_times, 0.5)]

class ClipScoreService:
    """Service for analyzing podcast episodes and scoring potential clip moments"""
    
    def __init__(self, episode_service):
        self.episode_service = episode_service


    def pre_rank_candidates(self, segments: List[Dict], episode_id: str) -> List[Dict]:
        return pre_rank_candidates(segments, episode_id)

    def get_safety_candidates(self, segments: List[Dict]) -> List[Dict]:
        return get_safety_candidates(segments)

    def pick_stratified(self, candidates: List[Dict], target_count: int) -> List[Dict]:
        return pick_stratified(candidates, target_count)

    def analyze_episode(self, audio_path: str, transcript: List[TranscriptSegment], episode_id: str = None) -> List[MomentScore]:
        """Analyze episode and return scored moments"""
        try:
            logger.info(f"Starting episode analysis for {audio_path}")
            
            # Convert transcript to segments for ranking using improved moment detection
            segments = self._transcript_to_segments(transcript, genre='general', platform='tiktok')
            
            # Rank candidates using secret sauce V4 with genre awareness
            # SPEED: Reduce top_k for faster processing (still processes all segments, just ranks fewer)
            top_k = int(os.getenv("TOP_K_CANDIDATES", "15"))  # Default 15, was 10
            ranked_segments = self.rank_candidates(segments, audio_path, top_k=top_k, platform='tiktok', genre='general', episode_id=episode_id)
            
            # Convert back to MomentScore objects
            moment_scores = []
            for seg in ranked_segments:
                moment_score = MomentScore(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    duration=seg["end"] - seg["start"],
                    hook_score=seg["features"]["hook_score"],
                    emotion_score=seg["features"]["emotion_score"],
                    arousal_score=seg["features"]["arousal_score"],
                    payoff_score=seg["features"]["payoff_score"],
                    loopability_score=seg["features"]["loopability"],
                    question_or_list_score=seg["features"]["question_score"],
                    info_density_score=seg["features"]["info_density"],
                    total_score=seg["score"]
                )
                moment_scores.append(moment_score)
            
            logger.info(f"Found {len(moment_scores)} potential moments")
            return moment_scores
            
        except Exception as e:
            logger.error(f"Episode analysis failed: {e}")
            raise

    def _is_repetitive_content(self, text: str) -> bool:
        """Check if content is repetitive or filler (like 'Yeah. Yeah. Yeah...')"""
        if not text or len(text.strip()) < 5:
            return True
        
        words = text.lower().split()
        if len(words) < 5:
            return True
        
        # Check for repetitive patterns
        # Count unique words vs total words
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        # If more than 70% of words are repeated, it's likely filler
        if repetition_ratio < 0.3:
            return True
        
        # Check for specific repetitive patterns
        repetitive_patterns = [
            r'^(yeah\.?\s*){3,}',  # "Yeah. Yeah. Yeah..."
            r'^(uh\.?\s*){3,}',    # "Uh. Uh. Uh..."
            r'^(um\.?\s*){3,}',    # "Um. Um. Um..."
            r'^(ok\.?\s*){3,}',    # "Ok. Ok. Ok..."
            r'^(right\.?\s*){3,}', # "Right. Right. Right..."
            r'^(so\.?\s*){3,}',    # "So. So. So..."
            r'^(and\.?\s*){3,}',   # "And. And. And..."
        ]
        
        import re
        for pattern in repetitive_patterns:
            if re.match(pattern, text.lower()):
                return True
        
        # Check if the same word appears more than 50% of the time
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_count = max(word_counts.values()) if word_counts else 0
        if max_count > len(words) * 0.5:
            return True
        
        return False

    def _fails_quality(self, feats: dict) -> str | None:
        """Check if segment fails quality gates (soft reject) - Soft hook penalty, strong ad clamp"""
        # Conditional hook threshold based on payoff and arousal
        payoff = feats.get("payoff_score", 0.0)
        arousal = feats.get("arousal_score", 0.0)
        question = feats.get("question_score", 0.0)
        
        # Lower hook threshold when payoff and arousal are strong
        if payoff >= 0.6 and arousal >= 0.45:
            hook_threshold = 0.06  # τ_cond for high-quality clips
        elif payoff >= 0.4 and arousal >= 0.35:
            hook_threshold = 0.10  # τ_cond for good clips
        else:
            hook_threshold = 0.15  # τ for standard clips
        
        hook = feats.get("hook_score", 0.0)
        weak_hook = hook < hook_threshold
        has_early_question = question >= 0.50
        no_payoff = payoff < 0.25
        ad_like = feats.get("_ad_flag", False) or (feats.get("_ad_penalty", 0.0) >= 0.3)
        
        # Hard fail ONLY when (weak hook) AND (ad_like OR no_payoff)
        if hook < 0.08 and (ad_like or no_payoff):
            if ad_like and no_payoff:
                return "ad_like;weak_hook;no_payoff"
            elif ad_like:
                return "ad_like;weak_hook"
            else:
                return "weak_hook;no_payoff"
        
        # Soft penalties for weak hooks (never force 0.05 unless above condition)
        if hook < 0.08:
            return "weak_hook_very_soft"  # ×0.65 penalty
        elif hook < hook_threshold:
            return "weak_hook_mild_soft"  # ×0.90 penalty
        
        if no_payoff and not has_early_question:
            return "no_payoff"
        
        if arousal < 0.20:  # Increased from 0.15 to allow more variation
            return "low_energy"
        
        return None

    def rank_candidates(self, segments: List[Dict], audio_file: str, top_k=5, platform: str = 'tiktok', genre: str = 'general', episode_id: str = None) -> List[Dict]:
        """Rank candidates using two-stage scoring: pre-rank + full V4 scoring"""
        scored = []
        
        # Two-stage scoring: pre-rank first, then full scoring on top candidates
        if PRERANK_ENABLED and len(segments) > 10:  # Only use pre-rank for larger sets
            logger.info(f"Using two-stage scoring: {len(segments)} segments")
            
            # Stage 1: Pre-rank with cheap features
            prerank_candidates = self.pre_rank_candidates(segments, episode_id or "unknown")
            
            # Get safety candidates (obvious bangers)
            safety_candidates = self.get_safety_candidates(segments)
            
            # Combine and deduplicate
            safety_added = [s for s in safety_candidates if s not in prerank_candidates]
            all_candidates = prerank_candidates + safety_added
            
            logger.info(f"Pre-rank: {len(prerank_candidates)} candidates")
            logger.info(f"Safety net: {len(safety_candidates)} bangers found, {len(safety_added)} new ones added")
            logger.info(f"Combined: {len(all_candidates)} total candidates")
            
            # Apply stratified selection if enabled
            if STRATIFY_ENABLED:
                target_count = min(len(all_candidates), max(TOP_K_MIN, math.ceil(TOP_K_RATIO * len(segments))))
                final_candidates = self.pick_stratified(all_candidates, target_count)
            else:
                final_candidates = all_candidates
                
            logger.info(f"Two-stage: {len(segments)} -> {len(final_candidates)} candidates for full scoring")
            segments_to_score = final_candidates
        else:
            logger.info(f"Using full scoring: {len(segments)} segments")
            segments_to_score = segments
        
        # preload audio once for arousal (big speed-up)
        try:
            import librosa
            y_sr = librosa.load(audio_file, sr=None)
        except Exception:
            y_sr = None

        # Stage 2: Full V4 scoring on selected candidates
        write_progress(episode_id or "unknown", "scoring:full", 20, f"Full scoring {len(segments_to_score)} candidates...")
        
        for i, seg in enumerate(segments_to_score):
            # Progress update every 10 items
            if i % 10 == 0:
                progress_pct = 20 + int((i / len(segments_to_score)) * 70)  # 20-90%
                write_progress(episode_id or "unknown", "scoring:full", progress_pct, f"Scoring candidate {i+1}/{len(segments_to_score)}...")
            
            # Use enhanced V4 feature computation with all Phase 1-3 features
            feats = None
            try:
                feats = _compute_features(
                    segment=seg,
                    audio_file=audio_file,
                    y_sr=y_sr,
                    platform=platform,
                    genre=genre,
                    segments=segments_to_score,  # give enhanced fn episode context
                )
                _log_feature_coverage(logger, feats)
                seg["features"] = feats
                
                # Debug: log what features we got
                logger.info(f"Enhanced V4 Features computed: {list(feats.keys())}")
                logger.info(f"Feature values: hook={feats.get('hook_score', 0):.3f}, arousal={feats.get('arousal_score', 0):.3f}, payoff={feats.get('payoff_score', 0):.3f}")
                logger.info(f"Hook reasons: {feats.get('hook_reasons', 'none')}")
                logger.info(f"Payoff type: {feats.get('payoff_type', 'none')}")
                logger.info(f"Moment type: {feats.get('type', 'general')}")
            except Exception as e:
                logger.warning(f"Enhanced features failed for segment {seg.get('id', i)}: {e}; falling back to v4")
                try:
                    feats = compute_features_v4(seg, audio_file, y_sr=y_sr, platform=platform, genre=genre)
                    _log_feature_coverage(logger, feats)
                    seg["features"] = feats
                    
                    # Debug: log what features we got
                    logger.info(f"Fallback V4 Features computed: {list(feats.keys())}")
                    logger.info(f"Feature values: hook={feats.get('hook_score', 0):.3f}, arousal={feats.get('arousal_score', 0):.3f}, payoff={feats.get('payoff_score', 0):.3f}")
                except Exception as e2:
                    logger.exception(f"Both enhanced and fallback feature compute failed for segment {seg.get('id', i)}: {e2}")
                    continue
            
            # Skip if no features were computed
            if feats is None:
                continue
            
            # Get raw score using V4 multi-path scoring with genre awareness
            current_weights = get_clip_weights()
            logger.info(f"Available weights: {list(current_weights.keys())}")
            
            # Use V4 multi-path scoring with genre awareness
            scoring_result = score_segment_v4(feats, current_weights, genre=genre, platform=platform)
            raw_score = scoring_result["final_score"]
            winning_path = scoring_result["winning_path"]
            path_scores = scoring_result["path_scores"]
            synergy_multiplier = scoring_result["synergy_multiplier"]
            bonuses_applied = scoring_result["bonuses_applied"]
            bonus_reasons = scoring_result["bonus_reasons"]
            
            seg["winning_path"] = winning_path
            seg["path_scores"] = path_scores
            seg["synergy_multiplier"] = synergy_multiplier
            seg["bonuses_applied"] = bonuses_applied
            seg["bonus_reasons"] = bonus_reasons
            
            logger.info(f"V4 Scoring: {raw_score:.3f}, Path: {winning_path}, Synergy: {synergy_multiplier:.3f}, Bonuses: {bonuses_applied:.3f}")
            logger.info(f"Bonus reasons: {bonus_reasons}")
            
            # Apply ad penalty AFTER scoring, BEFORE calibration
            ad_flag = feats.get("_ad_flag", False)
            ad_pen = feats.get("_ad_penalty", 0.0)
            ad_reason = feats.get("_ad_reason", "none")
            
            # Apply ad penalty to score
            raw_score *= (1.0 - ad_pen)
            seg["ad_penalty"] = float(ad_pen)
            seg["_ad_flag"] = ad_flag
            seg["_ad_reason"] = ad_reason
            
            # Features are already clamped in compute_features for ads
            # This ensures consistency between feature calculation and scoring
            
            # Soft quality gate with genre-specific penalties
            reason = fails_quality(feats)
            seg["discard_reason"] = reason
            
            if reason:
                # Get genre-specific quality gate config
                from config_loader import get_config
                config = get_config()
                quality_config = config.get("quality_gate", {})
                global_config = quality_config.get("global", {})
                genre_config = quality_config.get(genre, {})
                
                # Apply genre-specific penalties
                if reason == "weak_hook_very_soft":
                    penalty_mult = global_config.get("weak_hook_very_soft", 0.65)
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"very_weak_hook_{penalty_mult}"
                    logger.info(f"Clip got soft penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif reason == "weak_hook_mild_soft":
                    penalty_mult = global_config.get("weak_hook_mild_soft", 0.90)
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"mild_weak_hook_{penalty_mult}"
                    logger.info(f"Clip got mild penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif reason == "no_payoff":
                    # Check for insight/question fallback
                    has_insight = feats.get("insight_score", 0.0) >= 0.70
                    has_question = feats.get("question_score", 0.0) >= 0.50
                    
                    if has_insight or has_question:
                        # Use genre-specific soften for insights/questions
                        penalty_mult = genre_config.get("no_payoff_soften", global_config.get("no_payoff_soften", 0.85))
                        raw_score *= penalty_mult
                        seg["soft_penalty"] = f"no_payoff_with_insight_{penalty_mult}"
                        logger.info(f"Clip softened for insight/question: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                    else:
                        # Standard no_payoff penalty
                        penalty_mult = genre_config.get("no_payoff_soften", global_config.get("no_payoff_soften", 0.70))
                        raw_score *= penalty_mult
                        seg["soft_penalty"] = f"no_payoff_{penalty_mult}"
                        logger.info(f"Clip got no_payoff penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif reason == "low_energy":
                    penalty_mult = global_config.get("low_energy_soften", 0.75)
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"low_energy_{penalty_mult}"
                    logger.info(f"Clip got low_energy penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                elif "ad_like" in reason:
                    # Hard floor only for ads
                    logger.info(f"Clip failed quality gate: {reason}, setting score to 0.05")
                    raw_score = 0.05
                else:
                    # Fallback for other reasons
                    penalty_mult = 0.70
                    raw_score *= penalty_mult
                    seg["soft_penalty"] = f"other_{penalty_mult}"
                    logger.info(f"Clip got other penalty: {reason}, applying {penalty_mult}x, final score: {raw_score:.3f}")
                seg["raw_score"] = raw_score
            else:
                seg["raw_score"] = raw_score
                logger.info(f"Clip passed quality gate, final score: {raw_score:.3f}")
            
            # Gentle energy dampener for low-arousal clips (optional)
            arousal = feats.get("arousal_score", 0.0)
            if arousal < 0.38:
                energy_dampener = 0.95  # small nudge for low energy
                raw_score *= energy_dampener
                seg["energy_dampener"] = f"low_energy_{energy_dampener}"
                logger.info(f"Applied energy dampener: {energy_dampener}x for arousal={arousal:.3f}, final score: {raw_score:.3f}")
            
            # Calibrate score for better user experience with wider range
            calibrated_score = self._calibrate_score_for_ui(raw_score)
            seg["display_score"] = calibrated_score["score"]
            seg["clip_score_100"] = calibrated_score["score"]  # Set clip_score_100 for frontend
            seg["confidence"] = calibrated_score["confidence"]
            seg["confidence_color"] = calibrated_score["color"]
            
            seg["score"] = seg["raw_score"]  # For backward compatibility
            
            # Use V4 explanation system
            seg["explain"] = explain_segment_v4(feats, genre=genre)
            
            # Add V4 viral score and platform recommendations
            length_s = float(seg["end"] - seg["start"])
            seg["viral"] = viral_potential_v4(feats, length_s, "general")
            
            # -------- DEBUG JSON LINE (feed to logs for analysis) --------
            if logger.isEnabledFor(logging.INFO):
                dbg = {
                    "episode_id": episode_id or "unknown",
                    "clip_id": seg.get("id", f"clip_{i}"),
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                    "duration_sec": round(length_s, 3),
                    "final_score": round(raw_score, 4),
                    "hook": round(feats.get("hook_score", 0.0), 4),
                    "arousal": round(feats.get("arousal_score", 0.0), 4),
                    "payoff": round(feats.get("payoff_score", 0.0), 4),
                    "emotion": round(feats.get("emotion_score", 0.0), 4),
                    "info_density": round(feats.get("info_density", 0.0), 4),
                    "question_score": round(feats.get("question_score", 0.0), 4),
                    "platform_length": round(feats.get("platform_length_score_v2", feats.get("platform_len_match", 0.0)), 4),
                    "insight_score": round(feats.get("insight_score", 0.0), 4),
                    "loopability": round(feats.get("loopability", 0.0), 4),
                    "_synergy_boost": round(feats.get("_synergy_boost", 0.0), 4),
                    "_bonuses": {
                        "insight": round(feats.get("insight_score", 0.0) * 0.15 if feats.get("insight_score", 0.0) >= 0.7 else 0.0, 4),
                        "question": round(feats.get("question_score", 0.0) * 0.10 if feats.get("question_score", 0.0) >= 0.6 else 0.0, 4),
                        "platform_fit": round(0.02 if feats.get("platform_length_score_v2", feats.get("platform_len_match", 0.0)) >= 0.90 else 0.0, 4)
                    },
                    "_penalties": {
                        "ad_penalty": round(feats.get("_ad_penalty", 0.0), 4),
                        "soft_penalty": seg.get("soft_penalty", "none"),
                        "energy_dampener": seg.get("energy_dampener", "none")
                    },
                    "winning_path": winning_path,
                    "path_scores": {k: round(v, 4) for k, v in path_scores.items()},
                    "title": seg.get("title", ""),
                    "transcript_preview": (seg.get("text", "") or "").strip()[:240]
                }
                # Single-line JSON: greppable and easy to paste
                logger.info("CLIP_SCORING %s", json.dumps(dbg, ensure_ascii=False))
            
            scored.append(seg)
        
        # Completion progress update
        write_progress(episode_id or "unknown", "scoring:completed", 100, f"Scoring complete: {len(scored)} candidates processed")
        
        # Sort by raw_score descending but prefer non-discarded
        scored_sorted = sorted(scored, key=lambda s: (s.get("discard_reason") is not None, -s["raw_score"]))
        return scored_sorted[:top_k]
    
    def _calibrate_score_for_ui(self, raw_score: float) -> dict:
        """Transform raw 0-1 scores into user-friendly 45-95 range with confidence bands"""
        
        # Wider range for better differentiation: 0.1 (weak) to 0.8 (excellent)
        min_score = 0.1   # weakest passing clip
        max_score = 0.8   # best observed clip
        
        if raw_score < min_score:
            return {"score": 40, "confidence": "⚠️ Fair", "color": "text-red-600"}
        else:
            # Map 0.1 → 45%, 0.8 → 95%, cap at 100%
            normalized = (raw_score - min_score) / (max_score - min_score)
            calibrated = min(int(normalized * 50 + 45), 100)  # Cap at 100%
            
            # Confidence bands
            if calibrated >= 90:
                confidence = "🔥 Exceptional"
                color = "text-green-600"
            elif calibrated >= 80:
                confidence = "⭐ Premium"
                color = "text-blue-600"
            elif calibrated >= 70:
                confidence = "✅ Strong"
                color = "text-yellow-600"
            elif calibrated >= 60:
                confidence = "👍 Good"
                color = "text-orange-600"
            else:
                confidence = "⚠️ Fair"
                color = "text-red-600"
            
            return {
                "score": calibrated,
                "confidence": confidence,
                "color": color
            }

    def _find_topic_boundaries(self, transcript: List[TranscriptSegment]) -> List[int]:
        """Find natural topic transitions in conversation"""
        boundaries = [0]
        
        topic_markers = [
            "so anyway", "moving on", "let me tell you", 
            "here's the thing", "the point is", "basically",
            "now", "so", "well", "okay", "right",
            "let's talk about", "speaking of", "by the way",
            "the thing is", "what I mean is", "the bottom line",
            "to be honest", "frankly", "honestly",
            "you know what", "here's what", "this is why",
            "the reason is", "because", "since",
            "first of all", "secondly", "finally",
            "in conclusion", "to sum up", "overall"
        ]
        
        for i, seg in enumerate(transcript):
            text = seg.text.lower()
            # Check for topic transition markers
            if any(marker in text for marker in topic_markers):
                boundaries.append(i)
        
        # Add the end if not already included
        if boundaries[-1] != len(transcript) - 1:
            boundaries.append(len(transcript) - 1)
        
        return boundaries

    def _transcript_to_segments_aligned(self, transcript: List[TranscriptSegment]) -> List[Dict]:
        """Create topic-based segments that are more self-contained"""
        segments = []
        
        if not transcript:
            return segments
        
        # Find topic boundaries
        boundaries = self._find_topic_boundaries(transcript)
        logger.info(f"Found {len(boundaries)} topic boundaries")
        
        # Create segments from topic boundaries with non-overlapping logic
        current_end = 0.0  # Track where the last segment ended
        
        # Sort boundaries by start time to ensure proper ordering
        boundary_times = [(transcript[boundaries[i]].start, boundaries[i]) for i in range(len(boundaries))]
        boundary_times.sort()
        
        for i in range(len(boundary_times) - 1):
            start_idx = boundary_times[i][1]
            end_idx = boundary_times[i + 1][1]
            
            # Get the segment
            segment_start = transcript[start_idx].start
            segment_end = transcript[end_idx].end
            
            # Only create segment if it doesn't overlap with previous ones
            if segment_start >= current_end:
                # Calculate duration
                duration = segment_end - segment_start
                
                # More flexible duration range to get more segments
                if 12 <= duration <= 60:  # Expanded range for more variety
                    segment_text = " ".join([t.text for t in transcript[start_idx:end_idx+1]])
                    
                    # Less restrictive self-contained check for topic-based segments
                    if self._is_self_contained_topic(segment_text):
                        segment = {
                            "start": segment_start,
                            "end": segment_end,
                            "text": segment_text
                        }
                        segments.append(segment)
                        current_end = segment_end  # Update the end point
                        logger.info(f"Created topic segment: {segment_start:.1f}s - {segment_end:.1f}s ({duration:.1f}s)")
        
        # Final check: ensure no overlaps exist by filtering overlapping segments
        if len(segments) > 1:
            non_overlapping = [segments[0]]
            for seg in segments[1:]:
                if seg["start"] >= non_overlapping[-1]["end"]:
                    non_overlapping.append(seg)
            segments = non_overlapping
            logger.info(f"Filtered to {len(segments)} non-overlapping topic segments")
        
        # If we didn't find enough topic-based segments, fall back to the original method
        if len(segments) < 3:  # Lowered threshold from 5 to 3
            logger.info("Topic-based segmentation produced too few segments, falling back to window-based approach")
            return self._transcript_to_segments(transcript, platform='tiktok')
        
        logger.info(f"Created {len(segments)} topic-based segments")
        return segments

    def _is_self_contained(self, text: str) -> bool:
        """Check if a segment is self-contained and doesn't require prior context"""
        text_lower = text.lower()
        
        # Context-dependent patterns that indicate the segment needs prior context
        context_patterns = [
            r'\b(he|she|they|it|this|that|these|those)\b.*\b(too|also|as well|either)\b',
            r'\b(so|therefore|thus|hence|consequently)\b.*\b(that\'s why|this is why)\b',
            r'\b(you need to|you should|you have to)\b.*\b(this way|that way)\b',
            r'\b(they feel|they think|they know)\b.*\b(the care|the love|the attention)\b',
            r'\b(that\'s why|this is why)\b.*\b(we\'re|you\'re|they\'re)\b.*\b(successful|good|better)\b',
            r'\b(oh yeah|yeah|right)\b.*\b(and how|and what|and why)\b',
            r'\b(you know|you see|you understand)\b.*\b(what I mean|what I\'m saying)\b',
            r'\b(like I said|as I mentioned|as we discussed)\b',
            r'\b(going back to|returning to|coming back to)\b',
            r'\b(in the same way|similarly|likewise)\b'
        ]
        
        # Check for context-dependent patterns
        for pattern in context_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for incomplete thoughts
        incomplete_patterns = [
            r'\b(and|but|so|because|since|while|although)\s*$',
            r'\b(is|are|was|were|has|have|had)\s*$',
            r'\b(you|he|she|they|it|this|that)\s*$',
            r'\.{3}$',  # Ends with ellipsis
            r'\?$'  # Ends with question (often indicates incomplete thought)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for minimum meaningful content
        words = text.split()
        if len(words) < 8:  # Too short to be meaningful
            return False
        
        # Check for proper sentence structure
        if not re.search(r'[.!?]$', text):  # Doesn't end with proper punctuation
            return False
        
        return True

    def _is_self_contained_topic(self, text: str) -> bool:
        """Less restrictive self-contained check for topic-based segments"""
        text_lower = text.lower()
        
        # Only check for the most obvious context-dependent patterns
        obvious_context_patterns = [
            r'\b(like I said|as I mentioned|as we discussed)\b',
            r'\b(going back to|returning to|coming back to)\b',
            r'\b(in the same way|similarly|likewise)\b',
            r'\b(oh yeah|yeah|right)\b.*\b(and how|and what|and why)\b'
        ]
        
        # Check for obvious context-dependent patterns
        for pattern in obvious_context_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for incomplete thoughts (less strict)
        incomplete_patterns = [
            r'\b(and|but|so|because|since|while|although)\s*$',
            r'\.{3}$'  # Ends with ellipsis
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for minimum meaningful content (reduced from 8 to 6)
        words = text.split()
        if len(words) < 6:  # Less strict minimum
            return False
        
        # Check for proper sentence structure (more flexible)
        if not re.search(r'[.!?]$', text):  # Doesn't end with proper punctuation
            # Allow segments that end with incomplete thoughts if they're long enough
            if len(words) < 15:  # Only require punctuation for short segments
                return False
        
        return True

    def _is_repetitive_content(self, text: str) -> bool:
        """Check if content is repetitive/filler (like 'Yeah. Yeah. Yeah...')"""
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        # Check for repetitive patterns
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 50% of the time, it's repetitive
        max_count = max(word_counts.values())
        if max_count > len(words) * 0.5:
            return True
        
        # Check for specific repetitive patterns
        repetitive_patterns = [
            "yeah yeah yeah",
            "uh uh uh",
            "um um um",
            "so so so",
            "and and and",
            "the the the"
        ]
        
        text_lower = text.lower()
        for pattern in repetitive_patterns:
            if pattern in text_lower:
                return True
        
        return False

    def _is_intro_content(self, text: str) -> bool:
        """Check if content is intro/greeting material"""
        text_lower = text.lower().strip()
        
        intro_patterns = [
            r"^(yo|hey|hi|hello|what's up|how's it going|good morning|good afternoon|good evening)",
            r"^(it's|this is) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"^(i'm|my name is) \w+",
            r"^(welcome to|thanks for|thank you for)",
            r"^(hope you|hope everyone)",
            r"^(let's get|let's start|let's begin)",
            r"^(today we're|today i'm|today let's)"
        ]
        
        import re
        for pattern in intro_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False

    def _is_natural_ending(self, text: str) -> bool:
        """Check if text ends naturally"""
        text = text.strip()
        # Good endings
        if text.endswith(('.', '!', '?')):
            # Check it's not mid-sentence
            last_words = text.split()[-3:]
            if not any(w in ['the', 'a', 'an', 'to', 'of', 'and', 'but'] for w in last_words):
                return True
        return False

    def _transcript_to_segments(self, transcript: List[TranscriptSegment], genre: str = 'general', platform: str = 'tiktok') -> List[Dict]:
        """Create dynamic segments based on natural content boundaries and platform optimization"""
        try:
            # Convert TranscriptSegment objects to dicts
            transcript_dicts = []
            for seg in transcript:
                if hasattr(seg, '__dict__'):
                    transcript_dicts.append({
                        'text': seg.text,
                        'start': seg.start,
                        'end': seg.end
                    })
                else:
                    transcript_dicts.append(seg)
            
            logger.info(f"Starting with {len(transcript_dicts)} raw transcript segments")
            
            # Filter out intro/filler content first
            filtered_segments = []
            for seg in transcript_dicts:
                if not (self._is_intro_content(seg['text']) or self._is_repetitive_content(seg['text'])):
                    filtered_segments.append(seg)
            
            logger.info(f"Filtered to {len(filtered_segments)} non-intro segments")
            
            # Use dynamic segmentation based on natural content boundaries
            from services.secret_sauce_pkg.features import create_dynamic_segments
            
            dynamic_segments = create_dynamic_segments(filtered_segments, platform)
            
            logger.info(f"Created {len(dynamic_segments)} dynamic segments based on natural boundaries")
            
            # Log segment details for debugging
            for i, seg in enumerate(dynamic_segments[:3]):
                duration = seg['end'] - seg['start']
                boundary_type = seg.get('boundary_type', 'unknown')
                confidence = seg.get('confidence', 0.0)
                logger.info(f"Dynamic Segment {i+1}: {duration:.1f}s ({boundary_type}, conf={confidence:.2f}) - {seg['text'][:60]}...")
            
            return dynamic_segments
            
        except Exception as e:
            logger.error(f"Dynamic segmentation failed: {e}, falling back to window-based")
            return self._window_based_segments(transcript_dicts, window=30, step=20)
    
    def _window_based_segments(self, transcript: List[Dict], window: float = 25, step: float = 15) -> List[Dict]:
        """Fallback window-based segmentation with larger steps"""
        segments = []
        
        # Find total duration
        if transcript:
            total_duration = max(seg.get('end', 0) for seg in transcript)
        else:
            return segments
        
        for start_time in range(0, int(total_duration), int(step)):
            end_time = min(start_time + window, total_duration)
            
            if end_time - start_time < 15:  # Minimum duration (increased from 12 to 15)
                continue
            
            segment_text = self._get_transcript_segment_text(transcript, start_time, end_time)
            
            if len(segment_text.strip()) > 20:  # Minimum text length
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': segment_text,
                    'type': 'window'
            })
        
        return segments
    
    def _get_transcript_segment_text(self, transcript: List[Dict], start_time: float, end_time: float) -> str:
        """Get transcript text for a specific time window from dict format"""
        segment_texts = []
        
        for seg in transcript:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check if this segment overlaps with our window
            if seg_start < end_time and seg_end > start_time:
                # Calculate overlap
                overlap_start = max(seg_start, start_time)
                overlap_end = min(seg_end, end_time)
                
                # If significant overlap, include the text
                if overlap_end - overlap_start > 0.5:  # At least 0.5 seconds overlap
                    segment_texts.append(seg.get('text', ''))
        
        return ' '.join(segment_texts)
    
    def _filter_overlapping_candidates(self, candidates: List[Dict], min_gap: float = 15.0) -> List[Dict]:
        """Remove overlapping candidates, keeping highest scoring ones"""
        if not candidates:
            return candidates
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        filtered = []
        
        for candidate in sorted_candidates:
            # Check if this overlaps with any already selected
            overlaps = False
            for selected in filtered:
                # Check for time overlap
                if (candidate['start'] < selected['end'] and 
                    candidate['end'] > selected['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(candidate)
        
        return filtered
    
    def _filter_low_quality(self, candidates: List[Dict], min_score: int = 40) -> List[Dict]:
        """Filter out low-quality candidates"""
        filtered = []
        
        for candidate in candidates:
            # Skip if score too low
            if candidate.get('display_score', 0) < min_score:
                continue
                
            # Skip if text too short or repetitive
            text = candidate.get('text', '')
            if len(text.split()) < 20:
                continue
                
            # Skip if starts mid-sentence (basic check)
            if text and text[0].islower():
                continue
                
            filtered.append(candidate)
        
        # If we filtered too many, relax criteria
        if len(filtered) < 3 and len(candidates) >= 3:
            return candidates[:5]  # Return top 5 regardless
        
        return filtered

    def _get_transcript_segment(self, transcript: List[TranscriptSegment], 
                               start_time: float, end_time: float) -> str:
        """Get transcript text for a specific time window"""
        segment_texts = []
        
        for segment in transcript:
            # More precise overlap check
            if segment.end > start_time and segment.start < end_time:
                segment_texts.append(segment.text)
        
        return ' '.join(segment_texts)

    def select_best_moments(self, moment_scores: List[MomentScore], 
                           target_count: int = 3, min_duration: int = 12, 
                           max_duration: int = 30) -> List[MomentScore]:
        """Select the best moments ensuring diversity and constraints"""
        if not moment_scores:
            return []
        
        # Filter by duration constraints
        valid_moments = [
            m for m in moment_scores 
            if min_duration <= m.duration <= max_duration
        ]
        
        if not valid_moments:
            return []
        
        # Sort by score
        valid_moments.sort(key=lambda x: x.total_score, reverse=True)
        
        # Select moments ensuring temporal and content diversity
        selected_moments = []
        min_gap = 45  # Increased minimum gap between clips
        
        for moment in valid_moments:
            if len(selected_moments) >= target_count:
                break
            
            # Check if this moment is far enough from already selected ones
            is_diverse = True
            for selected in selected_moments:
                time_gap = abs(moment.start_time - selected.start_time)
                if time_gap < min_gap:
                    is_diverse = False
                    break
                
                # Also check for content similarity (basic check)
                if hasattr(moment, 'text') and hasattr(selected, 'text'):
                    moment_words = set(moment.text.lower().split()[:10])  # First 10 words
                    selected_words = set(selected.text.lower().split()[:10])
                    if len(moment_words.intersection(selected_words)) > 5:  # If more than 5 words overlap
                        is_diverse = False
                        break
            
            if is_diverse:
                selected_moments.append(moment)
        
        return selected_moments[:target_count]

    async def get_candidates(self, episode_id: str, platform: str = "tiktok_reels", genre: str = None) -> List[Dict]:
        """Get AI-scored clip candidates for an episode with platform/genre optimization using the complete pipeline"""
        try:
            from services.secret_sauce_pkg import (
                find_viral_clips_enhanced, resolve_platform, detect_podcast_genre
            )

            # Get episode
            episode = await self.episode_service.get_episode(episode_id)
            if not episode or not episode.transcript:
                logger.error(f"Episode {episode_id} not found or has no transcript")
                return []

            # Resolve platform
            backend_platform = resolve_platform(platform)

            # Detect or resolve genre first
            detected_genre = detect_podcast_genre(episode.transcript)
            if genre:
                # User explicitly selected a genre
                final_genre = genre
            else:
                # Use auto-detected genre
                final_genre = detected_genre

            logger.info(f"Using genre: {final_genre} (detected: {detected_genre}, user_selected: {genre})")

            # Convert transcript to segments using intelligent moment detection
            segments = self._transcript_to_segments(episode.transcript, genre=final_genre, platform=platform)

            # Build EOS index early to determine fallback mode
            from services.secret_sauce_pkg.features import build_eos_index
            episode_words = getattr(episode, 'words', None) if episode else None
            episode_raw_text = getattr(episode, 'raw_text', None) if episode else None
            eos_times, word_end_times, eos_source = build_eos_index(segments, episode_words, episode_raw_text)
            
            # One-time fallback mode decision per episode (freeze this value)
            eos_density = len(eos_times) / max(len(word_end_times), 1)
            episode_fallback_mode = eos_density < 0.020
            logger.info(f"EOS_UNIFIED: count={len(eos_times)}, src={eos_source}, density={eos_density:.3f}, fallback={episode_fallback_mode}")

            # Use the enhanced viral clips pipeline with all Phase 1-3 improvements
            logger.info(f"Using enhanced viral clips pipeline with {len(segments)} segments")
            viral_result = find_viral_clips_enhanced(segments, episode.audio_path, genre=final_genre, platform=backend_platform, fallback_mode=episode_fallback_mode, effective_eos_times=eos_times, effective_word_end_times=word_end_times, eos_source=eos_source)
            
            # Debug: log ft_status from enhanced pipeline
            if viral_result and "clips" in viral_result:
                ft_statuses = [c.get("ft_status", "missing") for c in viral_result["clips"]]
                logger.info(f"Enhanced pipeline ft_statuses: {ft_statuses}")
            
            # Use authoritative fallback mode from enhanced pipeline
            episode_fallback_mode = authoritative_fallback_mode(viral_result)
            
            # Optional emergency override while testing:
            import os
            env_force = os.getenv("FT_FORCE_FALLBACK")
            if env_force is not None:
                episode_fallback_mode = (env_force == "1")
            
            logger.info(f"GATE_MODE: fallback={episode_fallback_mode} (authoritative from enhanced pipeline)")
            
            if "error" in viral_result:
                logger.error(f"Enhanced viral clips pipeline failed: {viral_result['error']}")
                return []
            
            clips = viral_result.get('clips', [])
            logger.info(f"Found {len(clips)} viral clips using enhanced pipeline")
            
            # Convert episode transcript to full text for display
            full_episode_transcript = ""
            if episode.transcript:
                full_episode_transcript = " ".join([seg.text for seg in episode.transcript if hasattr(seg, 'text') and seg.text])
            
            # EOS index already built above for fallback mode determination
            
            # Sanity assertions
            assert eos_times is not None, "EOS times should never be None"
            if len(eos_times) == 0:
                logger.warning("EOS_FALLBACK_ONLY: No EOS markers found, using fallback mode")
            
            # Add fallback flag to each candidate for hook honesty cap
            for clip in clips:
                clip_meta = clip.get("meta") or {}
                clip_meta["is_fallback"] = bool(episode_fallback_mode)
                clip["meta"] = clip_meta
            
            # Convert to candidate format with title generation and grades
            candidates = format_candidates(clips, final_genre, backend_platform, episode_id, full_episode_transcript, episode)
            logger.info(f"Formatted {len(candidates)} candidates")
            
            # PRE-NMS: Log top 10 before NMS with proper tie-breaking
            def sort_key(c):
                return (
                    round(c.get('final_score', 0), 3),
                    round(c.get('platform_length_score_v2', 0.0), 3),
                    round(c.get('end', 0) - c.get('start', 0), 2)
                )
            
            for c in sorted(candidates, key=sort_key, reverse=True)[:10]:
                logger.info("PRE-NMS: fs=%.2f dur=%.1f pl_v2=%.2f text='%s'",
                    c.get("final_score", 0), c.get("end", 0) - c.get("start", 0), 
                    c.get("platform_length_score_v2", 0.0),
                    (c.get("text", "")[:50]).replace("\n", " ")
                )
            
            # A) Prove what each top clip scored (one-line instrumentation)
            for i, c in enumerate(sorted(candidates, key=sort_key, reverse=True)[:10]):
                # Enhanced diagnostics
                text = c.get("text", "")
                is_q = text.strip().endswith("?")
                has_payoff = c.get("payoff_score", 0.0) >= 0.20
                looks_ad = c.get("ad_penalty", 0.0) >= 0.3
                words = c.get("text_length", 0)
                final_score = c.get("final_score", 0)
                
                # Verification asserts
                if is_q and words < 12 and c.get("payoff_score", 0) < 0.20 and not c.get("_has_answer", False):
                    assert final_score <= 0.55 + 1e-6, f"Question cap failed: {final_score} '{text[:60]}'"
                
                logger.info(
                    "Top #%d: score=%.2f w=%s hook=%.2f arous=%.2f payoff=%.2f info=%.2f ql=%.2f pl_v2=%.2f ad_pen=%.2f q=%s payoff_ok=%s ad=%s caps=%s text='%s'",
                    i+1,
                    final_score,
                    words,
                    c.get("hook_score", 0),
                    c.get("arousal_score", 0),
                    c.get("payoff_score", 0),
                    c.get("info_density", 0),
                    c.get("q_list_score", 0),
                    c.get("platform_length_score_v2", 0),
                    c.get("ad_penalty", 0),
                    is_q,
                    has_payoff,
                    looks_ad,
                    c.get("flags", {}).get("caps_applied", []),
                    (text[:80]).replace("\n", " ")
                )
            
            # Debug: Log candidate scores
            for i, candidate in enumerate(candidates[:3]):  # Log first 3 candidates
                logger.info(f"Candidate {i}: display_score={candidate.get('display_score', 'MISSING')}, text_length={len(candidate.get('text', '').split())}")
            
            # Collapse consecutive question runs (keep only the strongest)
            from services.secret_sauce_pkg.scoring_utils import collapse_question_runs
            candidates = collapse_question_runs(candidates)
            logger.info(f"After question collapse: {len(candidates)} candidates")
            
            # Filter overlapping candidates
            filtered_candidates = filter_overlapping_candidates(candidates)
            logger.info(f"After overlap filtering: {len(filtered_candidates)} candidates")
            
            # Normalize candidates before any gates
            filtered_candidates = [normalize_ft_status(c) for c in filtered_candidates]
            
            # Duration/np.float64 guard (prevents weird types later)
            try:
                import numpy as np
                for c in filtered_candidates:
                    if isinstance(c.get("duration"), np.floating):
                        c["duration"] = float(c["duration"])
            except Exception:
                pass
            
            # Apply quality filtering with safety net
            quality_filtered = filter_low_quality(filtered_candidates, min_score=15)
            logger.info(f"QUALITY_FILTER: kept={len(quality_filtered)} of {len(filtered_candidates)}")
            
            # CANDIDATES_BEFORE_SAFETY: Log before safety pass
            logger.info(f"CANDIDATES_BEFORE_SAFETY: {len(quality_filtered)}")
            
            # --- Safety / shrink / finish-thought pass (updates ft_status) ---
            updated = 0
            for c in quality_filtered:
                meta = run_safety_or_shrink(c, eos_times, word_end_times, backend_platform, episode_fallback_mode)
                if isinstance(meta, dict) and meta.get("status"):
                    c["ft_status"] = meta["status"]                 # 'finished' | 'sparse_finished' | 'unresolved' | 'extended'
                    c["ft_meta"] = {**c.get("ft_meta", {}), **meta}
                    updated += 1
            logger.info(f"SAFETY_PASS: updated {updated}/{len(quality_filtered)}")
            
            # --- Apply quality gate with auto-relaxation ---
            quality_filtered = apply_quality_gate(quality_filtered, mode="strict" if not episode_fallback_mode else "fallback")
            
            # Apply adaptive gate if we have too few clips
            if len(quality_filtered) < 3:
                quality_filtered = adaptive_gate(quality_filtered, min_count=3)
            
            # Apply temporal deduplication to remove near-duplicate clips
            iou = float(os.getenv("DEDUP_IOU_THRESHOLD", "0.85"))
            target_len = float(os.getenv("DEDUP_TARGET_LEN", "18.0"))
            before = len(quality_filtered)
            quality_filtered = _dedup_by_time_iou(quality_filtered, iou_thresh=iou, target_len=target_len)
            after = len(quality_filtered)
            logger.info(f"DEDUP_BY_TIME: {before} → {after} kept ({before-after} removed), thresh={iou:.2f}")
            
            # Apply text-based deduplication to remove semantic duplicates
            text_thresh = float(os.getenv("DEDUP_TEXT_THRESHOLD", "0.90"))
            before_text = len(quality_filtered)
            quality_filtered = _dedup_by_text(quality_filtered, text_thresh=text_thresh)
            after_text = len(quality_filtered)
            logger.info(f"DEDUP_BY_TEXT: {before_text} → {after_text} kept ({before_text-after_text} removed), thresh={text_thresh:.2f}")
            
            # Apply duration clamping to prevent overlong clips
            platform_max_sec = float(os.getenv("PLATFORM_MAX_SEC", "30.0"))
            clamped_count = 0
            for c in quality_filtered:
                old_dur = c.get("end", 0) - c.get("start", 0)
                if old_dur > platform_max_sec:
                    c["end"] = c.get("start", 0) + platform_max_sec
                    new_dur = c["end"] - c.get("start", 0)
                    logger.info(f"CLIP_DURATION_CLAMP: {c.get('id', 'unknown')} {old_dur:.2f}s → {new_dur:.2f}s")
                    clamped_count += 1
            if clamped_count > 0:
                logger.info(f"CLIP_DURATION_CLAMP: {clamped_count} clips clamped to {platform_max_sec}s max")
            
            # POST-NMS: Log final durations (cast to float to avoid numpy dtype issues)
            safe_durs = [float(round(c.get("end", 0) - c.get("start", 0), 1)) for c in quality_filtered]
            logger.info("POST-NMS: durs=%s", safe_durs)
            
            # Diversity check: warn if all candidates have same duration
            durs = [round(c.get("end", 0) - c.get("start", 0), 1) for c in quality_filtered]
            if len(quality_filtered) >= 6 and len(set(durs)) < 2:
                logger.warning("DIVERSITY: all candidates have same duration=%s; favoring longer pl_v2 ties", durs[0] if durs else "unknown")
            
            # Final diversity guard: ensure at least one long platform-fit clip
            finals = quality_filtered[:10]
            if len(finals) >= 4 and max(c.get('end', 0) - c.get('start', 0) for c in finals) < 16.0:
                def sort_key(c):
                    return (round(c.get('final_score', 0), 3),
                            round(c.get('platform_length_score_v2', 0.0), 3),
                            round(c.get('end', 0) - c.get('start', 0), 2))
                
                # Find best long candidate from all candidates
                long_best = next((c for c in sorted(candidates, key=sort_key, reverse=True)
                                if (c.get('end', 0) - c.get('start', 0)) >= 18.0 
                                and c.get('platform_length_score_v2', 0) >= 0.80), None)
                if long_best and long_best not in finals:
                    long_best["protected"] = True  # Mark as protected
                    finals[-1] = long_best  # Replace last slot
                    finals.sort(key=sort_key, reverse=True)  # Re-sort
                    logger.info(f"DIVERSITY GUARD: promoted long clip dur={long_best.get('end', 0) - long_best.get('start', 0):.1f}s pl_v2={long_best.get('platform_length_score_v2', 0):.2f}")
            
            # Enforce final soft floor after all gates
            finals = enforce_soft_floor(finals, min_count=3)
            
            # ---- FINISHED-THOUGHT BACKSTOP -----------------------------------------
            # If finals contain zero "finished_thought", promote the best finished clip
            # from prior survivors by swapping out the weakest final (bounded swap).
            try:
                has_finished = any(bool(c.get("finished_thought")) for c in finals)
                if not has_finished:
                    pool = [c for c in candidates if c.get("finished_thought")]
                    if pool:
                        pool.sort(key=lambda c: float(c.get("final_score", c.get("score", 0.0))), reverse=True)
                        candidate = pool[0]
                        if finals:
                            finals.sort(key=lambda c: float(c.get("final_score", c.get("score", 0.0))))
                            weakest = finals[0]
                            if float(candidate.get("final_score", candidate.get("score", 0.0))) >= \
                               float(weakest.get("final_score", weakest.get("score", 0.0))) - 0.05:
                                finals[0] = candidate
                                logger.info("QUALITY_BACKSTOP: promoted finished_thought clip %s", candidate.get("id"))
            except Exception as e:
                logger.error("QUALITY_BACKSTOP_ERROR: %s", e)
            # -----------------------------------------------------------------------
            
            # Mark all selected clips with protected status
            for c in finals:
                c["protected"] = c.get("protected", False)
            
            # Stable final ordering: protected clips first, then by score
            finals.sort(key=lambda c: (c.get("protected", False), c.get("final_score", 0)), reverse=True)
            
            # Apply temporal deduplication to remove near-duplicate clips
            if finals:
                iou_thresh = float(os.getenv("DEDUP_IOU_THRESHOLD", "0.85"))
                platform_target = 18.0  # Default target duration
                before_count = len(finals)
                finals = _dedup_by_time_iou(finals, iou_thresh=iou_thresh, target_len=platform_target)
                after_count = len(finals)
                if before_count != after_count:
                    logger.info(f"DEDUP_BY_TIME: {before_count} → {after_count} kept ({before_count - after_count} removed), thresh={iou_thresh}")
            
            # Debug: Log final saved set with durations and protected flags
            final_info = []
            for c in finals:
                dur = round(c.get('end', 0) - c.get('start', 0), 1)
                prot = "prot=True" if c.get("protected", False) else ""
                final_info.append(f"{dur}s {prot}".strip())
            logger.info(f"FINAL_SAVE: n={len(finals)} :: [{', '.join(final_info)}]")
            
            # Safety pass already completed before quality gate
            candidates_after = len(finals)
            
            # Pre-save logging
            logger.info(f"PRE_SAVE: final_count={candidates_after} ids={[c.get('id', 'unknown') for c in finals]}")
            
            # Remove hard assert on zero; degrade gracefully
            if candidates_after == 0:
                logger.error("NO_CANDIDATES: returning empty set after gating; consider enabling FT_SOFT_RELAX_ON_ZERO=1")
                return []
            
            # Add ranking and default clip selection
            ranked_clips = rank_clips(finals)
            default_clip_id = choose_default_clip(ranked_clips)
            
            # Add metadata to each clip
            for clip in ranked_clips:
                clip["protected_long"] = clip.get("protected", False)
                clip["duration"] = round(clip.get('end', 0) - clip.get('start', 0), 2)
                clip["pl_v2"] = clip.get("platform_length_score_v2", 0.0)
                clip["finished_thought"] = clip.get("text", "").strip().endswith(('.', '!', '?'))
                
                # Ensure ft_status is set (fallback to finished_thought if not set)
                if "ft_status" not in clip:
                    if clip.get("finished_thought", 0) == 1:
                        clip["ft_status"] = "finished"
                    else:
                        clip["ft_status"] = "unresolved"
            
            # Final sanity check before save
            assert len(ranked_clips) > 0, "Should have at least one clip"
            
            # Calculate longest clip duration
            longest_dur = max((c.get("end", 0) - c.get("start", 0)) for c in ranked_clips) if ranked_clips else 0.0
            
            # Use the authoritative FT summary returned from the enhanced pipeline when available
            if viral_result and "ft" in viral_result:
                ft_data = viral_result["ft"]
                logger.info("FT_SUMMARY: finished=%d sparse=%d total=%d ratio_strict=%.2f ratio_sparse_ok=%.2f longest=%.1fs",
                            ft_data["finished"], ft_data["sparse_finished"], ft_data["total"],
                            ft_data["ratio_strict"], ft_data["ratio_sparse_ok"], longest_dur)
            else:
                # Fallback to local calculation if no FT data available
                finished_count = sum(1 for c in ranked_clips if c.get('ft_status') == 'finished')
                sparse_count = sum(1 for c in ranked_clips if c.get('ft_status') == 'sparse_finished')
                total = len(ranked_clips)
                ratio_strict = finished_count / max(total, 1)
                ratio_sparse_ok = (finished_count + sparse_count) / max(total, 1) if episode_fallback_mode else ratio_strict
                
                logger.info("FT_SUMMARY: finished=%d sparse=%d total=%d ratio_strict=%.2f ratio_sparse_ok=%.2f longest=%.1fs",
                            finished_count, sparse_count, total, ratio_strict, ratio_sparse_ok, longest_dur)
            
            # Telemetry that can't disagree with itself
            wc, ec, dens = len(word_end_times), len(eos_times), len(eos_times)/max(len(word_end_times), 1)
            logger.info(f"TELEMETRY: word_count={wc}, eos_count={ec}, eos_density={dens:.3f}")
            logger.info(f"TELEMETRY: fallback_mode={episode_fallback_mode} (authoritative)")
            
            # Candidate-level FT summary (post-gates)
            cand_total = len(ranked_clips)
            cand_finished = sum(1 for c in ranked_clips if c.get("ft_status") == "finished")
            cand_sparse = sum(1 for c in ranked_clips if c.get("ft_status") == "sparse_finished")
            cand_ratio_strict = (cand_finished / cand_total) if cand_total else 0.0
            cand_ratio_sparse_ok = ((cand_finished + (cand_sparse if episode_fallback_mode else 0)) / cand_total) if cand_total else 0.0
            
            logger.info("FT_SUMMARY_CANDIDATES: total=%d finished=%d sparse=%d ratio_strict=%.2f ratio_sparse_ok=%.2f",
                        cand_total, cand_finished, cand_sparse, cand_ratio_strict, cand_ratio_sparse_ok)
            logger.info(f"POST_SAFETY: kept={len(ranked_clips)} ids={[c.get('id', 'unknown') for c in ranked_clips]}")
            
            # Candidate-based warning (use env tunable threshold; defaults are sensible)
            warn_min_strict = float(os.getenv("FT_WARN_MIN_CAND_RATIO_STRICT", "0.60"))
            warn_min_fallback = float(os.getenv("FT_WARN_MIN_CAND_RATIO_FALLBACK", "0.70"))
            if not episode_fallback_mode and cand_ratio_strict < warn_min_strict:
                logger.warning("FINISH_RATIO_LOW_CAND: %.2f < %.2f (strict)", cand_ratio_strict, warn_min_strict)
            elif episode_fallback_mode and cand_ratio_sparse_ok < warn_min_fallback:
                logger.warning("FINISH_RATIO_LOW_CAND: %.2f < %.2f (fallback)", cand_ratio_sparse_ok, warn_min_fallback)
            
            # Upstream/auth summary should be informational only now:
            if viral_result and "ft" in viral_result:
                ft_data = viral_result["ft"]
                logger.info("FT_SUMMARY_AUTH: total=%d finished=%d sparse=%d ratio_strict=%.2f ratio_sparse_ok=%.2f",
                            ft_data["total"], ft_data["finished"], ft_data["sparse_finished"], 
                            ft_data["ratio_strict"], ft_data["ratio_sparse_ok"])
            
            return ranked_clips, default_clip_id
            
        except Exception as e:
            logger.error(f"Failed to get candidates: {e}", exc_info=True)
            return [], None

async def score_episode(episode, segments):
    """
    Score an episode and return ranked clips.
    This is a standalone function for compatibility with the existing main.py usage.
    """
    try:
        # Create a ClipScoreService instance
        from services.episode_service import EpisodeService
        episode_service = EpisodeService()
        clip_service = ClipScoreService(episode_service)
        
        # Get audio path for the episode
        audio_path = episode.audio_path
        if not audio_path:
            logger.error(f"No audio path found for episode {episode.id}")
            return []
        
        # Convert segments to the format expected by the service
        transcript_segments = []
        for seg in segments:
            transcript_segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', ''),
                'confidence': seg.get('confidence', 0.0)
            })
        
        # Get candidates using the service (await the async call)
        candidates, default_clip_id = await clip_service.get_candidates(
            episode_id=episode.id,
            platform="tiktok_reels",
            genre="general"
        )
        
        # Convert candidates to the expected format
        scored_clips = []
        for candidate in candidates:
            scored_clips.append({
                'id': candidate.get('id', ''),
                'start': candidate.get('start', 0),
                'end': candidate.get('end', 0),
                'text': candidate.get('text', ''),
                'score': candidate.get('score', 0),
                'confidence': candidate.get('confidence', ''),
                'genre': candidate.get('genre', 'general'),
                'platform': candidate.get('platform', 'tiktok'),
                'features': candidate.get('features', {}),
                'meta': candidate.get('meta', {})
            })
        
        return scored_clips
        
    except Exception as e:
        logger.error(f"Failed to score episode {episode.id}: {e}", exc_info=True)
        return []
