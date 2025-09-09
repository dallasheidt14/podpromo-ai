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
from typing import List, Dict, Tuple
from models import AudioFeatures, TranscriptSegment, MomentScore
from config.settings import (
    UPLOAD_DIR, OUTPUT_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    PRERANK_ENABLED, TOP_K_RATIO, TOP_K_MIN, TOP_K_MAX, STRATIFY_ENABLED, 
    SAFETY_KEEP_ENABLED, COMPARE_SCORING_MODES, PRERANK_WEIGHTS,
    DURATION_TARGET_MIN, DURATION_TARGET_MAX
)

from services.secret_sauce_pkg import compute_features_v4, score_segment_v4, explain_segment_v4, viral_potential_v4, get_clip_weights
from services.viral_moment_detector import ViralMomentDetector
from services.progress_writer import write_progress
from services.prerank import pre_rank_candidates, get_safety_candidates, pick_stratified
from services.quality_filters import fails_quality, filter_overlapping_candidates, filter_low_quality
from services.candidate_formatter import format_candidates
from config_loader import get_config

# ensure v2 is ON everywhere
def is_on(_): return True  # temporary until flags are unified

logger = logging.getLogger(__name__)

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
            
            # Use V4 feature computation with enhanced detectors and genre awareness
            feats = compute_features_v4(seg, audio_file, y_sr=y_sr, platform=platform, genre=genre)
            seg["features"] = feats
            
            # Debug: log what features we got
            logger.info(f"V4 Features computed: {list(feats.keys())}")
            logger.info(f"Feature values: hook={feats.get('hook_score', 0):.3f}, arousal={feats.get('arousal_score', 0):.3f}, payoff={feats.get('payoff_score', 0):.3f}")
            logger.info(f"Hook reasons: {feats.get('hook_reasons', 'none')}")
            logger.info(f"Payoff type: {feats.get('payoff_type', 'none')}")
            logger.info(f"Moment type: {feats.get('type', 'general')}")
            
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
                find_viral_clips, resolve_platform, detect_podcast_genre
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

            # Use the complete viral clips pipeline with ad/intro filtering and dynamic segmentation
            logger.info(f"Using complete viral clips pipeline with {len(segments)} segments")
            viral_result = find_viral_clips(segments, episode.audio_path, genre=final_genre, platform=backend_platform)
            
            if "error" in viral_result:
                logger.error(f"Viral clips pipeline failed: {viral_result['error']}")
                return []
            
            clips = viral_result.get('clips', [])
            logger.info(f"Found {len(clips)} viral clips using complete pipeline")
            
            # Convert to candidate format with title generation and grades
            candidates = format_candidates(clips, final_genre, backend_platform, episode_id)
            
            # Filter overlapping candidates
            filtered_candidates = filter_overlapping_candidates(candidates)
            
            # Apply quality filtering (temporarily lowered for debugging)
            quality_filtered = filter_low_quality(filtered_candidates, min_score=40)
            
            return quality_filtered[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Failed to get candidates: {e}", exc_info=True)
            return []
