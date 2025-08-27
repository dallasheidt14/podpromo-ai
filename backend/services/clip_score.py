"""
ClipScore Service - Glue layer between secret_sauce and the rest of the system.
This service orchestrates the clip scoring pipeline using the proprietary algorithms.
"""

import numpy as np
import logging
from typing import List, Dict
from models import AudioFeatures, TranscriptSegment, MomentScore
from config import settings

from services.secret_sauce import compute_features, score_segment, CLIP_WEIGHTS

logger = logging.getLogger(__name__)

class ClipScoreService:
    """Service for analyzing podcast episodes and scoring potential clip moments"""
    
    def __init__(self, episode_service):
        self.episode_service = episode_service

    def analyze_episode(self, audio_path: str, transcript: List[TranscriptSegment]) -> List[MomentScore]:
        """Analyze episode and return scored moments"""
        try:
            logger.info(f"Starting episode analysis for {audio_path}")
            
            # Convert transcript to segments for ranking
            segments = self._transcript_to_segments(transcript)
            
            # Rank candidates using secret sauce
            ranked_segments = self.rank_candidates(segments, audio_path, top_k=10)
            
            # Convert back to MomentScore objects
            moment_scores = []
            for seg in ranked_segments:
                moment_score = MomentScore(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    duration=seg["end"] - seg["start"],
                    hook_score=seg["features"]["hook_score"],
                    emotion_score=seg["features"]["emotion_score"],
                    prosody_score=seg["features"]["prosody_score"],
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

    def rank_candidates(self, segments: List[Dict], audio_file: str, top_k=5) -> List[Dict]:
        """Rank candidates using secret sauce scoring"""
        scored = []
        for seg in segments:
            feats = compute_features(seg, audio_file)
            seg["features"] = feats
            seg["score"] = score_segment(feats, CLIP_WEIGHTS)
            scored.append(seg)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _transcript_to_segments(self, transcript: List[TranscriptSegment]) -> List[Dict]:
        """Convert transcript to segment format expected by secret sauce"""
        segments = []
        
        # Create overlapping windows with better diversity
        window_size = 20  # seconds
        step_size = 15    # seconds (increased from 5 to reduce overlap)
        
        # Find the total duration
        if transcript:
            total_duration = max(seg.end for seg in transcript)
        else:
            total_duration = 0
        
        for start_time in np.arange(0, total_duration, step_size):
            end_time = min(start_time + window_size, total_duration)
            
            if end_time - start_time < settings.MIN_CLIP_DURATION:
                continue
            
            # Get transcript text for this window
            segment_text = self._get_transcript_segment(transcript, start_time, end_time)
            
            # Skip if text is too similar to previous segments
            if segments and len(segment_text.strip()) < 10:  # Skip very short segments
                continue
                
            segments.append({
                "start": start_time,
                "end": end_time,
                "text": segment_text
            })
        
        return segments

    def _get_transcript_segment(self, transcript: List[TranscriptSegment], 
                               start_time: float, end_time: float) -> str:
        """Get transcript text for a specific time window"""
        segment_texts = []
        
        for segment in transcript:
            # Check if segment overlaps with time window
            if (segment.start <= end_time and segment.end >= start_time):
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

    async def get_candidates(self, episode_id: str) -> List[Dict]:
        """Get AI-scored clip candidates for an episode"""
        try:
            # Get episode transcript
            episode = await self.episode_service.get_episode(episode_id)
            if not episode or not episode.transcript:
                return []
            
            # Convert transcript to segments
            segments = self._transcript_to_segments(episode.transcript)
            
            # Rank candidates using secret sauce
            candidates = self.rank_candidates(segments, episode.audio_path, top_k=5)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get candidates for episode {episode_id}: {e}")
            return []
