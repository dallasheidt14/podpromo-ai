"""
Clip Service - Manages clip generation and processing
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import ffmpeg

from models import Clip, MomentScore, ClipGenerationRequest
from config import settings
from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

logger = logging.getLogger(__name__)

class ClipService:
    """Service for generating video clips from podcast episodes"""
    
    def __init__(self, episode_service, clip_score_service):
        self.clips: dict = {}  # In-memory storage for MVP
        self.clip_score_service = clip_score_service
        self.episode_service = episode_service
        self.jobs: dict = {}  # Track generation jobs
    
    async def generate_clips(self, job_id: str, episode_id: str, target_count: int = 3,
                           min_duration: int = 12, max_duration: int = 30):
        """Generate clips from an episode"""
        try:
            logger.info(f"Starting clip generation job {job_id} for episode {episode_id}")
            
            # Update job status
            self.jobs[job_id] = {"status": "processing", "progress": 0}
            
            # Get episode
            episode = await self.episode_service.get_episode(episode_id)
            if not episode:
                raise ValueError(f"Episode {episode_id} not found")
            
            # Get episode file path
            episode_path = os.path.join(settings.UPLOAD_DIR, episode.filename)
            
            # Analyze episode to find best moments
            self.jobs[job_id]["progress"] = 20
            moment_scores = await self._analyze_episode(episode_path, episode_id)
            
            # Select best moments
            self.jobs[job_id]["progress"] = 40
            selected_moments = self.clip_score_service.select_best_moments(
                moment_scores, target_count, min_duration, max_duration
            )
            
            if not selected_moments:
                raise ValueError("No suitable moments found for clips")
            
            # Generate clips
            self.jobs[job_id]["progress"] = 60
            generated_clips = []
            
            for i, moment in enumerate(selected_moments):
                try:
                    clip = await self._generate_single_clip(
                        episode_id, moment, episode_path, i + 1
                    )
                    generated_clips.append(clip)
                    
                    # Update progress
                    progress = 60 + (i + 1) * (30 / len(selected_moments))
                    self.jobs[job_id]["progress"] = min(progress, 90)
                    
                except Exception as e:
                    logger.error(f"Failed to generate clip {i + 1}: {e}")
                    # Continue with other clips
            
            # Update job status
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            
            logger.info(f"Clip generation job {job_id} completed: {len(generated_clips)} clips")
            
        except Exception as e:
            logger.error(f"Clip generation job {job_id} failed: {e}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
    
    async def _analyze_episode(self, episode_path: str, episode_id: str) -> List[MomentScore]:
        """Analyze episode to find best moments using ClipScore algorithm"""
        try:
            # Get episode transcript from episode service
            episode = await self.episode_service.get_episode(episode_id)
            if not episode or not episode.transcript:
                raise ValueError(f"Episode {episode_id} has no transcript")
            
            # Use ClipScore service to analyze the episode
            moment_scores = self.clip_score_service.analyze_episode(episode_path, episode.transcript)
            
            logger.info(f"Found {len(moment_scores)} potential moments using ClipScore")
            return moment_scores
            
        except Exception as e:
            logger.error(f"Episode analysis failed: {e}")
            raise
    
    async def _generate_single_clip(self, episode_id: str, moment: MomentScore, 
                                  episode_path: str, clip_number: int) -> Clip:
        """Generate a single video clip"""
        try:
            # Create clip record
            clip_id = str(uuid.uuid4())
            clip = Clip(
                id=clip_id,
                episode_id=episode_id,
                start_time=moment.start_time,
                end_time=moment.end_time,
                duration=moment.duration,
                score=moment.total_score,
                title=f"Clip {clip_number}",
                description=f"Best moment from episode",
                status="generating"
            )
            
            # Store clip
            self.clips[clip_id] = clip
            
            # Generate output filename
            output_filename = f"clip_{clip_id}.mp4"
            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
            
            # Generate video clip
            await self._render_video_clip(
                episode_path, output_path, moment.start_time, moment.end_time
            )
            
            # Update clip with output path
            clip.output_path = output_path
            clip.download_url = f"/clips/{output_filename}"
            clip.status = "completed"
            clip.completed_at = datetime.utcnow()
            
            logger.info(f"Generated clip {clip_id}: {output_filename}")
            return clip
            
        except Exception as e:
            logger.error(f"Failed to generate clip: {e}")
            if clip:
                clip.status = "failed"
                clip.error = str(e)
            raise
    
    async def _render_video_clip(self, input_path: str, output_path: str, 
                                start_time: float, end_time: float):
        """Render video clip using FFmpeg"""
        try:
            duration = end_time - start_time
            
            # Create a simple video with audio
            # For MVP, we'll create a vertical video with captions
            
            # Extract audio segment
            audio_stream = (
                ffmpeg
                .input(input_path, ss=start_time, t=duration)
                .audio
                .filter('loudnorm')  # Normalize audio
                .filter('highpass', f=80)  # Remove low frequency noise
                .filter('lowpass', f=8000)  # Remove high frequency noise
            )
            
            # Create video background (solid color)
            video_stream = (
                ffmpeg
                .input('color=c=#1a1a1a:size=1080x1920:duration=' + str(duration), f='lavfi')
                .video
                .filter('fps', fps=settings.FPS)
            )
            
            # Add captions (simple text overlay)
            # In production, this would use the actual transcript
            caption_text = f"Podcast Clip\n{start_time:.0f}s - {end_time:.0f}s"
            
            # Combine audio and video
            output_stream = (
                ffmpeg
                .output(
                    video_stream,
                    audio_stream,
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    video_bitrate=settings.VIDEO_BITRATE,
                    audio_bitrate=settings.AUDIO_BITRATE,
                    preset='fast',
                    movflags='faststart'
                )
            )
            
            # Run FFmpeg
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Video clip rendered: {output_path}")
            
        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            raise
    
    async def get_clip(self, clip_id: str) -> Optional[Clip]:
        """Get clip by ID"""
        return self.clips.get(clip_id)

    async def render_clip(self, clip_id: str, audio_path: str, start_time: float, 
                         end_time: float, output_filename: str):
        """Render a single clip from audio file"""
        try:
            logger.info(f"Rendering clip {clip_id}: {start_time}s to {end_time}s")
            
            # Ensure output directory exists
            os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
            
            # Calculate duration
            duration = end_time - start_time
            
            # Use FFmpeg to extract the clip
            input_stream = ffmpeg.input(audio_path, ss=start_time, t=duration)
            
            # For MVP, create a simple video with audio
            # In production, you'd add captions, effects, etc.
            output_stream = (
                input_stream
                .output(
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    video_bitrate=settings.VIDEO_BITRATE,
                    audio_bitrate=settings.AUDIO_BITRATE,
                    preset='fast',
                    movflags='faststart',
                    # Create a simple video (black background with audio)
                    f='lavfi',
                    i='color=black:size=1080x1920:duration=' + str(duration)
                )
            )
            
            # Run FFmpeg
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Clip {clip_id} rendered successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to render clip {clip_id}: {e}")
            raise
    
    async def get_episode_clips(self, episode_id: str) -> List[Clip]:
        """Get all clips for an episode"""
        return [clip for clip in self.clips.values() if clip.episode_id == episode_id]
    
    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get job status and progress"""
        return self.jobs.get(job_id)
    
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            # Try to run ffmpeg -version
            result = ffmpeg.probe('dummy')
            return True
        except:
            try:
                # Alternative check
                import subprocess
                result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, text=True)
                return result.returncode == 0
            except:
                return False
    
    def check_whisper(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except:
            return False
    
    async def cleanup_old_clips(self, max_age_hours: int = 24):
        """Clean up old clips to save storage"""
        try:
            current_time = datetime.utcnow()
            clips_to_delete = []
            
            for clip_id, clip in self.clips.items():
                if clip.created_at:
                    age_hours = (current_time - clip.created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        clips_to_delete.append(clip_id)
            
            for clip_id in clips_to_delete:
                await self.delete_clip(clip_id)
            
            if clips_to_delete:
                logger.info(f"Cleaned up {len(clips_to_delete)} old clips")
                
        except Exception as e:
            logger.error(f"Clip cleanup failed: {e}")
    
    async def delete_clip(self, clip_id: str) -> bool:
        """Delete clip and associated files"""
        try:
            clip = self.clips.get(clip_id)
            if not clip:
                return False
            
            # Remove output file
            if clip.output_path and os.path.exists(clip.output_path):
                os.remove(clip.output_path)
            
            # Remove from memory
            del self.clips[clip_id]
            
            logger.info(f"Clip {clip_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete clip {clip_id}: {e}")
            return False
