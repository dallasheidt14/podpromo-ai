"""
Clip Service - Handles video clip generation and rendering with enhanced features.
"""

import os
import json
import time
import subprocess
import shlex
import logging
from datetime import datetime
from typing import Dict, List, Optional
from .caption_service import CaptionService
from .loop_service import LoopService

logger = logging.getLogger(__name__)

# History tracking
HISTORY_DIR = os.path.join("backend", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)
HISTORY_LOG = os.path.join(HISTORY_DIR, "clips.jsonl")

class ClipService:
    """Service for generating and rendering video clips"""
    
    def __init__(self, episode_service, clip_score_service):
        self.episode_service = episode_service
        self.clip_score_service = clip_score_service
        self.caption_service = CaptionService()
        self.loop_service = LoopService()
        self.output_dir = "./outputs"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _log_history(self, payload: dict):
        """Log clip creation to history"""
        try:
            payload = dict(payload)
            payload["ts"] = datetime.now().isoformat() + "Z"
            with open(HISTORY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log history: {e}")
    
    def list_history(self, limit: int = 50):
        """Get clip history with optional limit"""
        try:
            items = []
            if os.path.exists(HISTORY_LOG):
                with open(HISTORY_LOG, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            pass
            items.sort(key=lambda x: x.get("ts", ""), reverse=True)
            return items[:limit]
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []
    
    async def render_clip_enhanced(
        self, 
        clip_id: str, 
        audio_path: str, 
        start_time: float, 
        end_time: float, 
        output_filename: str,
        style: str = "bold",
        captions: bool = True,
        punch_ins: bool = True,
        loop_seam: bool = False
    ) -> Dict:
        """
        Render a clip with enhanced features: captions, punch-ins, and optional loop seam.
        
        Args:
            clip_id: Unique identifier for the clip
            audio_path: Path to the source audio/video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_filename: Name for the output file
            style: Caption style (bold, clean, caption-heavy)
            captions: Whether to add captions
            punch_ins: Whether to add punch-in effects
            loop_seam: Whether to create a seamless loop
        
        Returns:
            Dict with success status and output path
        """
        try:
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Step 1: Create base clip with trim and format
            base_clip_path = self._create_base_clip(
                audio_path, start_time, end_time, clip_id
            )
            
            if not base_clip_path:
                return {"success": False, "error": "Failed to create base clip"}
            
            # Step 2: Add captions if requested
            if captions:
                captioned_path = await self._add_captions(
                    base_clip_path, start_time, end_time, style, clip_id
                )
                if captioned_path:
                    # Clean up base clip
                    if os.path.exists(base_clip_path):
                        os.remove(base_clip_path)
                    base_clip_path = captioned_path
            
            # Step 3: Add punch-in effects if requested
            if punch_ins:
                punched_path = self._add_punch_ins(
                    base_clip_path, clip_id
                )
                if punched_path:
                    # Clean up previous version
                    if os.path.exists(base_clip_path) and base_clip_path != audio_path:
                        os.remove(base_clip_path)
                    base_clip_path = punched_path
            
            # Step 4: Create loop seam if requested
            if loop_seam:
                looped_path = self._create_loop_seam(
                    base_clip_path, clip_id
                )
                if looped_path:
                    # Clean up previous version
                    if os.path.exists(base_clip_path) and base_clip_path != audio_path:
                        os.remove(base_clip_path)
                    base_clip_path = looped_path
            
            # Step 5: Finalize output
            final_path = self._finalize_clip(
                base_clip_path, output_path, clip_id
            )
            
            # Clean up intermediate files
            if os.path.exists(base_clip_path) and base_clip_path != audio_path:
                os.remove(base_clip_path)
            
            if final_path and os.path.exists(final_path):
                # Log to history
                self._log_history({
                    "clip_id": clip_id,
                    "format": "vertical",
                    "output": output_filename,
                    "duration": end_time - start_time,
                    "style": style,
                    "captions": captions,
                    "punch_ins": punch_ins,
                    "loop_seam": loop_seam,
                    "start_time": start_time,
                    "end_time": end_time,
                    "audio_path": audio_path
                })
                
                return {
                    "success": True,
                    "output": output_filename,
                    "path": final_path,
                    "duration": end_time - start_time,
                    "features": {
                        "captions": captions,
                        "punch_ins": punch_ins,
                        "loop_seam": loop_seam,
                        "style": style
                    }
                }
            else:
                return {"success": False, "error": "Failed to finalize clip"}
                
        except Exception as e:
            logger.error(f"Enhanced clip rendering failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def render_variant(
        self, 
        clip_id: str, 
        variant: str = "square", 
        style: str = "bold", 
        captions: bool = True
    ) -> Dict:
        """
        Create alternate export formats. For now: 'square' (1080x1080).
        """
        try:
            # Get clip metadata from history
            clip_meta = self._get_clip_metadata(clip_id)
            if not clip_meta:
                # Try to get from episode service if not in history
                episodes = await self.episode_service.get_all_episodes()
                if episodes:
                    latest_episode = episodes[-1]
                    if latest_episode.audio_path and os.path.exists(latest_episode.audio_path):
                        # Use the latest episode as fallback
                        src = latest_episode.audio_path
                        start_time = 0  # Default to start
                        end_time = 15   # Default to 15 seconds
                        duration = 15
                    else:
                        return {"success": False, "error": "No audio source available"}
                else:
                    return {"success": False, "error": "No episodes found"}
            else:
                # Use metadata from history
                src = clip_meta.get("audio_path")
                if not src or not os.path.exists(src):
                    return {"success": False, "error": "Source file not found"}
                
                start_time = clip_meta.get("start_time", 0)
                end_time = clip_meta.get("end_time", 0)
                duration = end_time - start_time
            
            if duration <= 0:
                duration = 15  # Default duration if invalid
            
            if duration <= 0:
                return {"success": False, "error": "Invalid clip duration"}
            
            # Generate output filename
            out_name = f"{clip_id}_{variant}.mp4"
            out_path = os.path.join(self.output_dir, out_name)
            
            # Build FFmpeg command for square format
            if variant == "square":
                # For square: center-crop/pad to 1080x1080
                vf = "scale=1080:1080:force_original_aspect_ratio=cover,crop=1080:1080"
                
                # Add captions if requested
                ass_file = None
                if captions:
                    # Generate captions for this variant
                    transcript = await self._get_clip_transcript(start_time, end_time)
                    if transcript:
                        ass_content = self.caption_service.generate_ass_captions(transcript, style)
                        ass_file = os.path.join(self.output_dir, f"captions_{clip_id}_{variant}.ass")
                        
                        with open(ass_file, 'w', encoding='utf-8') as f:
                            f.write(ass_content)
                        
                        vf += f",subtitles='{ass_file.replace(os.sep, '/')}'"
                
                # Build FFmpeg command
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-i", src,
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-c:a", "aac",
                    "-b:a", "160k",
                    "-movflags", "+faststart",
                    out_path
                ]
                
                # Run FFmpeg
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Clean up caption file
                if captions and ass_file and os.path.exists(ass_file):
                    os.remove(ass_file)
                
                # Log to history
                self._log_history({
                    "clip_id": clip_id,
                    "format": variant,
                    "output": out_name,
                    "duration": duration,
                    "style": style,
                    "captions": captions,
                    "start_time": start_time,
                    "end_time": end_time,
                    "audio_path": src
                })
                
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return {
                        "success": True,
                        "output": out_name,
                        "format": variant,
                        "path": out_path
                    }
                else:
                    return {"success": False, "error": "Failed to create square variant"}
            
            else:
                return {"success": False, "error": f"Unsupported variant: {variant}"}
                
        except Exception as e:
            logger.error(f"Variant rendering failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_clip_metadata(self, clip_id: str) -> Optional[Dict]:
        """Get clip metadata from history"""
        try:
            items = self.list_history(limit=1000)
            for item in items:
                if item.get("clip_id") == clip_id:
                    return item
            return None
        except Exception as e:
            logger.error(f"Failed to get clip metadata: {e}")
            return None
    
    def _create_base_clip(self, audio_path: str, start_time: float, end_time: float, clip_id: str) -> Optional[str]:
        """Create the base clip with proper trimming and format"""
        try:
            temp_path = os.path.join(self.output_dir, f"temp_base_{clip_id}.mp4")
            
            # Basic clip extraction with vertical format
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                return temp_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Base clip creation failed: {e}")
            return None
    
    async def _add_captions(self, video_path: str, start_time: float, end_time: float, style: str, clip_id: str) -> Optional[str]:
        """Add captions to the video"""
        try:
            # Get transcript for the clip duration
            transcript = await self._get_clip_transcript(start_time, end_time)
            if not transcript:
                logger.warning("No transcript available for captions")
                return video_path
            
            # Generate ASS captions
            ass_content = self.caption_service.generate_ass_captions(transcript, style)
            ass_file = os.path.join(self.output_dir, f"captions_{clip_id}.ass")
            
            with open(ass_file, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            # Add captions to video
            captioned_path = os.path.join(self.output_dir, f"captioned_{clip_id}.mp4")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles='{ass_file}'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "copy",
                captioned_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Clean up ASS file
            if os.path.exists(ass_file):
                os.remove(ass_file)
            
            if os.path.exists(captioned_path) and os.path.getsize(captioned_path) > 0:
                return captioned_path
            else:
                return video_path
                
        except Exception as e:
            logger.error(f"Caption addition failed: {e}")
            return video_path
    
    def _add_punch_ins(self, video_path: str, clip_id: str) -> Optional[str]:
        """Add punch-in zoom effects to the video"""
        try:
            punched_path = os.path.join(self.output_dir, f"punched_{clip_id}.mp4")
            
            # Add subtle zoom effect that increases over time
            filter_complex = (
                "zoompan=z='min(zoom+0.001,1.05)':d=1:x=iw/2-(iw/zoom/2):y=ih/2-(ih/zoom/2)"
            )
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", filter_complex,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "copy",
                punched_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(punched_path) and os.path.getsize(punched_path) > 0:
                return punched_path
            else:
                return video_path
                
        except Exception as e:
            logger.error(f"Punch-in addition failed: {e}")
            return video_path
    
    def _create_loop_seam(self, video_path: str, clip_id: str) -> Optional[str]:
        """Create a seamless loop ending"""
        try:
            looped_path = os.path.join(self.output_dir, f"looped_{clip_id}.mp4")
            
            # Use the loop service to create seamless ending
            result = self.loop_service.create_loop_seam(video_path, looped_path)
            
            if result["success"]:
                return looped_path
            else:
                # Fallback to simple loop if crossfade fails
                fallback_result = self.loop_service.create_simple_loop(video_path, looped_path)
                if fallback_result["success"]:
                    return looped_path
                else:
                    logger.warning("Loop seam creation failed, using original")
                    return video_path
                    
        except Exception as e:
            logger.error(f"Loop seam creation failed: {e}")
            return video_path
    
    def _finalize_clip(self, input_path: str, output_path: str, clip_id: str) -> Optional[str]:
        """Finalize the clip with proper encoding and metadata"""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",  # Optimize for web streaming
                "-metadata", f"title=PodPromo AI Clip {clip_id}",
                "-metadata", "artist=PodPromo AI",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Clip finalization failed: {e}")
            return None
    
    async def _get_clip_transcript(self, start_time: float, end_time: float) -> Optional[List[Dict]]:
        """Get transcript segment for the specified time range"""
        try:
            # Get the most recent episode
            episodes = await self.episode_service.list_episodes()
            if not episodes:
                return None
            
            latest_episode = episodes[-1]
            if not latest_episode.transcript:
                return None
            
            # Filter transcript for the clip duration
            clip_transcript = []
            for word in latest_episode.transcript:
                # Handle both TranscriptSegment objects and dictionaries
                if hasattr(word, 'start'):
                    word_start = word.start
                    word_end = word.end
                    word_text = word.text if hasattr(word, 'text') else str(word)
                else:
                    word_start = word.get("start", 0)
                    word_end = word.get("end", 0)
                    word_text = word.get("text", str(word))
                
                # Check if word overlaps with clip time
                if (word_start <= end_time and word_end >= start_time):
                    # Adjust timing relative to clip start
                    adjusted_word = {
                        "start": max(0, word_start - start_time),
                        "end": min(end_time - start_time, word_end - start_time),
                        "text": word_text
                    }
                    clip_transcript.append(adjusted_word)
            
            return clip_transcript
            
        except Exception as e:
            logger.error(f"Transcript extraction failed: {e}")
            return None
    
    async def render_clip(self, clip_id: str, audio_path: str, start_time: float, end_time: float, output_filename: str) -> Dict:
        """
        Legacy render method for backward compatibility.
        Use render_clip_enhanced for new features.
        """
        return await self.render_clip_enhanced(
            clip_id, audio_path, start_time, end_time, output_filename,
            style="bold", captions=True, punch_ins=False, loop_seam=False
        )
    
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def check_whisper(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def check_storage(self) -> bool:
        """Check if storage is accessible"""
        try:
            return os.access("./uploads", os.W_OK) and os.access("./outputs", os.W_OK)
        except Exception:
            return False
