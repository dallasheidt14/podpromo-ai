"""
A/B Test Service - Generates different hook versions for testing scroll-stopping effectiveness.
"""

import os
import subprocess
import uuid
from typing import Dict, List, Optional
from .caption_service import CaptionService
import json
from datetime import datetime

class ABTestService:
    """Service for creating A/B test versions of clip hooks"""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        self.hook_duration = 5.0  # 5 seconds for hook testing
        self.caption_service = CaptionService()
    
    def create_ab_tests(self, input_file: str, transcript: List[Dict], 
                        start_time: float, end_time: float) -> Dict:
        """
        Create A/B test versions of the first 5 seconds of a clip.
        
        Args:
            input_file: Path to input video/audio file
            transcript: Transcript with word-level timestamps
            start_time: Start time of the clip
            end_time: End time of the clip
        
        Returns:
            Dict with paths to text-first and face-first versions
        """
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate unique ID for this A/B test
            test_id = str(uuid.uuid4())[:8]
            
            # Create text-first version (freeze frame + bold hook text)
            text_first_path = self._create_text_first_version(
                input_file, transcript, start_time, test_id
            )
            
            # Create face-first version (punch-in crop + captions only)
            face_first_path = self._create_face_first_version(
                input_file, transcript, start_time, test_id
            )
            
            return {
                "success": True,
                "ab_tests": {
                    "text_first": text_first_path,
                    "face_first": face_first_path,
                    "test_id": test_id
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_text_first_version(self, input_file: str, transcript: List[Dict], 
                                 start_time: float, test_id: str) -> str:
        """
        Create text-first version: freeze frame with bold hook text immediately visible.
        This tests if the text hook alone can stop scrolling.
        """
        try:
            output_path = os.path.join(self.output_dir, f"ab_text_{test_id}.mp4")
            
            # Extract first 5 seconds
            hook_end = start_time + self.hook_duration
            
            # Get transcript for the hook portion
            hook_transcript = self._get_transcript_segment(transcript, start_time, hook_end)
            
            # Generate captions for the hook
            ass_content = self.caption_service.generate_ass_captions(hook_transcript, "caption-heavy")
            ass_file = os.path.join(self.output_dir, f"ab_text_{test_id}.ass")
            
            with open(ass_file, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            # Create text-first version with freeze frame and bold captions
            filter_complex = (
                f"[0:v]trim={start_time}:{hook_end},setpts=PTS-STARTPTS,"
                f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                f"freezeframe=n=1:r=1[v]; "
                f"[0:a]atrim={start_time}:{hook_end},asetpts=PTS-STARTPTS[a]"
            )
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-vf", f"subtitles='{ass_file}':force_style='Fontsize=60,PrimaryColour=&HFFFFFF&,Outline=2,Shadow=2,Alignment=2'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                "-t", str(self.hook_duration),
                output_path
            ]
            
            # Execute ffmpeg command
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Clean up temporary ASS file
            if os.path.exists(ass_file):
                os.remove(ass_file)
            
            return output_path
            
        except Exception as e:
            print(f"Text-first A/B test creation failed: {e}")
            return None
    
    def _create_face_first_version(self, input_file: str, transcript: List[Dict], 
                                 start_time: float, test_id: str) -> str:
        """
        Create face-first version: punch-in crop with captions only.
        This tests if visual engagement + captions can stop scrolling.
        """
        try:
            output_path = os.path.join(self.output_dir, f"ab_face_{test_id}.mp4")
            
            # Extract first 5 seconds
            hook_end = start_time + self.hook_duration
            
            # Get transcript for the hook portion
            hook_transcript = self._get_transcript_segment(transcript, start_time, hook_end)
            
            # Generate captions for the hook
            ass_content = self.caption_service.generate_ass_captions(hook_transcript, "clean")
            ass_file = os.path.join(self.output_dir, f"ab_face_{test_id}.ass")
            
            with open(ass_file, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            # Create face-first version with punch-in effect and clean captions
            filter_complex = (
                f"[0:v]trim={start_time}:{hook_end},setpts=PTS-STARTPTS,"
                f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"crop=1080:1920,"
                f"zoompan=z='min(zoom+0.02,1.1)':d=150:zoom=1[x=iw/2-(iw/zoom/2):y=ih/2-(ih/zoom/2)][v]; "
                f"[0:a]atrim={start_time}:{hook_end},asetpts=PTS-STARTPTS[a]"
            )
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-vf", f"subtitles='{ass_file}':force_style='Fontsize=42,PrimaryColour=&HFFFFFF&,Outline=0,Shadow=0,Alignment=2'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                "-t", str(self.hook_duration),
                output_path
            ]
            
            # Execute ffmpeg command
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Clean up temporary ASS file
            if os.path.exists(ass_file):
                os.remove(ass_file)
            
            return output_path
            
        except Exception as e:
            print(f"Face-first A/B test creation failed: {e}")
            return None
    
    def _get_transcript_segment(self, transcript: List[Dict], start_time: float, end_time: float) -> List[Dict]:
        """Get transcript segment for the specified time range"""
        segment = []
        for word in transcript:
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            
            # Check if word overlaps with time range
            if (word_start <= end_time and word_end >= start_time):
                # Adjust word timing relative to segment start
                adjusted_word = word.copy()
                adjusted_word["start"] = max(0, word_start - start_time)
                adjusted_word["end"] = min(end_time - start_time, word_end - start_time)
                segment.append(adjusted_word)
        
        return segment
    
    def log_ab_test_choice(self, test_id: str, choice: str, clip_id: str) -> Dict:
        """
        Log which A/B test version the user chose.
        This data feeds into the nightly weight tuner.
        
        Args:
            test_id: Unique identifier for the A/B test
            choice: "text_first" or "face_first"
            clip_id: ID of the original clip
        
        Returns:
            Dict with success status
        """
        try:
            # In a production system, this would log to a database
            # For MVP, we'll log to a simple JSON file
            log_entry = {
                "test_id": test_id,
                "choice": choice,
                "clip_id": clip_id,
                "timestamp": str(datetime.now()),
                "version": "text_first" if choice == "text_first" else "face_first"
            }
            
            # Append to log file
            log_file = os.path.join(self.output_dir, "ab_test_logs.json")
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            return {"success": True, "logged": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
