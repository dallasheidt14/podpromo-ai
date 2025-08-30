"""
Loop Service - Creates seamless loop endings with crossfades for social media platforms.
"""

import os
import subprocess
import shlex
from typing import Dict, Optional

class LoopService:
    """Service for creating seamless loop endings"""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        self.fade_duration = 0.12  # 120ms fade for seamless looping
    
    def create_loop_seam(self, input_file: str, output_file: str) -> Dict:
        """
        Create a seamless loop by crossfading the end into the beginning.
        
        Args:
            input_file: Path to input video file
            output_file: Path for output looped video
        
        Returns:
            Dict with success status and output path
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Get video duration
            duration = self._get_video_duration(input_file)
            if duration is None:
                return {"success": False, "error": "Could not determine video duration"}
            
            # Create loop seam with crossfade
            success = self._apply_loop_seam(input_file, output_file, duration)
            
            if success:
                return {
                    "success": True,
                    "output": output_file,
                    "duration": duration,
                    "fade_duration": self.fade_duration
                }
            else:
                return {"success": False, "error": "Failed to create loop seam"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_video_duration(self, input_file: str) -> Optional[float]:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", input_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None
    
    def _apply_loop_seam(self, input_file: str, output_file: str, duration: float) -> bool:
        """
        Apply loop seam using ffmpeg with crossfade effect.
        
        The technique:
        1. Split video into two parts: first 120ms and last 120ms
        2. Crossfade the last 120ms into the first 120ms
        3. Concatenate: [main content] + [crossfade transition]
        """
        try:
            # Calculate timing
            fade_duration = self.fade_duration
            main_end = duration - fade_duration
            
            # Create complex filter for loop seam
            filter_complex = (
                f"[0:v]split[v1][v2]; "
                f"[0:a]asplit[a1][a2]; "
                f"[v1]trim=0:{fade_duration},setpts=PTS-STARTPTS[first_v]; "
                f"[v2]trim={main_end}:{duration},setpts=PTS-STARTPTS[last_v]; "
                f"[a1]atrim=0:{fade_duration},asetpts=PTS-STARTPTS[first_a]; "
                f"[a2]atrim={main_end}:{duration},asetpts=PTS-STARTPTS[last_a]; "
                f"[last_v][first_v]xfade=transition=fade:duration={fade_duration}:offset=0[v_loop]; "
                f"[last_a][first_a]acrossfade=d={fade_duration}[a_loop]; "
                f"[v_loop][a_loop]concat=n=1:v=1:a=1[outv][outa]"
            )
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",  # Overwrite output
                "-i", input_file,
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                "-shortest",  # Ensure output duration matches shortest stream
                output_file
            ]
            
            # Execute ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output file was created
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False
        except Exception as e:
            print(f"Loop seam error: {e}")
            return False
    
    def create_simple_loop(self, input_file: str, output_file: str) -> Dict:
        """
        Create a simpler loop by trimming the last frame and duplicating the first frame.
        This is a fallback if the complex crossfade fails.
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Get video duration
            duration = self._get_video_duration(input_file)
            if duration is None:
                return {"success": False, "error": "Could not determine video duration"}
            
            # Simple approach: trim last frame and duplicate first frame
            filter_complex = (
                f"[0:v]trim=0:{duration-0.04},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration=0.04[v]; "
                f"[0:a]atrim=0:{duration-0.04},asetpts=PTS-STARTPTS,"
                f"apad=whole_dur={duration}[a]"
            )
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "160k",
                output_file
            ]
            
            # Execute ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output file was created
            success = os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
            return {
                "success": success,
                "output": output_file if success else None,
                "duration": duration,
                "method": "simple_loop"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
