"""
YouTube service for downloading and processing YouTube videos
"""

import os
import re
import uuid
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import subprocess
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self):
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        self.upload_dir.mkdir(exist_ok=True)
        
    def is_youtube_url(self, url: str) -> bool:
        """Check if the URL is a valid YouTube URL"""
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        return False
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in youtube_patterns:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video information using yt-dlp"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                return {
                    'id': video_id,
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', ''),
                }
        except Exception as e:
            logger.error(f"Failed to get video info for {video_id}: {e}")
            return None
    
    async def download_video(self, video_id: str, episode_id: str) -> Optional[Dict[str, Any]]:
        """Download YouTube video and return file path"""
        try:
            import yt_dlp
            
            # Create episode directory
            episode_dir = self.upload_dir / episode_id
            episode_dir.mkdir(exist_ok=True)
            
            # Try multiple format configurations for better compatibility
            format_configs = [
                # First try: best audio with MP3 conversion
                {
                    'format': 'bestaudio/best',
                    'outtmpl': str(episode_dir / 'audio.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                },
                # Fallback: any audio format
                {
                    'format': 'bestaudio',
                    'outtmpl': str(episode_dir / 'audio.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                },
                # Last resort: any format
                {
                    'format': 'best',
                    'outtmpl': str(episode_dir / 'audio.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                }
            ]
            
            info = None
            for i, ydl_opts in enumerate(format_configs):
                try:
                    logger.info(f"Trying download format config {i+1} for video {video_id}")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                        logger.info(f"Successfully downloaded video {video_id} with config {i+1}")
                        break
                except Exception as e:
                    logger.warning(f"Format config {i+1} failed for video {video_id}: {e}")
                    if i == len(format_configs) - 1:  # Last attempt
                        raise e
                    continue
            
            if not info:
                raise Exception("All download format configurations failed")
            
            # Find the downloaded file
            downloaded_files = list(episode_dir.glob('audio.*'))
            if not downloaded_files:
                logger.error(f"No audio file found after download for {video_id}")
                return None
            
            audio_file = downloaded_files[0]
            
            # Get video info
            video_info = {
                    'id': video_id,
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'filename': audio_file.name,
                    'file_path': str(audio_file),
                    'file_size': audio_file.stat().st_size,
            }
            
            logger.info(f"Downloaded YouTube video {video_id} to {audio_file}")
            return video_info
                
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg:
                logger.error(f"YouTube video {video_id} is blocked (403 Forbidden). This may be due to geographic restrictions, copyright protection, or YouTube's anti-bot measures.")
            elif "format is not available" in error_msg:
                logger.error(f"YouTube video {video_id} has no downloadable formats available.")
            else:
                logger.error(f"Failed to download YouTube video {video_id}: {e}")
            return None
    
    def validate_youtube_url(self, url: str) -> Dict[str, Any]:
        """Validate YouTube URL and return validation result"""
        if not url or not isinstance(url, str):
            return {"valid": False, "error": "No URL provided"}
        
        # Clean up URL
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        if not self.is_youtube_url(url):
            return {"valid": False, "error": "Invalid YouTube URL format"}
        
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"valid": False, "error": "Could not extract video ID from URL"}
        
        # Check if video exists and is accessible
        video_info = self.get_video_info(video_id)
        if not video_info:
            return {"valid": False, "error": "Video not found or not accessible"}
        
        # Check duration (max 4 hours)
        duration = video_info.get('duration', 0)
        if duration > 4 * 3600:  # 4 hours
            return {"valid": False, "error": "Video too long (max 4 hours)"}
        
        if duration < 60:  # 1 minute
            return {"valid": False, "error": "Video too short (min 1 minute)"}
        
        return {
            "valid": True,
            "video_id": video_id,
            "video_info": video_info,
            "url": url
        }

# Global instance
youtube_service = YouTubeService()
