"""
Episode Service - Manages episode uploads and transcription
"""

import os
import logging
import asyncio
from typing import List, Optional
from datetime import datetime
import whisper
import aiofiles
import ffmpeg
from pydub import AudioSegment

from models import Episode, TranscriptSegment
from config import settings

logger = logging.getLogger(__name__)

class EpisodeService:
    """Service for managing podcast episodes"""
    
    def __init__(self):
        self.episodes: dict = {}  # In-memory storage for MVP
        self.whisper_model = None
        self._init_whisper()
    
    def _init_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            # Use base model for stability
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model 'base' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    async def create_episode(self, episode_id: str, file, original_filename: str) -> Episode:
        """Create a new episode from uploaded file"""
        try:
            # Generate unique filename
            file_ext = os.path.splitext(original_filename)[1]
            filename = f"{episode_id}{file_ext}"
            file_path = os.path.join(settings.UPLOAD_DIR, filename)
            
            # Save uploaded file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Create episode record
            episode = Episode(
                id=episode_id,
                filename=filename,
                original_name=original_filename,
                size=file_size,
                status="uploading"
            )
            
            # Store episode
            self.episodes[episode_id] = episode
            
            logger.info(f"Created episode {episode_id}: {original_filename}")
            return episode
            
        except Exception as e:
            logger.error(f"Failed to create episode {episode_id}: {e}")
            raise
    
    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID"""
        return self.episodes.get(episode_id)
    
    async def get_all_episodes(self) -> List[Episode]:
        """Get all episodes"""
        return list(self.episodes.values())

    async def list_episodes(self) -> List[Episode]:
        """Get all episodes (alias for get_all_episodes)"""
        return await self.get_all_episodes()
    
    async def process_episode(self, episode_id: str):
        """Process episode: transcribe and analyze"""
        try:
            episode = self.episodes.get(episode_id)
            if not episode:
                logger.error(f"Episode {episode_id} not found")
                return
            
            # Update status
            episode.status = "processing"
            episode.uploaded_at = datetime.utcnow()
            
            logger.info(f"Processing episode {episode_id}")
            
            # Get file path
            file_path = os.path.join(settings.UPLOAD_DIR, episode.filename)
            
            # Convert to audio if needed
            audio_path = await self._ensure_audio_format(file_path)
            
            # Store the audio path in the episode
            episode.audio_path = audio_path
            
            # Get duration
            duration = await self._get_audio_duration(audio_path)
            episode.duration = duration
            
            # Transcribe audio
            transcript = await self._transcribe_audio(audio_path)
            
            # Store transcript in episode
            episode.transcript = transcript
            logger.info(f"Transcription completed for {episode_id}: {len(transcript)} segments")
            
            # Update episode status
            episode.status = "completed"
            episode.processed_at = datetime.utcnow()
            
            logger.info(f"Episode {episode_id} processing completed")
            
        except Exception as e:
            logger.error(f"Episode processing failed for {episode_id}: {e}")
            episode = self.episodes.get(episode_id)
            if episode:
                episode.status = "failed"
                episode.error = str(e)
    
    async def _ensure_audio_format(self, file_path: str) -> str:
        """Convert file to audio format if needed"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # If already audio format, return as is
            if file_ext in ['.mp3', '.wav', '.m4a']:
                return file_path
            
            # Convert video to audio
            if file_ext in ['.mp4', '.mov', '.avi']:
                audio_path = file_path.rsplit('.', 1)[0] + '.wav'
                
                logger.info(f"Converting {file_path} to audio format")
                
                # Use ffmpeg to extract audio
                stream = ffmpeg.input(file_path)
                stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ar=settings.SAMPLE_RATE)
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                return audio_path
            
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            raise
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            # Use pydub for duration
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 300.0  # Default to 5 minutes
    
    async def _transcribe_audio(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_model:
                raise RuntimeError("Whisper model not available")
            
            logger.info(f"Starting transcription for {audio_path}")
            
            # Run Whisper transcription with timeout
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Run transcription in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self.whisper_model.transcribe(
                    audio_path,
                    language=settings.WHISPER_LANGUAGE,
                    word_timestamps=True,
                    fp16=False,  # Force FP32 to avoid warnings
                    verbose=False  # Reduce logging
                )
            )
            
            # Convert to our format
            transcript = []
            for segment in result['segments']:
                words = []
                if 'words' in segment:
                    for word_info in segment['words']:
                        words.append({
                            'word': word_info['word'],
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('confidence', 0.0)
                        })
                
                transcript_segment = TranscriptSegment(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment['text'].strip(),
                    confidence=segment.get('confidence', 0.0),
                    words=words
                )
                transcript.append(transcript_segment)
            
            logger.info(f"Transcription completed: {len(transcript)} segments")
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            logger.info("Attempting fallback transcription...")
            
            # Fallback: try without word timestamps
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(
                        audio_path,
                        language=settings.WHISPER_LANGUAGE,
                        word_timestamps=False,  # Disable word timestamps
                        fp16=False,
                        verbose=False
                    )
                )
                
                # Convert to our format (without word-level details)
                transcript = []
                for segment in result['segments']:
                    transcript_segment = TranscriptSegment(
                        start=segment['start'],
                        end=segment['end'],
                        text=segment['text'].strip(),
                        confidence=segment.get('confidence', 0.0),
                        words=[]  # Empty words list for fallback
                    )
                    transcript.append(transcript_segment)
                
                logger.info(f"Fallback transcription completed: {len(transcript)} segments")
                return transcript
                
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {fallback_error}")
                raise RuntimeError(f"Transcription failed: {e}. Fallback failed: {fallback_error}")
    
    def check_storage(self) -> bool:
        """Check if storage is accessible"""
        try:
            # Check if upload and output directories are writable
            test_file = os.path.join(settings.UPLOAD_DIR, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return False
    
    async def delete_episode(self, episode_id: str) -> bool:
        """Delete episode and associated files"""
        try:
            episode = self.episodes.get(episode_id)
            if not episode:
                return False
            
            # Remove files
            file_path = os.path.join(settings.UPLOAD_DIR, episode.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from memory
            del self.episodes[episode_id]
            
            logger.info(f"Episode {episode_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete episode {episode_id}: {e}")
            return False
    
    async def cleanup_old_episodes(self, max_age_hours: int = 24):
        """Clean up old episodes to save storage"""
        try:
            current_time = datetime.utcnow()
            episodes_to_delete = []
            
            for episode_id, episode in self.episodes.items():
                if episode.uploaded_at:
                    age_hours = (current_time - episode.uploaded_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        episodes_to_delete.append(episode_id)
            
            for episode_id in episodes_to_delete:
                await self.delete_episode(episode_id)
            
            if episodes_to_delete:
                logger.info(f"Cleaned up {len(episodes_to_delete)} old episodes")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
