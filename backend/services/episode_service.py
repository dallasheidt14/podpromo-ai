"""
Episode Service - Manages episode uploads and transcription
"""

import os
import logging
import asyncio
from typing import List, Optional, Dict
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
        self.progress: Dict[str, Dict] = {}  # Progress tracking for each episode
        self.whisper_model = None
        self._init_whisper()
        self._processing_episodes: set[str] = set() # Track episodes currently being processed
    
    def _init_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            # Use base model for stability
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model 'base' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.warning("Will attempt to load model on first use")
            self.whisper_model = None
    
    def _update_progress(self, episode_id: str, stage: str, percentage: float, message: str = ""):
        """Update progress for an episode"""
        if episode_id not in self.progress:
            self.progress[episode_id] = {}
        
        self.progress[episode_id].update({
            "stage": stage,
            "percentage": percentage,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Log progress with visual indicator like the backend logs
        progress_bar = "█" * int(percentage / 10) + "▌" * (10 - int(percentage / 10))
        logger.info(f"Episode {episode_id}: {stage} - {percentage:.1f}%|{progress_bar}| {message}")
    
    def get_progress(self, episode_id: str) -> Optional[Dict]:
        """Get current progress for an episode"""
        return self.progress.get(episode_id)
    
    async def create_episode(self, episode_id: str, file, original_filename: str) -> Episode:
        """Create a new episode from uploaded file"""
        try:
            # Check if episode already exists
            if episode_id in self.episodes:
                logger.warning(f"Episode {episode_id} already exists, skipping duplicate creation")
                return self.episodes[episode_id]
            
            # Check if we're already processing this episode
            if episode_id in self._processing_episodes:
                logger.warning(f"Episode {episode_id} is already being processed, skipping duplicate")
                return self.episodes.get(episode_id)
            
            logger.info(f"Creating new episode {episode_id}: {original_filename}")
            
            # Initialize progress tracking
            self._update_progress(episode_id, "uploading", 0.0, "Starting upload...")
            
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
            
            # Update progress
            self._update_progress(episode_id, "uploaded", 25.0, f"File uploaded ({file_size / 1024 / 1024:.1f} MB)")
            
            logger.info(f"Created episode {episode_id}: {original_filename}")
            return episode
            
        except Exception as e:
            logger.error(f"Failed to create episode {episode_id}: {e}")
            self._update_progress(episode_id, "error", 0.0, f"Upload failed: {str(e)}")
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
            # Add to processing guard
            self._processing_episodes.add(episode_id)
            
            episode = self.episodes.get(episode_id)
            if not episode:
                logger.error(f"Episode {episode_id} not found")
                return
            
            # Update status and progress
            episode.status = "processing"
            episode.uploaded_at = datetime.now()
            self._update_progress(episode_id, "processing", 30.0, "Starting audio processing...")
            
            logger.info(f"Processing episode {episode_id}")
            
            # Get file path
            file_path = os.path.join(settings.UPLOAD_DIR, episode.filename)
            
            # Convert to audio if needed
            self._update_progress(episode_id, "converting", 35.0, "Converting to audio format...")
            audio_path = await self._ensure_audio_format(file_path)
            
            # Store the audio path in the episode
            episode.audio_path = audio_path
            self._update_progress(episode_id, "converted", 45.0, "Audio conversion completed")
            
            # Get duration
            self._update_progress(episode_id, "analyzing", 50.0, "Analyzing audio duration...")
            duration = await self._get_audio_duration(audio_path)
            episode.duration = duration
            self._update_progress(episode_id, "analyzed", 55.0, f"Audio duration: {duration:.1f}s")
            
            # Transcribe audio
            self._update_progress(episode_id, "transcribing", 60.0, "Starting transcription with Whisper...")
            transcript = await self._transcribe_audio(audio_path, episode_id)
            
            # Store transcript in episode
            episode.transcript = transcript
            self._update_progress(episode_id, "transcribed", 85.0, f"Transcription completed: {len(transcript)} segments")
            
            # Update episode status
            episode.status = "completed"
            episode.processed_at = datetime.now()
            self._update_progress(episode_id, "completed", 100.0, "Episode processing completed successfully!")
            
            logger.info(f"Episode {episode_id} processing completed")
            
        except Exception as e:
            logger.error(f"Episode processing failed for {episode_id}: {e}")
            episode = self.episodes.get(episode_id)
            if episode:
                episode.status = "failed"
                episode.error = str(e)
                self._update_progress(episode_id, "error", 0.0, f"Processing failed: {str(e)}")
        finally:
            # Remove from processing guard
            self._processing_episodes.discard(episode_id)
    
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
    
    async def _transcribe_audio(self, audio_path: str, episode_id: str) -> List[TranscriptSegment]:
        """Transcribe audio using Whisper with progress updates"""
        try:
            # Lazy load Whisper model if not available
            if not self.whisper_model:
                logger.info("Whisper model not loaded, attempting to load now...")
                try:
                    self.whisper_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
                except Exception as load_error:
                    logger.error(f"Failed to load Whisper model: {load_error}")
                    raise RuntimeError(f"Whisper model not available: {load_error}")
            
            logger.info(f"Starting transcription for {audio_path}")
            
            # Run Whisper transcription with timeout
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Check if Whisper model is loaded
            if self.whisper_model is None:
                logger.error("Whisper model not loaded, attempting to load now...")
                self._init_whisper()
                if self.whisper_model is None:
                    raise Exception("Failed to load Whisper model")
            
            # Update progress to show transcription is running
            self._update_progress(episode_id, "transcribing", 65.0, "Whisper model processing audio...")
            
            # Run transcription in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self.whisper_model.transcribe(
                    audio_path,
                    language=settings.WHISPER_LANGUAGE,
                    word_timestamps=False,  # Disable word timestamps to avoid tensor issues
                    fp16=False,  # Force FP32 to avoid warnings
                    verbose=False,  # Reduce logging
                    beam_size=1,  # Faster decoding
                    best_of=1,    # Faster decoding
                    temperature=0.0  # Deterministic, faster
                )
            )
            
            # Update progress during processing
            self._update_progress(episode_id, "processing", 75.0, "Processing transcription results...")
            
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
            
            # Update progress for fallback
            self._update_progress(episode_id, "fallback", 70.0, "Primary transcription failed, trying fallback...")
            
            # Fallback: try with minimal configuration
            try:
                # Ensure model is loaded for fallback
                if self.whisper_model is None:
                    self._init_whisper()
                    if self.whisper_model is None:
                        raise Exception("Failed to load Whisper model for fallback")
                
                result = await loop.run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(
                        audio_path,
                        language=settings.WHISPER_LANGUAGE,
                        word_timestamps=False,  # Disable word timestamps
                        fp16=False,
                        verbose=False,
                        beam_size=1,
                        best_of=1,
                        temperature=0.0
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
                logger.info("Attempting final fallback without language specification...")
                
                # Final fallback: try without language specification
                try:
                    # Ensure model is loaded for final fallback
                    if self.whisper_model is None:
                        self._init_whisper()
                        if self.whisper_model is None:
                            raise Exception("Failed to load Whisper model for final fallback")
                    
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.whisper_model.transcribe(
                            audio_path,
                            word_timestamps=False,
                            fp16=False,
                            verbose=False,
                            beam_size=1,
                            best_of=1,
                            temperature=0.0
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
                    
                    logger.info(f"Final fallback transcription completed: {len(transcript)} segments")
                    return transcript
                    
                except Exception as final_error:
                    logger.error(f"Final fallback transcription failed: {final_error}")
                    raise RuntimeError(f"All transcription methods failed: {e}, {fallback_error}, {final_error}")
    
    def check_storage(self) -> bool:
        """Check if storage is accessible"""
        try:
            return os.path.exists(settings.UPLOAD_DIR) and os.access(settings.UPLOAD_DIR, os.W_OK)
        except Exception:
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
            current_time = datetime.now()
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
