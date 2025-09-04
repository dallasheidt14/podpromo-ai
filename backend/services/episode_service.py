"""
Episode Service - Manages episode uploads and transcription
"""

import os
import logging
import asyncio
import json
import time
from typing import List, Optional, Dict
from datetime import datetime
import whisper
import aiofiles
import ffmpeg
from pydub import AudioSegment

from models import Episode, TranscriptSegment
from config.settings import UPLOAD_DIR, OUTPUT_DIR, SAMPLE_RATE, WHISPER_LANGUAGE, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, PROGRESS_TRACKER_TTL

logger = logging.getLogger(__name__)

class EnhancedProgressTracker:
    """Enhanced progress tracker with band mapping to prevent jumps"""
    
    BANDS = {
        "initializing": (0.0, 5.0),
        "audio_processing": (5.0, 15.0),
        "transcription": (15.0, 85.0),
        "feature_extraction": (85.0, 95.0),
        "scoring": (95.0, 99.0),
        "finalizing": (99.0, 100.0),
        "error": (99.0, 99.0),  # Keep at 99% to preserve progress
    }
    
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.current_stage = "initializing"
        self._cache = {
            "percentage": 0, 
            "stage": "initializing", 
            "message": "Starting...",
            "detail": ""
        }
        
    def update_stage(self, stage: str, pct: float, msg: str = None, detail: str = None):
        """Update progress with band mapping to prevent jumps"""
        lo, hi = self.BANDS.get(stage, (0, 100))
        overall = lo + (hi - lo) * min(1.0, max(0.0, pct / 100.0))
        
        self._cache = {
            "percentage": int(overall),
            "stage": stage,
            "message": msg or f"Processing {stage.replace('_', ' ')}...",
            "detail": detail or f"{pct:.0f}%" if pct > 0 else ""
        }
        
        return overall
    
    def get_progress(self) -> Dict:
        """Get current progress cache"""
        return self._cache.copy()

class EpisodeService:
    """Service for managing podcast episodes"""
    
    def __init__(self):
        self.episodes: dict = {}  # In-memory storage for MVP
        self.progress: Dict[str, Dict] = {}  # Progress tracking for each episode
        self.enhanced_trackers: Dict[str, tuple] = {}  # Enhanced progress tracking with TTL: (tracker, expires_at)
        self.whisper_model = None
        self._init_whisper()
        self._processing_episodes: set[str] = set() # Track episodes currently being processed
        
        # Create storage directory for transcripts
        self.storage_dir = os.path.join(UPLOAD_DIR, "transcripts")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing episodes on startup
        self._load_episodes()
    
    def _cleanup_expired_trackers(self):
        """Remove expired progress trackers"""
        now = time.time()
        expired_episodes = []
        
        for episode_id, (tracker, expires_at) in self.enhanced_trackers.items():
            if now > expires_at:
                expired_episodes.append(episode_id)
        
        for episode_id in expired_episodes:
            del self.enhanced_trackers[episode_id]
            logger.debug(f"Cleaned up expired tracker for episode {episode_id}")
        
        if expired_episodes:
            logger.info(f"Cleaned up {len(expired_episodes)} expired progress trackers")
    
    def _has_faster_whisper(self) -> bool:
        """Check if faster-whisper is available"""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False
    
    def _load_episodes(self):
        """Load episodes from storage on startup"""
        try:
            if not os.path.exists(self.storage_dir):
                logger.warning(f"Storage directory {self.storage_dir} does not exist")
                return
                
            files = os.listdir(self.storage_dir)
            logger.info(f"Found {len(files)} files in storage directory")
            
            for filename in files:
                if filename.endswith('.json'):
                    episode_id = filename[:-5]  # Remove .json extension
                    filepath = os.path.join(self.storage_dir, filename)
                    logger.info(f"Loading episode {episode_id} from {filepath}")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Convert ISO datetime strings back to datetime objects
                            if data.get('uploaded_at'):
                                data['uploaded_at'] = datetime.fromisoformat(data['uploaded_at'])
                            if data.get('processed_at'):
                                data['processed_at'] = datetime.fromisoformat(data['processed_at'])
                            episode = Episode(**data)
                            self.episodes[episode_id] = episode
                            logger.info(f"Successfully loaded episode {episode_id}")
                    except Exception as e:
                        logger.error(f"Failed to load episode {episode_id}: {e}")
                        
            logger.info(f"Loaded {len(self.episodes)} episodes from storage")
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")
    
    def _save_episode(self, episode: Episode):
        """Save episode to storage"""
        try:
            # Always keep episode in memory first
            self.episodes[episode.id] = episode
            logger.info(f"Episode {episode.id} kept in memory")
            
            # Try to save to file (optional)
            try:
                filepath = os.path.join(self.storage_dir, f"{episode.id}.json")
                # Convert episode to dict and handle datetime serialization
                episode_dict = episode.dict()
                # Convert datetime objects to ISO strings
                if episode_dict.get('processed_at'):
                    episode_dict['processed_at'] = episode_dict['processed_at'].isoformat()
                if episode_dict.get('uploaded_at'):
                    episode_dict['uploaded_at'] = episode_dict['uploaded_at'].isoformat()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(episode_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved episode {episode.id} to storage")
            except Exception as file_error:
                logger.warning(f"Failed to save episode {episode.id} to file: {file_error}, but kept in memory")
        except Exception as e:
            logger.error(f"Failed to save episode {episode.id}: {e}")
    
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
    
    def _episode_paths(self, episode_id: str):
        """Get the expected file paths for an episode"""
        audio = UPLOAD_DIR / f"{episode_id}.mp3"
        transcript = UPLOAD_DIR / "transcripts" / f"{episode_id}.json"
        return audio, transcript

    def _infer_progress_from_disk(self, episode_id: str):
        """Infer progress status from disk files"""
        audio, transcript = self._episode_paths(episode_id)

        if transcript.exists() and transcript.stat().st_size > 10:
            return {
                "ok": True,
                "progress": {
                    "percentage": 100,
                    "stage": "completed",
                    "message": "Transcript exists, marking completed",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "status": "completed"
            }

        if audio.exists() and audio.stat().st_size > 10:
            return {
                "ok": True,
                "progress": {
                    "percentage": 10,
                    "stage": "queued",
                    "message": "Audio found, but transcript not yet generated",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "status": "queued"
            }

        return {
            "ok": False,
            "progress": {
                "percentage": 0,
                "stage": "unknown",
                "message": "Episode not found on disk",
                "timestamp": datetime.utcnow().isoformat()
            },
            "status": "unknown"
        }

    def get_progress(self, episode_id: str) -> Optional[Dict]:
        """Get current progress for an episode (enhanced if available)"""
        # Clean up expired trackers first
        self._cleanup_expired_trackers()
        
        # Use enhanced tracker if available and not expired
        if episode_id in self.enhanced_trackers:
            tracker, expires_at = self.enhanced_trackers[episode_id]
            if time.time() <= expires_at:
                enhanced = tracker.get_progress()
                # Convert to legacy format for backward compatibility
                return {
                    "progress": enhanced["percentage"],
                    "status": enhanced["stage"],
                    "message": enhanced["message"],
                    "detail": enhanced.get("detail", "")
                }
            else:
                # Tracker expired, remove it
                del self.enhanced_trackers[episode_id]
        
        # Fallback to disk-based progress checking
        disk_progress = self._infer_progress_from_disk(episode_id)
        if disk_progress["ok"]:
            return {
                "progress": disk_progress["progress"]["percentage"],
                "status": disk_progress["progress"]["stage"],
                "message": disk_progress["progress"]["message"],
                "detail": disk_progress["progress"].get("detail", "")
            }
        
        # Final fallback to legacy progress
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
                if episode_id in self.episodes:
                    return self.episodes[episode_id]
                else:
                    raise HTTPException(status_code=409, detail="Episode is being processed")
            
            logger.info(f"Creating new episode {episode_id}: {original_filename}")
            
            # Initialize progress tracking
            self._update_progress(episode_id, "uploading", 0.0, "Starting upload...")
            
            # Generate unique filename
            file_ext = os.path.splitext(original_filename)[1]
            filename = f"{episode_id}{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            
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
            
            # Create enhanced progress tracker with TTL
            tracker = EnhancedProgressTracker(episode_id)
            expires_at = time.time() + PROGRESS_TRACKER_TTL
            self.enhanced_trackers[episode_id] = (tracker, expires_at)
            
            # Update status and progress
            episode.status = "processing"
            episode.uploaded_at = datetime.now()
            tracker.update_stage("initializing", 100, "Starting audio processing...")
            self._update_progress(episode_id, "processing", 30.0, "Starting audio processing...")
            
            logger.info(f"Processing episode {episode_id}")
            
            # Get file path
            file_path = os.path.join(UPLOAD_DIR, episode.filename)
            
            # Convert to audio if needed
            tracker.update_stage("audio_processing", 0, "Converting to audio format...")
            self._update_progress(episode_id, "converting", 35.0, "Converting to audio format...")
            audio_path = await self._ensure_audio_format(file_path)
            
            # Store the audio path in the episode
            episode.audio_path = audio_path
            tracker.update_stage("audio_processing", 50, "Audio conversion completed")
            self._update_progress(episode_id, "converted", 45.0, "Audio conversion completed")
            
            # Get duration
            tracker.update_stage("audio_processing", 75, "Analyzing audio duration...")
            self._update_progress(episode_id, "analyzing", 50.0, "Analyzing audio duration...")
            duration = await self._get_audio_duration(audio_path)
            episode.duration = duration
            tracker.update_stage("audio_processing", 100, f"Audio ready - Duration: {duration:.1f}s")
            self._update_progress(episode_id, "analyzed", 55.0, f"Audio duration: {duration:.1f}s")
            
            # Transcribe audio with enhanced progress
            if self._has_faster_whisper():
                tracker.update_stage("transcription", 0, "Starting transcription with Faster-Whisper...")
                self._update_progress(episode_id, "transcribing", 60.0, "Starting transcription with Faster-Whisper...")
                transcript = await self._transcribe_with_faster_whisper(audio_path, episode_id, tracker, duration)
            else:
                tracker.update_stage("transcription", 0, "Starting transcription with Whisper...")
                self._update_progress(episode_id, "transcribing", 60.0, "Starting transcription with Whisper...")
                transcript = await self._transcribe_audio_with_progress(audio_path, episode_id, tracker)
            
            # Store transcript in episode
            episode.transcript = transcript
            tracker.update_stage("transcription", 100, f"Transcription completed: {len(transcript)} segments")
            self._update_progress(episode_id, "transcribed", 85.0, f"Transcription completed: {len(transcript)} segments")
            
            # Save episode with transcript
            self._save_episode(episode)
            
            # Generate clips after transcription is complete
            tracker.update_stage("scoring", 0, "Finding viral clips...")
            self._update_progress(episode_id, "scoring", 90.0, "Finding viral clips...")
            
            try:
                # Import clip scoring service
                from services.clip_score import ClipScoreService
                clip_score_service = ClipScoreService(self)
                
                # Generate clips using the scoring service
                clips = await clip_score_service.get_candidates(episode_id)
                episode.clips = clips  # Store clips in episode
                tracker.update_stage("scoring", 100, f"Found {len(clips)} viral clips")
                self._update_progress(episode_id, "scored", 95.0, f"Found {len(clips)} clips")
                logger.info(f"Generated {len(clips)} clips for episode {episode_id}")
            except Exception as e:
                logger.error(f"Clip generation failed: {e}")
                episode.clips = []
                tracker.update_stage("scoring", 100, "Clip generation failed, continuing...")
                self._update_progress(episode_id, "scored", 95.0, "Clip generation failed, continuing...")
            
            # Update episode status
            episode.status = "completed"
            episode.processed_at = datetime.now()
            tracker.update_stage("finalizing", 100, "Episode processing completed successfully!")
            self._update_progress(episode_id, "completed", 100.0, "Episode processing completed successfully!")
            
            # Extend TTL for completed episodes so they remain accessible
            new_expires_at = time.time() + PROGRESS_TRACKER_TTL
            self.enhanced_trackers[episode_id] = (tracker, new_expires_at)
            
            # Save final episode state
            self._save_episode(episode)
            
            logger.info(f"Episode {episode_id} processing completed")
            
        except Exception as e:
            logger.error(f"Episode processing failed for {episode_id}: {e}")
            episode = self.episodes.get(episode_id)
            if episode:
                episode.status = "failed"
                episode.error = str(e)
                # Update enhanced tracker if available
                if episode_id in self.enhanced_trackers:
                    tracker, _ = self.enhanced_trackers[episode_id]
                    tracker.update_stage("error", 99, f"Processing failed: {str(e)[:50]}")
                    # Extend TTL for failed episodes so they remain accessible
                    new_expires_at = time.time() + PROGRESS_TRACKER_TTL
                    self.enhanced_trackers[episode_id] = (tracker, new_expires_at)
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
                stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ar=SAMPLE_RATE)
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
            
            # Ensure Whisper model is loaded
            if self.whisper_model is None:
                logger.warning("Whisper model not loaded, attempting to load now...")
                self._init_whisper()
                if self.whisper_model is None:
                    raise Exception("Failed to load Whisper model")
            
            # Run transcription in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self.whisper_model.transcribe(
                    audio_path,
                    language=WHISPER_LANGUAGE,
                    word_timestamps=True,  # Enable word timestamps for better accuracy
                    fp16=False,  # Force FP32 to avoid warnings
                    verbose=False,  # Disable verbose logging for performance
                    beam_size=5,  # Better quality decoding
                    best_of=5,    # Better quality decoding
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Multiple temperatures for better results
                    condition_on_previous_text=True,  # Better context
                    initial_prompt="This is a podcast episode with clear speech and natural conversation."  # Help Whisper understand context
                )
            )
            
            # Update progress during processing
            self._update_progress(episode_id, "processing", 75.0, "Processing transcription results...")
            
            # Convert to our format
            transcript = []
            for i, segment in enumerate(result['segments']):
                words = []
                if 'words' in segment:
                    for word_info in segment['words']:
                        words.append({
                            'word': word_info['word'],
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('confidence', 0.0)
                        })
                
                # No transcript logging - just progress updates
                
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
            logger.info("Attempting fallback transcription with simpler settings...")
            
            # Fallback with simpler settings
            try:
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.whisper_model.transcribe(
                        audio_path,
                        language=WHISPER_LANGUAGE,
                        word_timestamps=False,
                        fp16=False,
                        verbose=True,
                        beam_size=1,
                        best_of=1,
                        temperature=0.0,
                        condition_on_previous_text=False
                    )
                )
                
                # Convert to our format
                transcript = []
                for i, segment in enumerate(result['segments']):
                    # No transcript logging - just progress updates
                    
                    transcript_segment = TranscriptSegment(
                        start=segment['start'],
                        end=segment['end'],
                        text=segment['text'].strip(),
                        confidence=segment.get('confidence', 0.0),
                        words=[]
                    )
                    transcript.append(transcript_segment)
                
                logger.info(f"Fallback transcription completed: {len(transcript)} segments")
                return transcript
                
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {fallback_error}")
                raise Exception(f"Both primary and fallback transcription failed: {e}, {fallback_error}")
    
    async def _transcribe_audio_with_progress(self, audio_path: str, episode_id: str, tracker: EnhancedProgressTracker) -> List[TranscriptSegment]:
        """Enhanced transcription with better progress tracking"""
        try:
            # Lazy load Whisper model if not available
            if not self.whisper_model:
                tracker.update_stage("transcription", 5, "Loading Whisper model...")
                try:
                    self.whisper_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
                except Exception as load_error:
                    logger.error(f"Failed to load Whisper model: {load_error}")
                    raise RuntimeError(f"Whisper model not available: {load_error}")
            
            logger.info(f"Starting enhanced transcription for {audio_path}")
            tracker.update_stage("transcription", 10, "Starting Whisper transcription...")
            
            # Get audio duration for progress calculation
            duration = await self._get_audio_duration(audio_path)
            
            # Run Whisper transcription with timeout
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Update progress to show transcription is running
            tracker.update_stage("transcription", 20, "Whisper model processing audio...")
            
            # Run transcription in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self.whisper_model.transcribe(
                    audio_path,
                    language=WHISPER_LANGUAGE,
                    word_timestamps=True,
                    fp16=False,
                    verbose=False,
                    beam_size=5,
                    best_of=5,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    condition_on_previous_text=True,
                    initial_prompt="This is a podcast episode with clear speech and natural conversation."
                )
            )
            
            # Update progress during processing
            tracker.update_stage("transcription", 60, "Processing transcription results...")
            
            # Convert to our format with progress updates
            transcript = []
            total_segments = len(result.get("segments", []))
            
            for i, segment in enumerate(result.get("segments", [])):
                # Update progress based on segment processing
                segment_progress = 60 + (i / max(1, total_segments)) * 35  # 60-95%
                tracker.update_stage("transcription", segment_progress, 
                                   f"Processing segments... {i+1}/{total_segments}")
                
                transcript_segment = TranscriptSegment(
                    start=segment.get("start", 0.0),
                    end=segment.get("end", 0.0),
                    text=segment.get("text", "").strip()
                )
                transcript.append(transcript_segment)
            
            logger.info(f"Enhanced transcription completed: {len(transcript)} segments")
            return transcript
            
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            # Fallback to original method
            logger.info("Falling back to original transcription method...")
            return await self._transcribe_audio(audio_path, episode_id)
    
    async def _transcribe_with_faster_whisper(self, audio_path: str, episode_id: str, tracker: EnhancedProgressTracker, total_duration: float) -> List[TranscriptSegment]:
        """Transcribe using faster-whisper with streaming progress"""
        try:
            from faster_whisper import WhisperModel
            
            # Load model once
            if not hasattr(self, 'fw_model'):
                tracker.update_stage("transcription", 5, "Loading Faster-Whisper model...")
                self.fw_model = WhisperModel("base", compute_type="int8")
                logger.info("Faster-Whisper model loaded successfully")
            
            segments_list = []
            tracker.update_stage("transcription", 10, "Starting Faster-Whisper transcription...")
            
            # Stream segments
            segments, info = self.fw_model.transcribe(
                audio_path, 
                language="en",
                vad_filter=True,
                beam_size=5
            )
            
            last_pct = 10
            for seg in segments:
                segments_list.append(TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip()
                ))
                
                # Calculate progress based on audio position
                pct = min(95.0, 10 + (seg.end / max(1.0, total_duration)) * 85.0)
                
                if pct - last_pct >= 2.0:  # Update every 2%
                    detail = f"{int(seg.end)}s / {int(total_duration)}s"
                    tracker.update_stage("transcription", pct, "Transcribing audio...", detail)
                    last_pct = pct
                    await asyncio.sleep(0)  # Yield control
                    
            tracker.update_stage("transcription", 100, "Faster-Whisper transcription complete")
            logger.info(f"Faster-Whisper transcription completed: {len(segments_list)} segments")
            return segments_list
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription failed: {e}")
            # Fallback to regular transcription
            logger.info("Falling back to regular Whisper transcription...")
            return await self._transcribe_audio_with_progress(audio_path, episode_id, tracker)
    
    def check_storage(self) -> bool:
        """Check if storage is accessible"""
        try:
            return os.path.exists(UPLOAD_DIR) and os.access(UPLOAD_DIR, os.W_OK)
        except Exception as e:
            logger.warning(f"Storage check failed: {e}")
            return False
    
    async def delete_episode(self, episode_id: str) -> bool:
        """Delete episode and associated files"""
        try:
            episode = self.episodes.get(episode_id)
            if not episode:
                return False
            
            # Remove files
            file_path = os.path.join(UPLOAD_DIR, episode.filename)
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
