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
from cachetools import TTLCache
import psutil
try:
    import faster_whisper
    WHISPER_AVAILABLE = True
    WHISPER_TYPE = "faster-whisper"
except ImportError:
    import whisper
    WHISPER_AVAILABLE = True
    WHISPER_TYPE = "whisper"
import aiofiles
import ffmpeg
from pydub import AudioSegment

from models import Episode, TranscriptSegment
from config.settings import UPLOAD_DIR, OUTPUT_DIR, SAMPLE_RATE, WHISPER_LANGUAGE, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, PROGRESS_TRACKER_TTL, ENABLE_ASR_V2, ASR_PROGRESS_LABEL_HQ, AUDIO_PREDECODE_PCM
from services.asr import transcribe_with_quality, ASRSettings
from services.audio_io import ensure_pcm_wav
from services.util import normalize_asr_result

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
    
    def __init__(self, maxsize: int = 200, ttl: int = 3600):
        # Auto-evict after 1h; cap memory
        self.episodes = TTLCache(maxsize=maxsize, ttl=ttl)
        self.progress: Dict[str, Dict] = {}  # Progress tracking for each episode
        self.enhanced_trackers: Dict[str, tuple] = {}  # Enhanced progress tracking with TTL: (tracker, expires_at)
        self.whisper_model = None
        self._init_whisper()
        self._processing_episodes: set[str] = set() # Track episodes currently being processed
        self._episodes_loaded = False  # Flag to track if episodes have been loaded
        self._hits = 0
        self._misses = 0
        
        # Create storage directory for transcripts
        self.storage_dir = os.path.join(UPLOAD_DIR, "transcripts")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Don't load episodes on startup - load them lazily when needed
        logger.info("EpisodeService initialized - episodes will be loaded on demand")
    
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
    
    def _ensure_episodes_loaded(self):
        """Lazy load episodes only when needed"""
        if self._episodes_loaded:
            return
            
        try:
            if not os.path.exists(self.storage_dir):
                logger.warning(f"Storage directory {self.storage_dir} does not exist")
                self._episodes_loaded = True
                return
                
            files = os.listdir(self.storage_dir)
            logger.info(f"Lazy loading {len(files)} episode files from storage")
            
            # Only load recent episodes (last 10) to avoid startup delay
            recent_files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(self.storage_dir, f)), reverse=True)[:10]
            
            for filename in recent_files:
                if filename.endswith('.json'):
                    # Skip words-only files - they're not episodes
                    if filename.endswith('_words.json'):
                        continue
                        
                    episode_id = filename[:-5]  # Remove .json extension
                    filepath = os.path.join(self.storage_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Convert ISO datetime strings back to datetime objects
                            if data.get('uploaded_at'):
                                data['uploaded_at'] = datetime.fromisoformat(data['uploaded_at'])
                            if data.get('processed_at'):
                                data['processed_at'] = datetime.fromisoformat(data['processed_at'])
                            
                            # Clean up clips data if it exists and is corrupted
                            if 'clips' in data and data['clips'] is not None:
                                if not isinstance(data['clips'], list):
                                    logger.warning(f"Corrupted clips data for episode {episode_id}, setting to None")
                                    data['clips'] = None
                                else:
                                    # Filter out non-dict items from clips
                                    cleaned_clips = []
                                    for item in data['clips']:
                                        if isinstance(item, dict):
                                            cleaned_clips.append(item)
                                    data['clips'] = cleaned_clips
                            
                            episode = Episode(**data)
                            self.episodes[episode_id] = episode
                    except Exception as e:
                        logger.error(f"Failed to load episode {episode_id}: {e}")
                        
            self._episodes_loaded = True
            logger.info(f"Lazy loaded {len(self.episodes)} recent episodes from storage")
        except Exception as e:
            logger.error(f"Failed to lazy load episodes: {e}")
            self._episodes_loaded = True  # Mark as loaded to avoid retrying
    
    def _save_episode(self, episode: Episode):
        """Save episode to storage"""
        try:
            # Always keep episode in memory first
            self.episodes[episode.id] = episode
            logger.info(f"Episode {episode.id} kept in memory")
            
            # Persist words to disk for transcript building
            self._save_words_to_disk(episode)
            
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
    
    def _save_words_to_disk(self, episode: Episode):
        """Save words data to disk for transcript building"""
        try:
            # Belt & suspenders: prevent None from ever reaching len()
            words = getattr(episode, 'words', []) or []
            words_normalized = getattr(episode, 'words_normalized', []) or []
            
            words_data = {
                "words": words,
                "words_normalized": words_normalized,
                "episode_id": episode.id,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            words_file = os.path.join(self.storage_dir, f"{episode.id}_words.json")
            with open(words_file, 'w', encoding='utf-8') as f:
                json.dump(words_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved {len(words_data['words'])} words to {words_file}")
        except Exception as e:
            logger.warning(f"Failed to save words for episode {episode.id}: {e}")
    
    def _init_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info(f"Loading {WHISPER_TYPE} model...")
            if WHISPER_TYPE == "faster-whisper":
                # Use faster-whisper with CPU for now to avoid CUDA library issues
                device = "cpu"  # Force CPU to avoid cublas64_12.dll issues
                compute_type = "int8"  # Use int8 for CPU
                self.whisper_model = faster_whisper.WhisperModel("base", device=device, compute_type=compute_type)
                logger.info(f"Faster-Whisper model 'base' loaded successfully on {device} with {compute_type}")
            else:
                # Fallback to regular whisper
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model 'base' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.warning("Will attempt to load model on first use")
            self.whisper_model = None
    
    def _should_use_fp16(self) -> bool:
        """Device-aware FP16 detection - use FP16 only if GPU supports it"""
        try:
            import torch
            if torch.cuda.is_available():
                # Check if GPU supports FP16 (compute capability >= 7.0)
                capability = torch.cuda.get_device_capability()
                if capability >= (7, 0):
                    return True
            return False
        except Exception:
            return False

    def _get_speed_preset(self) -> str:
        """Get speed preset: fast|balanced|quality"""
        return os.getenv("SPEED_PRESET", "balanced").lower()

    def _get_whisper_beam_size(self) -> int:
        """Get beam size based on speed preset"""
        preset = self._get_speed_preset()
        if preset == "fast":
            return 1
        elif preset == "quality":
            return 4
        else:  # balanced
            return 2

    def _get_whisper_best_of(self) -> int:
        """Get best_of based on speed preset"""
        preset = self._get_speed_preset()
        if preset == "fast":
            return 1
        elif preset == "quality":
            return 5
        else:  # balanced
            return 2

    def _get_whisper_temperature(self):
        """Get temperature based on speed preset"""
        preset = self._get_speed_preset()
        if preset == "fast":
            return 0.0
        elif preset == "quality":
            return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:  # balanced
            return 0.0

    def _get_whisper_condition_previous(self) -> bool:
        """Get condition_previous based on speed preset"""
        preset = self._get_speed_preset()
        if preset == "fast":
            return False
        else:  # balanced and quality
            return True
    
    def _attach_words_to_segments(self, episode_words: List[Dict], segments: List) -> None:
        """Attach episode-level words to each segment for local EOS detection"""
        from services.word_utils import normalize_word_tokens
        
        if not episode_words or not segments:
            return
        
        # Normalize all words first
        normalized_words = normalize_word_tokens(episode_words)
        if not normalized_words:
            return
            
        by_idx = 0
        n = len(normalized_words)
        
        for segment in segments:
            s_start = segment.start
            s_end = segment.end
            seg_words = []
            
            # Advance pointer to find first word in this segment
            while by_idx < n and normalized_words[by_idx].get('e', 0) <= s_start:
                by_idx += 1
            
            # Collect all words that overlap with this segment
            j = by_idx
            while j < n and normalized_words[j].get('s', 0) < s_end:
                seg_words.append(normalized_words[j])
                j += 1
            
            # Attach normalized words to segment
            segment.words = seg_words if seg_words else []
            
        logger.info(f"Attached words to {len(segments)} segments")

    def _update_progress(self, episode_id: str, stage: str, percentage: float, message: str = ""):
        """Update progress for an episode using atomic file persistence"""
        try:
            # Update in-memory cache for backward compatibility
            if episode_id not in self.progress:
                self.progress[episode_id] = {}
            
            self.progress[episode_id].update({
                "stage": stage,
                "percentage": percentage,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update persistent progress service
            from services.progress_service import progress_service
            progress_service.update_progress(
                episode_id=episode_id,
                stage=stage,
                percentage=int(percentage),
                message=message
            )
            
            # Log progress with visual indicator like the backend logs
            progress_bar = "█" * int(percentage / 10) + "▌" * (10 - int(percentage / 10))
            logger.info(f"Episode {episode_id}: {stage} - {percentage:.1f}%|{progress_bar}| {message}")
            
        except Exception as e:
            logger.error(f"Failed to update progress for {episode_id}: {e}")
            # Still log the progress even if persistence fails
            progress_bar = "█" * int(percentage / 10) + "▌" * (10 - int(percentage / 10))
            logger.info(f"Episode {episode_id}: {stage} - {percentage:.1f}%|{progress_bar}| {message}")
    
    def _mark_completed(self, episode_id: str, message: str = "Episode processing completed successfully!"):
        """Mark episode as completed using progress service"""
        try:
            from services.progress_service import progress_service
            progress_service.mark_completed(episode_id, message)
            self._update_progress(episode_id, "completed", 100.0, message)
        except Exception as e:
            logger.error(f"Failed to mark episode {episode_id} as completed: {e}")
            self._update_progress(episode_id, "completed", 100.0, message)
    
    def _mark_error(self, episode_id: str, error_message: str):
        """Mark episode as errored using progress service"""
        try:
            from services.progress_service import progress_service
            progress_service.mark_error(episode_id, error_message)
            self._update_progress(episode_id, "error", 0.0, f"Error: {error_message}")
        except Exception as e:
            logger.error(f"Failed to mark episode {episode_id} as errored: {e}")
            self._update_progress(episode_id, "error", 0.0, f"Error: {error_message}")
    
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
            self._mark_error(episode_id, f"Upload failed: {str(e)}")
            raise
    
    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID"""
        self._ensure_episodes_loaded()
        if episode_id in self.episodes:
            self._hits += 1
            return self.episodes[episode_id]
        self._misses += 1
        return None
    
    def get_or_load_episode(self, episode_id: str, with_words: bool = False) -> Optional[Episode]:
        """Get episode from cache or load from storage with optional words data"""
        # Try cache first
        episode = self.get_episode(episode_id)
        if episode and (not with_words or hasattr(episode, 'words')):
            return episode
        
        # Load from storage if not in cache
        try:
            episode_file = os.path.join(self.storage_dir, f"{episode_id}.json")
            if os.path.exists(episode_file):
                with open(episode_file, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                
                # Convert back to Episode object
                episode = Episode(**episode_data)
                self.episodes[episode_id] = episode
                
                # Load words if requested
                if with_words:
                    words_file = os.path.join(self.storage_dir, f"{episode_id}_words.json")
                    if os.path.exists(words_file):
                        with open(words_file, 'r', encoding='utf-8') as f:
                            words_data = json.load(f)
                        episode.words = words_data.get('words', [])
                        episode.words_normalized = words_data.get('words_normalized', [])
                        logger.debug(f"Loaded {len(episode.words)} words for episode {episode_id}")
                    else:
                        logger.warning(f"No words file found for episode {episode_id}")
                
                return episode
        except Exception as e:
            logger.error(f"Failed to load episode {episode_id}: {e}")
        
        return None
    
    async def get_all_episodes(self) -> List[Episode]:
        """Get all episodes"""
        self._ensure_episodes_loaded()
        return list(self.episodes.values())

    async def list_episodes(self) -> List[Episode]:
        """Get all episodes (alias for get_all_episodes)"""
        return await self.get_all_episodes()
    
    async def process_episode(self, episode_id: str):
        """Process episode: transcribe and analyze"""
        try:
            # Import progress writer
            from services.progress_writer import write_progress
            
            # Check if already processing to prevent rescoring loops
            if episode_id in self._processing_episodes:
                logger.warning(f"Episode {episode_id} is already being processed, skipping duplicate processing")
                return
            
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
            write_progress(episode_id, "converting", 1, "Preparing media...")
            
            logger.info(f"Processing episode {episode_id}")
            
            # Get file path
            file_path = os.path.join(UPLOAD_DIR, episode.filename)
            
            # Validate duration before processing (catches trimming issues)
            from services.duration_validator import assert_reasonable_duration
            try:
                duration = assert_reasonable_duration(file_path, min_sec=60)
                logger.info(f"Duration validation passed: {duration:.2f}s")
            except Exception as e:
                logger.error(f"Duration validation failed: {e}")
                # Continue processing but log the warning
                duration = 0.0
            
            # Convert to audio if needed
            tracker.update_stage("audio_processing", 0, "Converting to audio format...")
            write_progress(episode_id, "converting", 1, "Converting to audio format...")
            audio_path = await self._ensure_audio_format(file_path)
            
            # Store the audio path in the episode
            episode.audio_path = audio_path
            tracker.update_stage("audio_processing", 50, "Audio conversion completed")
            write_progress(episode_id, "converting", 1, "Audio conversion completed")
            
            # Get duration
            tracker.update_stage("audio_processing", 75, "Analyzing audio duration...")
            write_progress(episode_id, "transcribing", 1, "Transcribing...")
            duration = await self._get_audio_duration(audio_path)
            episode.duration = duration
            
            # Normalize word structure before processing
            from services.word_normalizer import normalize_words
            if hasattr(episode, 'words') and episode.words:
                episode.words = normalize_words(episode.words)
                logger.info(f"Normalized {len(episode.words)} words for episode {episode_id}")
            tracker.update_stage("audio_processing", 100, f"Audio ready - Duration: {duration:.1f}s")
            
            # Transcribe audio with single ASR path
            tracker.update_stage("transcription", 0, "Starting transcription...")
            transcript = await self._transcribe_audio_single_path(audio_path, episode_id, tracker, duration)
            
            # Store transcript in episode
            episode.transcript = transcript
            
            tracker.update_stage("transcription", 100, f"Transcription completed: {len(transcript)} segments")
            write_progress(episode_id, "transcribing", 95, f"Transcription completed: {len(transcript)} segments")
            
            # Save episode with transcript
            self._save_episode(episode)
            
            # Generate clips after transcription is complete
            tracker.update_stage("scoring", 0, "Finding viral clips...")
            write_progress(episode_id, "scoring", 85, "Finding viral moments...")
            
            try:
                # Check if clips already exist to avoid re-scoring
                # More robust check: ensure clips exist, are a list, and have content
                clips_exist = (hasattr(episode, 'clips') and 
                             episode.clips is not None and 
                             isinstance(episode.clips, list) and 
                             len(episode.clips) > 0)
                
                if clips_exist:
                    logger.info(f"Using existing clips for episode {episode_id}: {len(episode.clips)} clips")
                    clips = episode.clips
                    
                    # Ensure clips are also saved to file if they exist in memory but not on disk
                    try:
                        from pathlib import Path
                        episode_dir = Path(UPLOAD_DIR) / episode_id
                        clips_file = episode_dir / "clips.json"
                        
                        if not clips_file.exists():
                            episode_dir.mkdir(exist_ok=True)
                            
                            # Convert clips to serializable format
                            serializable_clips = []
                            for clip in clips:
                                serializable_clip = {}
                                for key, value in clip.items():
                                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                        serializable_clip[key] = value
                                    else:
                                        serializable_clip[key] = str(value)
                                serializable_clips.append(serializable_clip)
                            
                            with open(clips_file, 'w', encoding='utf-8') as f:
                                json.dump(serializable_clips, f, indent=2, ensure_ascii=False)
                            
                            logger.info(f"Saved existing {len(clips)} clips to {clips_file}")
                    except Exception as e:
                        logger.error(f"Failed to save existing clips to file: {e}")
                else:
                    logger.info(f"No existing clips found for episode {episode_id}, generating new clips...")
                    # Import clip scoring service
                    from services.clip_score import ClipScoreService
                    clip_score_service = ClipScoreService(self)
                    
                    # Generate clips using the scoring service
                    clips: list = []
                    try:
                        clips, _ = await clip_score_service.get_candidates(episode_id)
                    finally:
                        if clips is None:
                            clips = []  # ensure variable exists
                    
                    # Additional safety: treat empty finals as valid
                    if not clips:
                        logger.warning("REASON=EMPTY_AFTER_SALVAGE: episode has 0 clips")
                    
                    episode.clips = clips  # Store clips in episode
                    logger.info(f"Generated {len(clips)} clips for episode {episode_id}")
                
                tracker.update_stage("scoring", 100, f"Found {len(clips)} viral clips")
                write_progress(episode_id, "scoring", 100, f"Found {len(clips)} clips")
                
                # Save clips to clips.json file for title generation API
                try:
                    from pathlib import Path
                    from services.transcript_utils import words_between, words_to_text, words_to_captions, captions_to_vtt
                    
                    episode_dir = Path(UPLOAD_DIR) / episode_id
                    episode_dir.mkdir(exist_ok=True)
                    clips_file = episode_dir / "clips.json"
                    
                    # Convert clips to serializable format with exact transcript slicing
                    serializable_clips = []
                    for clip in clips:
                        # Build exact transcript from words that fall inside [start, end]
                        start = float(clip.get("start", 0))
                        end = float(clip.get("end", 0))
                        
                        # Get episode words for slicing
                        episode_words = getattr(episode, 'words', [])
                        if not episode_words:
                            # Fallback: try to get words from segments
                            episode_words = []
                            for seg in getattr(episode, 'segments', []):
                                if 'words' in seg and isinstance(seg['words'], list):
                                    episode_words.extend(seg['words'])
                        
                        # Debug: log word data structure
                        if episode_words:
                            logger.debug(f"CLIP_TRANSCRIPT: episode has {len(episode_words)} words")
                            # Check if words have the expected structure
                            if episode_words and isinstance(episode_words[0], dict):
                                sample_word = episode_words[0]
                                logger.debug(f"CLIP_TRANSCRIPT: sample word keys: {list(sample_word.keys())}")
                                if 'word' not in sample_word or 'start' not in sample_word or 'end' not in sample_word:
                                    logger.warning(f"CLIP_TRANSCRIPT: word structure missing required keys, using fallback")
                                    episode_words = []
                        else:
                            logger.warning(f"CLIP_TRANSCRIPT: no episode words found for clip {clip.get('id', 'unknown')}")
                        
                        # Use exact transcript builder
                        from services.transcript_builder import build_clip_transcript_exact
                        
                        try:
                            # Create a mock episode object with words data for transcript building
                            class MockEpisode:
                                def __init__(self, words_data):
                                    self.words = words_data
                                    self.words_normalized = words_data
                            
                            mock_episode = MockEpisode(episode_words) if episode_words else episode
                            
                            # Build exact transcript for the clip window
                            clip_text, transcript_source, transcript_meta = build_clip_transcript_exact(mock_episode, start, end)
                            
                            # For backward compatibility, create words array
                            if transcript_source == "word" and episode_words:
                                from services.word_utils import slice_transcript, normalize_word_token
                                words = slice_transcript(episode_words, start, end)
                                clip_words = [normalize_word_token(w) for w in words]
                                has_timestamps = all(w.get("start") is not None and w.get("end") is not None for w in clip_words)
                            else:
                                # Fallback: synthesize words array
                                clip_words = [{"word": t, "start": None, "end": None} for t in clip_text.split()[:50]]
                                has_timestamps = False
                            
                            # Convert to transcript utils format for captions
                            w = []
                            for word in clip_words:
                                if word.get("start") is not None and word.get("end") is not None:
                                    w.append({
                                        "word": word.get("word", ""),
                                        "start": word.get("start"),
                                        "end": word.get("end")
                                    })
                            
                            caps = words_to_captions(w, clip_start=start)
                            vtt = captions_to_vtt(caps)
                            
                        except Exception as e:
                            logger.error(f"CLIP_TRANSCRIPT: failed to slice words for clip {clip.get('id', 'unknown')}: {e}")
                            # Fallback to empty transcript
                            clip_text = "[Transcript unavailable]"
                            clip_words = []
                            has_timestamps = False
                            transcript_source = "error"
                            caps = []
                            vtt = "WEBVTT\n\n"
                        
                        # Create serializable clip with exact transcript data
                        serializable_clip = {}
                        for key, value in clip.items():
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                serializable_clip[key] = value
                            else:
                                serializable_clip[key] = str(value)
                        
                        # Add exact transcript data (always use actual start/end, never display jitter)
                        serializable_clip["actual_start"] = start
                        serializable_clip["actual_end"] = end
                        serializable_clip["actual_duration"] = round(end - start, 3)
                        
                        # Use new transcript structure
                        serializable_clip["transcript"] = {
                            "text": clip_text,
                            "words": clip_words,  # ALWAYS present
                            "has_timestamps": has_timestamps,
                            "source": transcript_source
                        }
                        
                        # Save the exact transcript text (built from words or segments)
                        serializable_clip["transcript"] = clip_text
                        serializable_clip["transcript_source"] = transcript_source
                        serializable_clip["transcript_meta"] = transcript_meta
                        serializable_clip["captions"] = caps
                        serializable_clip["vtt"] = vtt
                        
                        serializable_clips.append(serializable_clip)
                    
                    with open(clips_file, 'w', encoding='utf-8') as f:
                        json.dump(serializable_clips, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved {len(clips)} clips to {clips_file}")
                except Exception as e:
                    logger.error(f"Failed to save clips to file: {e}")
            except Exception as e:
                logger.error(f"Clip generation failed: {e}")
                episode.clips = []
                tracker.update_stage("scoring", 100, "Clip generation failed, continuing...")
                write_progress(episode_id, "scoring", 100, "Clip generation failed, continuing...")
            
            # Update episode status
            episode.status = "completed"
            episode.processed_at = datetime.now()
            tracker.update_stage("finalizing", 100, "Episode processing completed successfully!")
            write_progress(episode_id, "processing", 96, "Finalizing clips...")
            
            # ✅ NEW: mark as fully completed so frontend stops polling and fetches clips
            self._finalize_episode_processing(episode_id, len(clips))
            
            # Extend TTL for completed episodes so they remain accessible
            new_expires_at = time.time() + PROGRESS_TRACKER_TTL
            self.enhanced_trackers[episode_id] = (tracker, new_expires_at)
            
            # Save final episode state
            self._save_episode(episode)
            
            logger.info(f"Episode {episode_id} processing completed")
            
        except Exception as e:
            logger.error(f"Episode processing failed for {episode_id}: {e}")
            from services.progress_writer import write_progress
            write_progress(episode_id, "error", 0, f"{type(e).__name__}: {e}")
            
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
                self._mark_error(episode_id, f"Processing failed: {str(e)}")
        finally:
            # Remove from processing guard
            self._processing_episodes.discard(episode_id)

    def _finalize_episode_processing(self, episode_id: str, clip_count: int) -> None:
        """
        Idempotently mark an episode as finished. Safe to call multiple times.
        """
        try:
            from services.progress_writer import get_progress, write_progress
            cur = get_progress(episode_id) or {}
            progress_data = cur.get("progress", {})
            # already terminal?
            if progress_data.get("stage") == "completed":
                return
            # allow advancing scoring->completed even if percent is already 100
            write_progress(episode_id, "completed", 100, f"Ready: {clip_count} clips")
        except Exception as e:
            # don't fail the whole job on progress write
            logger.warning(
                "Finalize progress failed for %s: %s",
                episode_id, e,
            )
    
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
                
                # Skip conversion if WAV already exists
                if os.path.exists(audio_path):
                    logger.info(f"Audio file already exists, skipping conversion: {audio_path}")
                    return audio_path
                
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
            if WHISPER_TYPE == "faster-whisper":
                # Use faster-whisper API
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.whisper_model.transcribe(
                        audio_path,
                        language=WHISPER_LANGUAGE,
                        word_timestamps=True,
                        beam_size=self._get_whisper_beam_size(),
                        best_of=self._get_whisper_best_of(),
                        temperature=self._get_whisper_temperature(),
                        condition_on_previous_text=self._get_whisper_condition_previous(),
                        initial_prompt="This is a podcast episode with clear speech and natural conversation."
                    )
                )
            else:
                # Use regular whisper API
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.whisper_model.transcribe(
                        audio_path,
                        language=WHISPER_LANGUAGE,
                        word_timestamps=True,
                        fp16=self._should_use_fp16(),
                        verbose=False,
                        beam_size=self._get_whisper_beam_size(),
                        best_of=self._get_whisper_best_of(),
                        temperature=self._get_whisper_temperature(),
                        condition_on_previous_text=self._get_whisper_condition_previous(),
                        initial_prompt="This is a podcast episode with clear speech and natural conversation."
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
                        beam_size=1,
                        best_of=1,
                        temperature=0.0,
                        condition_on_previous_text=False
                    )
                )
                
                # Convert to our format
                transcript = []
                # Handle both tuple and dict return formats
                if isinstance(result, tuple):
                    segments, info = result
                else:
                    segments = result.get('segments', [])
                
                for i, segment in enumerate(segments):
                    # No transcript logging - just progress updates
                    
                    # Handle both dict and object segment formats
                    if isinstance(segment, dict):
                        transcript_segment = TranscriptSegment(
                            start=segment['start'],
                            end=segment['end'],
                            text=segment['text'].strip(),
                            confidence=segment.get('confidence', 0.0),
                            words=[]
                        )
                    else:
                        transcript_segment = TranscriptSegment(
                            start=getattr(segment, 'start', 0.0),
                            end=getattr(segment, 'end', 0.0),
                            text=getattr(segment, 'text', "").strip(),
                            confidence=getattr(segment, 'confidence', 0.0),
                            words=[]
                        )
                    transcript.append(transcript_segment)
                
                logger.info(f"Fallback transcription completed: {len(transcript)} segments")
                return transcript
                
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {fallback_error}")
                raise Exception(f"Both primary and fallback transcription failed: {e}, {fallback_error}")
    
    async def _transcribe_audio_single_path(self, audio_path: str, episode_id: str, tracker: EnhancedProgressTracker, total_duration: float) -> List[TranscriptSegment]:
        """Single ASR path using transcribe_with_quality with proper result normalization"""
        try:
            from services.progress_writer import write_progress
            
            # Pre-process audio to eliminate mpg123 errors
            asr_input = ensure_pcm_wav(audio_path)
            logger.info(f"Using pre-processed audio for ASR: {asr_input}")
            
            # Create ASR settings
            settings = ASRSettings()
            
            # --- Transcribe (retry here only) ---
            try:
                segments, info, words = transcribe_with_quality(asr_input, settings)
                logger.info(f"ASR_DONE: segments={len(segments)} words={'present' if words else 'none'}")
            except Exception as e:
                logger.error("TRANSCRIBE_FAIL (%s), falling back to CPU: %s", settings.device, e)
                cpu_settings = settings.with_cpu_fallback()
                segments, info, words = transcribe_with_quality(asr_input, cpu_settings)
                logger.info(f"ASR_DONE: segments={len(segments)} words={'present' if words else 'none'}")
            
            # --- Post-processing (never re-enters GPU) ---
            # Words are already extracted by transcribe_with_quality, but ensure they're safe
            safe_words = words or []  # belt & suspenders
            logger.info(f"WORDS_READY: count={len(safe_words)} (normalized/synthesized)")
            
            # No other ASR paths beyond this point
            # Persist words safely (never None)
            logger.info(f"SAVE_WORDS start")
            logger.info(f"WORDS_SAVE: n={len(safe_words)} eos_from_words={'yes' if safe_words else 'no'}")
            
            # Store quality metrics in episode metadata
            if hasattr(self, 'current_episode') and self.current_episode:
                self.current_episode.meta = self.current_episode.meta or {}
                self.current_episode.meta["asr_quality"] = info
                self.current_episode.words = safe_words
            
            logger.info(f"SAVE_WORDS done")
            
            # Convert segments to our format
            segments_list = []
            for seg in segments:
                words_for_seg = []
                if hasattr(seg, 'words') and seg.words:
                    words_for_seg = [{"text": w.word, "start": w.start, "end": w.end, "prob": getattr(w, 'probability', 0.0)} for w in seg.words]
                
                segments_list.append(TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    raw_text=seg.text.strip(),
                    words=words_for_seg
                ))
            
            # Update progress
            tracker.update_stage("transcription", 100, "Transcription complete")
            write_progress(episode_id, "transcribing", 95, "Transcription complete")
            logger.info(f"Single ASR path completed: {len(segments_list)} segments, {len(safe_words)} words")
            
            return segments_list
            
        except Exception as e:
            logger.error(f"Single ASR path failed: {e}")
            # Final fallback to original transcription method
            logger.info("Falling back to original transcription method...")
            return await self._transcribe_audio(audio_path, episode_id)

    async def _transcribe_audio_with_progress(self, audio_path: str, episode_id: str, tracker: EnhancedProgressTracker) -> List[TranscriptSegment]:
        """Enhanced transcription with better progress tracking"""
        try:
            from services.progress_writer import write_progress
            
            # Lazy load Whisper model if not available
            if not self.whisper_model:
                tracker.update_stage("transcription", 5, "Loading Whisper model...")
                write_progress(episode_id, "transcribing", 5, "Loading model...")
                try:
                    self.whisper_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
                except Exception as load_error:
                    logger.error(f"Failed to load Whisper model: {load_error}")
                    raise RuntimeError(f"Whisper model not available: {load_error}")
            
            logger.info(f"Starting enhanced transcription for {audio_path}")
            tracker.update_stage("transcription", 10, "Starting Whisper transcription...")
            write_progress(episode_id, "transcribing", 10, "Starting transcription...")
            
            # Get audio duration for progress calculation
            duration = await self._get_audio_duration(audio_path)
            
            # Run Whisper transcription with timeout
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Update progress to show transcription is running
            tracker.update_stage("transcription", 20, "Whisper model processing audio...")
            write_progress(episode_id, "transcribing", 20, "Processing audio...")
            
            # Run transcription in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    audio_path,
                    language=WHISPER_LANGUAGE,
                    word_timestamps=True,
                    beam_size=5,
                    best_of=5,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    condition_on_previous_text=True,
                    initial_prompt="This is a podcast episode with clear speech and natural conversation."
                )
            )
            
            # Update progress during processing
            tracker.update_stage("transcription", 60, "Processing transcription results...")
            write_progress(episode_id, "transcribing", 60, "Processing results...")
            
            # Convert to our format with progress updates
            transcript = []
            # Handle both tuple and dict return formats
            if isinstance(result, tuple):
                segments, info = result
            else:
                segments = result.get("segments", [])
            
            # Convert generator to list if needed
            if hasattr(segments, '__iter__') and not isinstance(segments, (list, tuple)):
                segments = list(segments)
            
            total_segments = len(segments)
            
            for i, segment in enumerate(segments):
                # Update progress based on segment processing
                segment_progress = 60 + (i / max(1, total_segments)) * 35  # 60-95%
                tracker.update_stage("transcription", segment_progress, 
                                   f"Processing segments... {i+1}/{total_segments}")
                write_progress(episode_id, "transcribing", int(segment_progress), 
                             f"Processing segments... {i+1}/{total_segments}")
                
                # Handle both dict and object segment formats
                if isinstance(segment, dict):
                    transcript_segment = TranscriptSegment(
                        start=segment.get("start", 0.0),
                        end=segment.get("end", 0.0),
                        text=segment.get("text", "").strip()
                    )
                else:
                    transcript_segment = TranscriptSegment(
                        start=getattr(segment, 'start', 0.0),
                        end=getattr(segment, 'end', 0.0),
                        text=getattr(segment, 'text', "").strip()
                    )
                transcript.append(transcript_segment)
            
            logger.info(f"Enhanced transcription completed: {len(transcript)} segments")
            # Use the final calculated progress (95%) instead of hardcoding 80%
            final_progress = 60 + (total_segments / max(1, total_segments)) * 35  # This should be 95%
            write_progress(episode_id, "transcribing", int(final_progress), f"Transcription complete: {len(transcript)} segments")
            return transcript
            
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            # Fallback to original method - still use pre-processed audio
            logger.info("Falling back to original transcription method...")
            return await self._transcribe_audio(audio_for_asr, episode_id)
    
    async def _transcribe_with_faster_whisper(self, audio_path: str, episode_id: str, tracker: EnhancedProgressTracker, total_duration: float) -> List[TranscriptSegment]:
        """Transcribe using faster-whisper with streaming progress"""
        try:
            from services.progress_writer import write_progress
            
            segments_list = []
            tracker.update_stage("transcription", 10, "Starting Faster-Whisper transcription...")
            write_progress(episode_id, "transcribing", 10, "Starting transcription...")
            
            if ENABLE_ASR_V2:
                # Use ASR v2 with quality-aware two-pass transcription
                
                # Pre-process audio to eliminate mpg123 errors
                if AUDIO_PREDECODE_PCM:
                    from services.audio_io import ensure_pcm_wav
                    audio_for_asr = ensure_pcm_wav(audio_path)
                    logger.info(f"Using pre-processed audio for ASR: {audio_for_asr}")
                else:
                    audio_for_asr = audio_path
                    logger.info(f"Using original audio for ASR: {audio_for_asr}")
                
                # Run ASR v2 in thread pool
                import asyncio
                loop = asyncio.get_event_loop()
                segments, info, quality_metrics = await loop.run_in_executor(
                    None,
                    lambda: transcribe_with_quality(audio_for_asr, language="en")
                )
                
                # Store quality metrics in episode metadata
                if hasattr(self, 'current_episode') and self.current_episode:
                    self.current_episode.meta = self.current_episode.meta or {}
                    self.current_episode.meta["asr_quality"] = quality_metrics
                
                # Update progress if retry was used
                if quality_metrics.get("retried"):
                    tracker.update_stage("transcription", 90, "High-quality retry completed...")
                    write_progress(episode_id, ASR_PROGRESS_LABEL_HQ, 90)
                
                # Convert segments to our format
                for seg in segments:
                    words = []
                    if hasattr(seg, 'words') and seg.words:
                        words = [{"text": w.word, "start": w.start, "end": w.end, "prob": getattr(w, 'probability', 0.0)} for w in seg.words]
                    
                    segments_list.append(TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                        raw_text=seg.text.strip(),
                        words=words
                    ))
                
                # Store normalized words for boundary refinement - SAFE EXTRACTION
                if hasattr(self, 'current_episode') and self.current_episode:
                    try:
                        # Use words from ASR response if available, otherwise extract from segments
                        words = quality_metrics.get("words")
                        if words is None:
                            words = _extract_words_safe(segments)
                        # ensure never None before len() or serialization
                        words = words or []
                        self.current_episode.words = words
                        logger.info("WORDS_SAVE: count=%d for episode %s", len(words), episode_id)
                    except Exception as e:
                        logger.exception("Failed to save words for episode %s: %s", episode_id, e)
                        self.current_episode.words = []
                
            # ASR v2 is the only path now - no legacy fallback needed
            
            # Collect all words for episode-level storage - SAFE EXTRACTION
            all_words = []
            last_pct = 10
            try:
                # Use words from ASR response if available, otherwise extract from segments
                if hasattr(self, 'current_episode') and self.current_episode and hasattr(self.current_episode, 'words'):
                    all_words = self.current_episode.words or []
                    logger.info("Using words from current episode: %d words", len(all_words))
                else:
                    # Fallback: extract from segments
                    words = _extract_words_safe(segments_list)
                    all_words = words or []
                    logger.info("Extracted %d words from %d segments", len(all_words), len(segments_list))
            except Exception as e:
                logger.exception("Failed to extract words from segments: %s", e)
                all_words = []
            
            # Calculate progress based on audio position
            for seg in segments_list:
                pct = min(95.0, 10 + (seg.end / max(1.0, total_duration)) * 85.0)
                
                if pct - last_pct >= 2.0:  # Update every 2%
                    detail = f"{int(seg.end)}s / {int(total_duration)}s"
                    tracker.update_stage("transcription", pct, "Transcribing audio...", detail)
                    write_progress(episode_id, "transcribing", int(pct), f"Transcribing... {detail}")
                    last_pct = pct
                    await asyncio.sleep(0)  # Yield control
                    
            # Validate word count and store episode-level data
            word_count = len(all_words)
            if word_count < 500:  # short ep threshold; use 1500–3000 for long eps
                logger.warning(f"EOS_SPARSE: word_count={word_count} too low; enabling fallback mode")
            
            # Store word data in episode for transcript slicing - SAFE STORAGE
            episode = await self.get_episode(episode_id)
            if episode:
                try:
                    # Normalize word structure before storing
                    from services.word_normalizer import normalize_words
                    normalized_words = normalize_words(all_words)
                    # ensure never None before len() or serialization
                    normalized_words = normalized_words or []
                    episode.words = normalized_words
                    episode.word_count = len(normalized_words)
                    episode.raw_text = " ".join([seg.text for seg in segments_list])
                    logger.info("WORDS_SAVE: saved %d normalized words for episode %s", len(normalized_words), episode_id)
                except Exception as e:
                    logger.exception("Failed to save normalized words for episode %s: %s", episode_id, e)
                    episode.words = []
                    episode.word_count = 0
                
                # Attach normalized words to segments for local EOS detection
                self._attach_words_to_segments(episode.words, segments_list)
                logger.info(f"Normalized {len(episode.words)} words for episode {episode_id}")
                
                self._save_episode(episode)
            
            tracker.update_stage("transcription", 100, "Faster-Whisper transcription complete")
            write_progress(episode_id, "transcribing", 95, "Transcription complete")
            logger.info(f"Faster-Whisper transcription completed: {len(segments_list)} segments, {word_count} words")
            return segments_list
            
        except Exception as e:
            logger.error(f"ASR v2 transcription failed: {e}")
            # Fallback to original transcription method - still use pre-processed audio
            logger.info("Falling back to original transcription method...")
            return await self._transcribe_audio(audio_for_asr, episode_id)
    
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

    def mark_completed(self, episode_id: str) -> None:
        """Optionally drop as soon as results are persisted"""
        self.episodes.pop(episode_id, None)


    def get_cache_stats(self) -> Dict[str, float]:
        total = self._hits + self._misses
        return {
            "size": float(len(self.episodes)),
            "maxsize": float(self.episodes.maxsize),
            "hits": float(self._hits),
            "misses": float(self._misses),
            "hit_rate": (self._hits / total) if total else 0.0,
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        }
