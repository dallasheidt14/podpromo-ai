from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import uuid

# Base models
class EpisodeBase(BaseModel):
    filename: str
    original_name: str
    size: int
    duration: Optional[float] = None
    status: str = "uploading"

class ClipBase(BaseModel):
    episode_id: str
    start_time: float
    end_time: float
    duration: float
    score: float
    title: str
    description: str
    status: str = "generating"

# Request models
class ClipGenerationRequest(BaseModel):
    episode_id: str
    target_count: Optional[int] = Field(default=3, ge=1, le=10)
    min_duration: Optional[int] = Field(default=12, ge=5, le=60)
    max_duration: Optional[int] = Field(default=30, ge=10, le=120)

    @validator('max_duration')
    def max_duration_must_be_greater_than_min(cls, v, values):
        if 'min_duration' in values and v <= values['min_duration']:
            raise ValueError('max_duration must be greater than min_duration')
        return v

# Response models
class UploadResponse(BaseModel):
    episode_id: str = Field(alias="episodeId")
    filename: str
    size: int
    status: str
    message: Optional[str] = None

    class Config:
        populate_by_name = True

class ClipGenerationResponse(BaseModel):
    job_id: str
    status: str
    clips: List['Clip']
    estimated_time: Optional[int] = None

class ApiError(BaseModel):
    error: str
    message: str
    status_code: int

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    services: dict

# Full models with all fields
class Episode(EpisodeBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    error: Optional[str] = None
    transcript: Optional[List['TranscriptSegment']] = None
    audio_path: Optional[str] = None  # Path to the audio file

    class Config:
        from_attributes = True

class Clip(ClipBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    output_path: Optional[str] = None
    download_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        from_attributes = True

# Audio and analysis models
class AudioFeatures(BaseModel):
    loudness: List[float]
    pitch: List[float]
    tempo: List[float]
    energy: List[float]
    spectral_centroid: List[float]

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: float
    words: List[dict]

class MomentScore(BaseModel):
    start_time: float
    end_time: float
    duration: float
    hook_score: float
    emotion_score: float
    prosody_score: float
    payoff_score: float
    loopability_score: float
    question_or_list_score: float
    info_density_score: float
    total_score: float

class ClipGenerationConfig(BaseModel):
    target_count: int = 3
    min_duration: int = 12
    max_duration: int = 30
    video_width: int = 1080
    video_height: int = 1920
    fps: int = 30
    audio_sample_rate: int = 44100
    caption_style: str = "modern"
    enable_punch_ins: bool = True
    enable_audio_normalization: bool = True
    enable_loopable_ending: bool = True

# Update forward references
ClipGenerationResponse.model_rebuild()
