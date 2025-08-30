"""
Pydantic models for the PodPromo AI backend.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TranscriptSegment(BaseModel):
    """Individual word or segment from transcript"""
    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None

class AudioFeatures(BaseModel):
    """Audio features extracted from a segment"""
    hook_score: float
    prosody_score: float
    emotion_score: float
    question_score: float
    payoff_score: float
    info_density: float
    loopability: float

class MomentScore(BaseModel):
    """Scored moment with all features"""
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

class EpisodeBase(BaseModel):
    """Base episode model"""
    filename: str
    original_filename: str
    size: Optional[int] = None
    duration: Optional[float] = None
    status: str = "uploaded"

class Episode(BaseModel):
    """Episode with full data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    original_name: str
    size: Optional[int] = None
    duration: Optional[float] = None
    status: str = "uploaded"
    uploaded_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error: Optional[str] = None
    transcript: Optional[List[TranscriptSegment]] = None
    audio_path: Optional[str] = None  # Path to the audio file

    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    """Response for file upload"""
    episode_id: str = Field(alias="episodeId")
    message: str

    class Config:
        populate_by_name = True

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, bool]

class ApiError(BaseModel):
    """API error response"""
    error: str
    detail: Optional[str] = None

class RenderRequest(BaseModel):
    """Request for enhanced clip rendering"""
    clip_id: str
    style: str = "bold"         # bold, clean, caption-heavy
    captions: bool = True
    punch_ins: bool = True
    loop_seam: bool = False

class Clip(BaseModel):
    """Generated clip model"""
    id: str
    episode_id: str
    start_time: float
    end_time: float
    duration: float
    score: float
    title: str
    description: str
    status: str
    output_path: Optional[str] = None
    download_url: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class ClipGenerationRequest(BaseModel):
    """Request for clip generation"""
    episode_id: str
    target_count: int = 3
    min_duration: int = 12
    max_duration: int = 30

class ClipGenerationResponse(BaseModel):
    """Response for clip generation"""
    jobId: str
    status: str
    clips: List[Clip]
    estimatedTime: int


