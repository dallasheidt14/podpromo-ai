"""
Database models for the podcast clip generation system
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
# Temporarily disabled for Python 3.13 compatibility
# from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
# from sqlalchemy.orm import relationship, declarative_base
# from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
import uuid

# Import database configuration
# from config.database import Base

# ============================================================================
# SQLAlchemy ORM Models (for database persistence) - TEMPORARILY DISABLED
# ============================================================================

# class EpisodeDB(Base):
#     """Database model for episodes"""
#     __tablename__ = "episodes"
#     
#     id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
#     filename = Column(String, nullable=False)
#     original_name = Column(String, nullable=False)
#     size = Column(Integer, nullable=False)  # File size in bytes
#     status = Column(String, default="uploading")  # uploading, processing, completed, failed
#     duration = Column(Float, nullable=True)  # Audio duration in seconds
#     audio_path = Column(String, nullable=True)
#     error = Column(Text, nullable=True)
#     uploaded_at = Column(DateTime, default=datetime.utcnow)
#     processed_at = Column(DateTime, nullable=True)
#     
#     # Relationships
#     transcript_segments = relationship("TranscriptSegmentDB", back_populates="episode", cascade="all, delete-orphan")
#     clips = relationship("ClipDB", back_populates="episode", cascade="all, delete-orphan")
#     feedback = relationship("FeedbackDB", back_populates="episode", cascade="all, delete-orphan")

# class TranscriptSegmentDB(Base):
#     """Database model for transcript segments"""
#     __tablename__ = "transcript_segments"
#     
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
#     start = Column(Float, nullable=False)
#     end = Column(Float, nullable=False)
#     text = Column(Text, nullable=False)
#     confidence = Column(Float, default=0.0)
#     words = Column(JSON, nullable=True)  # Store word-level timing as JSON
#     
#     # Relationships
#     episode = relationship("EpisodeDB", back_populates="transcript_segments")

# class ClipDB(Base):
#     """Database model for generated clips"""
#     __tablename__ = "clips"
#     
#     id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
#     episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
#     start_time = Column(Float, nullable=False)
#     end_time = Column(Float, nullable=False)
#     duration = Column(Float, nullable=False)
#     file_path = Column(String, nullable=True)
#     thumbnail_path = Column(String, nullable=True)
#     score = Column(Float, nullable=False)
#     confidence = Column(String, nullable=True)
#     genre = Column(String, default="general")
#     platform = Column(String, default="tiktok")
#     features = Column(JSON, nullable=True)  # Store computed features
#     created_at = Column(DateTime, default=datetime.utcnow)
#     
#     # Relationships
#     episode = relationship("EpisodeDB", back_populates="clips")

# class FeedbackDB(Base):
#     """Database model for user feedback"""
#     __tablename__ = "feedback"
#     
#     id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
#     episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
#     clip_id = Column(String, ForeignKey("clips.id"), nullable=True)
#     feedback_type = Column(String, nullable=False)  # like, dislike, flag, comment
#     rating = Column(Integer, nullable=True)  # 1-5 scale
#     comment = Column(Text, nullable=True)
#     user_id = Column(String, nullable=True)  # For future authentication
#     created_at = Column(DateTime, default=datetime.utcnow)
#     
#     # Relationships
#     episode = relationship("EpisodeDB", back_populates="feedback")
#     clip = relationship("ClipDB")

# class UserSessionDB(Base):
#     """Database model for user sessions (future authentication)"""
#     __tablename__ = "user_sessions"
#     
#     id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
#     user_id = Column(String, nullable=False)
#     session_token = Column(String, unique=True, nullable=False)
#     expires_at = Column(DateTime, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     last_activity = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# Pydantic Models (for API requests/responses)
# ============================================================================

class TranscriptSegment(BaseModel):
    """Pydantic model for transcript segments"""
    start: float
    end: float
    text: str
    confidence: float = 0.0
    words: Optional[List[Dict[str, Any]]] = None
    
    model_config = {
        "from_attributes": True
    }

class AudioFeatures(BaseModel):
    """Audio features for scoring"""
    duration: float
    energy: float
    tempo: float
    loudness: float
    speech_rate: float
    silence_ratio: float
    dynamic_range: float
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc_features: List[float]
    chroma_features: List[float]
    
    model_config = {
        "from_attributes": True
    }

class Episode(BaseModel):
    """Pydantic model for episode API responses"""
    id: str
    filename: str
    original_name: str
    size: int
    status: str
    duration: Optional[float] = None
    audio_path: Optional[str] = None
    transcript: Optional[List[TranscriptSegment]] = None
    clips: Optional[List[Dict[str, Any]]] = None  # Store generated clips
    words: Optional[List[Dict[str, Any]]] = None  # Store word-level timestamps
    word_count: Optional[int] = None  # Store word count for EOS index
    error: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    model_config = {
        "from_attributes": True,
        "frozen": False  # Allow mutation of fields after creation
    }

class MomentScore(BaseModel):
    """Pydantic model for moment scoring"""
    start_time: float
    end_time: float
    duration: float
    hook_score: float
    emotion_score: float
    arousal_score: float
    payoff_score: float
    loopability_score: float
    question_or_list_score: float
    info_density_score: float
    total_score: float

class ClipCandidate(BaseModel):
    """Pydantic model for clip candidates"""
    start: float
    end: float
    text: str
    score: float
    confidence: str
    genre: str = "general"
    platform: str = "tiktok"
    features: Dict[str, Any] = {}
    
    model_config = {
        "from_attributes": True
    }

class HealthCheck(BaseModel):
    """Pydantic model for health check responses"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, Union[bool, Dict[str, Any]]]

class UploadResponse(BaseModel):
    """Pydantic model for upload responses"""
    episode_id: str
    status: str
    message: str

class CandidatesResponse(BaseModel):
    """Pydantic model for candidates responses"""
    ok: bool
    candidates: List[ClipCandidate]
    error: Optional[str] = None

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

# User Authentication Models
class UserSignup(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    name: str = Field(..., description="User full name")

class UserLogin(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")

class UserProfile(BaseModel):
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User full name")
    plan: str = Field(..., description="Current plan (free/pro)")
    subscription_id: Optional[str] = Field(None, description="Paddle subscription ID")
    status: str = Field(..., description="Account status")
    created_at: str = Field(..., description="Account creation date")
    updated_at: str = Field(..., description="Last update date")

# Membership Models
class MembershipPlan(BaseModel):
    id: str = Field(..., description="Plan ID")
    name: str = Field(..., description="Plan name")
    price: float = Field(..., description="Monthly price in USD")
    billing: str = Field(..., description="Billing interval")
    features: List[str] = Field(..., description="Plan features")
    description: str = Field(..., description="Plan description")

class UserMembership(BaseModel):
    user_id: str = Field(..., description="User ID")
    plan: str = Field(..., description="Plan type (free/pro)")
    subscription_id: Optional[str] = Field(None, description="Paddle subscription ID")
    status: str = Field(..., description="Membership status")
    start_date: str = Field(..., description="Membership start date")
    end_date: Optional[str] = Field(None, description="Membership end date")

# Paddle Integration Models
class PaddleCheckout(BaseModel):
    user_id: str = Field(..., description="User ID")
    plan: str = Field(..., description="Plan type (free/pro)")

class PaddleWebhook(BaseModel):
    event_type: str = Field(..., description="Webhook event type")
    data: Dict[str, Any] = Field(..., description="Webhook data payload")

# Usage Tracking Models
class UsageLog(BaseModel):
    id: str = Field(..., description="Usage log ID")
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Action performed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    created_at: str = Field(..., description="Action timestamp")

class UsageSummary(BaseModel):
    user_id: str = Field(..., description="User ID")
    plan: str = Field(..., description="Current plan")
    current_month: Dict[str, int] = Field(..., description="Current month usage")
    limits: Dict[str, Any] = Field(..., description="Plan limits")
    can_upload: bool = Field(..., description="Whether user can upload")
    remaining_uploads: Optional[int] = Field(None, description="Remaining uploads (free plan only)")


