"""
Database models for the podcast clip generation system
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
import uuid

# Import database configuration
from config.database import Base

# ============================================================================
# SQLAlchemy ORM Models (for database persistence)
# ============================================================================

class EpisodeDB(Base):
    """Database model for episodes"""
    __tablename__ = "episodes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    size = Column(Integer, nullable=False)  # File size in bytes
    status = Column(String, default="uploading")  # uploading, processing, completed, failed
    duration = Column(Float, nullable=True)  # Audio duration in seconds
    audio_path = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    transcript_segments = relationship("TranscriptSegmentDB", back_populates="episode", cascade="all, delete-orphan")
    clips = relationship("ClipDB", back_populates="episode", cascade="all, delete-orphan")
    feedback = relationship("FeedbackDB", back_populates="episode", cascade="all, delete-orphan")

class TranscriptSegmentDB(Base):
    """Database model for transcript segments"""
    __tablename__ = "transcript_segments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    words = Column(JSON, nullable=True)  # Store word-level timing as JSON
    
    # Relationships
    episode = relationship("EpisodeDB", back_populates="transcript_segments")

class ClipDB(Base):
    """Database model for generated clips"""
    __tablename__ = "clips"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    file_path = Column(String, nullable=True)
    thumbnail_path = Column(String, nullable=True)
    score = Column(Float, nullable=False)
    confidence = Column(String, nullable=True)
    genre = Column(String, default="general")
    platform = Column(String, default="tiktok")
    features = Column(JSON, nullable=True)  # Store computed features
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    episode = relationship("EpisodeDB", back_populates="clips")

class FeedbackDB(Base):
    """Database model for user feedback"""
    __tablename__ = "feedback"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    clip_id = Column(String, ForeignKey("clips.id"), nullable=True)
    feedback_type = Column(String, nullable=False)  # like, dislike, flag, comment
    rating = Column(Integer, nullable=True)  # 1-5 scale
    comment = Column(Text, nullable=True)
    user_id = Column(String, nullable=True)  # For future authentication
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    episode = relationship("EpisodeDB", back_populates="feedback")
    clip = relationship("ClipDB")

class UserSessionDB(Base):
    """Database model for user sessions (future authentication)"""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    session_token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# Pydantic Models (for API requests/responses)
# ============================================================================

class Episode(BaseModel):
    """Pydantic model for episode API responses"""
    id: str
    filename: str
    original_name: str
    size: int
    status: str
    duration: Optional[float] = None
    audio_path: Optional[str] = None
    error: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class TranscriptSegment(BaseModel):
    """Pydantic model for transcript segments"""
    start: float
    end: float
    text: str
    confidence: float = 0.0
    words: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        from_attributes = True

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
    
    class Config:
        from_attributes = True

class HealthCheck(BaseModel):
    """Pydantic model for health check responses"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, bool]

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


