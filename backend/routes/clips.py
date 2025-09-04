# backend/routes/clips.py
from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel, Field, conint, confloat, validator
from typing import List, Optional
import uuid
from datetime import datetime
import hashlib
import json

router = APIRouter(prefix="/api/episodes", tags=["clips"])

# Constants
MAX_TARGET_COUNT = 50

class TimeWindow(BaseModel):
    startSec: float = Field(ge=0)
    endSec: float = Field(gt=0)
    
    @validator("endSec")
    def end_gt_start(cls, v, values):
        if "startSec" in values and v <= values["startSec"]:
            raise ValueError("endSec must be > startSec")
        return v

class LoopSeam(BaseModel):
    enabled: bool = True
    maxGapSec: float = Field(default=0.25, ge=0, le=1.0)

class ScoreWeights(BaseModel):
    hook: float = 0.35
    emotion: float = 0.25
    payoff: float = 0.20
    loop: float = 0.10
    novelty: float = 0.10
    
    @validator("*", pre=True)
    def nonneg(cls, v): 
        if v < 0: 
            raise ValueError("weights must be >=0")
        return v
    
    @validator("*", pre=True, check_fields=False)
    def normalize(cls, v):
        if hasattr(v, 'hook'):
            s = sum([v.hook, v.emotion, v.payoff, v.loop, v.novelty])
            if s <= 0: 
                raise ValueError("sum of weights must be > 0")
        return v

class ClipGenRequest(BaseModel):
    targetCount: conint(ge=1, le=MAX_TARGET_COUNT) = 12
    minDurationSec: confloat(ge=5, le=120) = 12
    maxDurationSec: confloat(ge=10, le=90) = 45
    excludeAds: bool = True
    scoreThreshold: confloat(ge=0, le=1) = 0.0
    scoreWeights: ScoreWeights = ScoreWeights()
    timeWindows: Optional[List[TimeWindow]] = None
    loopSeam: LoopSeam = LoopSeam()
    language: str = "en"
    strategy: str = Field("topk", pattern="^(topk|diverse|hybrid)$")
    diversityPenalty: confloat(ge=0, le=1) = 0.15
    seed: Optional[int] = None
    regenerate: bool = False
    notes: Optional[str] = None
    
    @validator("maxDurationSec")
    def max_ge_min(cls, v, values):
        if "minDurationSec" in values and v < values["minDurationSec"]:
            raise ValueError("maxDurationSec must be >= minDurationSec")
        return v

# Simple in-memory job tracking (in production, use Redis)
_job_tracking = {}
_idempotency_keys = {}

def _generate_cache_key(episode_id: str, params: dict) -> str:
    """Generate a cache key based on episode ID and parameters"""
    # Create a deterministic hash of the parameters
    param_str = json.dumps(params, sort_keys=True)
    return f"clips:{episode_id}:{hashlib.md5(param_str.encode()).hexdigest()[:8]}"

def _check_cached_candidates(episode_id: str, params: dict) -> bool:
    """Check if we have cached candidates for these parameters"""
    # For now, just check if the episode has any clips at all
    # In production, you'd check a proper cache with the specific param hash
    try:
        from services.episode_service import get_episode
        episode = get_episode(episode_id)
        return episode is not None and hasattr(episode, 'clips') and len(episode.clips) > 0
    except:
        return False

async def _get_episode_duration(episode_id: str) -> Optional[float]:
    """Get episode duration in seconds"""
    try:
        from services.episode_service import EpisodeService
        episode_service = EpisodeService()
        episode = await episode_service.get_episode(episode_id)
        if episode and hasattr(episode, 'duration_s'):
            return episode.duration_s
        elif episode and hasattr(episode, 'duration'):
            return episode.duration
        # If no duration, assume it exists and return a default
        return 3600.0  # 1 hour default
    except:
        return None

def _job_exists(episode_id: str, idempotency_key: str) -> bool:
    """Check if a job with this idempotency key is already running"""
    return idempotency_key in _idempotency_keys

def _enqueue_clip_job(episode_id: str, params: dict, idempotency_key: Optional[str] = None) -> str:
    """Enqueue a clip generation job"""
    job_id = f"clips:{episode_id}:{uuid.uuid4().hex[:8]}"
    
    # Store job metadata
    _job_tracking[job_id] = {
        "episode_id": episode_id,
        "params": params,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "idempotency_key": idempotency_key
    }
    
    if idempotency_key:
        _idempotency_keys[idempotency_key] = job_id
    
    # In production, you'd enqueue this to your actual job queue
    # For now, we'll simulate immediate processing
    print(f"[CLIPS] Enqueued job {job_id} for episode {episode_id}")
    
    return job_id

@router.post("/{episode_id}/clips")
async def generate_clips(
    episode_id: str,
    body: ClipGenRequest,
    x_idempotency_key: Optional[str] = Header(default=None, convert_underscores=True)
):
    """Generate clips for an episode with parameterized options"""
    
    # Validate episode exists
    duration = await _get_episode_duration(episode_id)
    if duration is None:
        raise HTTPException(status_code=404, detail="Unknown episode_id")
    
    # Validate time windows inside episode duration
    if body.timeWindows:
        for tw in body.timeWindows:
            if tw.endSec > duration + 1e-6:
                raise HTTPException(
                    status_code=422, 
                    detail=f"timeWindow exceeds episode duration ({duration:.1f}s)"
                )
    
    # Convert to dict for processing
    params = body.dict()
    
    # Cache short-circuit
    if not body.regenerate and _check_cached_candidates(episode_id, params):
        return {
            "ok": True, 
            "started": False, 
            "episodeId": episode_id, 
            "cached": True,
            "next": {"results": f"/api/episodes/{episode_id}/clips"}
        }
    
    # Idempotency conflict check
    if x_idempotency_key and _job_exists(episode_id, x_idempotency_key):
        return {
            "ok": True, 
            "started": True, 
            "episodeId": episode_id,
            "job": {
                "id": f"clips:{episode_id}", 
                "queuedAt": datetime.utcnow().isoformat(),
                "idempotencyKey": x_idempotency_key
            },
            "next": {
                "progress": f"/api/progress/{episode_id}", 
                "results": f"/api/episodes/{episode_id}/clips"
            }
        }
    
    # Enqueue job
    job_id = _enqueue_clip_job(
        episode_id=episode_id,
        params=params,
        idempotency_key=x_idempotency_key
    )
    
    return {
        "ok": True, 
        "started": True, 
        "episodeId": episode_id,
        "job": {
            "id": job_id, 
            "queuedAt": datetime.utcnow().isoformat(),
            "idempotencyKey": x_idempotency_key
        },
        "next": {
            "progress": f"/api/progress/{episode_id}", 
            "results": f"/api/episodes/{episode_id}/clips"
        }
    }
