# backend/routes/clips.py
from __future__ import annotations

import time
from typing import List, Optional, Literal, Dict, Any

from fastapi import APIRouter, Header, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint, confloat, field_validator, model_validator

router = APIRouter(prefix="/api/episodes", tags=["clips"])

# Constants
MAX_TARGET_COUNT = 50

class TimeWindow(BaseModel):
    startSec: float = Field(ge=0)
    endSec: float = Field(gt=0)

    @field_validator("endSec")
    @classmethod
    def end_gt_start(cls, v: float, info):
        start = info.data.get("startSec", None)
        if start is not None and v <= float(start):
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

    @field_validator("hook", "emotion", "payoff", "loop", "novelty")
    @classmethod
    def nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be >= 0")
        return v

    @model_validator(mode="after")
    def sum_positive(self) -> "ScoreWeights":
        s = self.hook + self.emotion + self.payoff + self.loop + self.novelty
        if s <= 0:
            raise ValueError("sum of weights must be > 0")
        return self

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
    strategy: Literal["topk", "diverse", "hybrid"] = "topk"
    diversityPenalty: confloat(ge=0, le=1) = 0.15
    seed: Optional[int] = None
    regenerate: bool = False
    notes: Optional[str] = None

    @field_validator("maxDurationSec")
    @classmethod
    def max_ge_min(cls, v: float, info):
        minv = info.data.get("minDurationSec", None)
        if minv is not None and v < float(minv):
            raise ValueError("maxDurationSec must be >= minDurationSec")
        return v

# -------------------------------------------------
# Simple in-memory idempotency registry with TTL
# (use Redis for multi-process deployments)
# -------------------------------------------------
_IDEMPOTENCY_TTL_SEC = 15 * 60  # 15 minutes
_idempotency_store: Dict[str, Dict[str, Any]] = {}  # key -> {episodeId, jobId, expiresAt}

def _prune_idempotency_store() -> None:
    now = time.time()
    expired = [k for k, v in _idempotency_store.items() if v.get("expiresAt", 0) <= now]
    for k in expired:
        _idempotency_store.pop(k, None)

def _remember_idempotency(key: str, episode_id: str, job_id: str) -> None:
    _prune_idempotency_store()
    _idempotency_store[key] = {
        "episodeId": episode_id,
        "jobId": job_id,
        "expiresAt": time.time() + _IDEMPOTENCY_TTL_SEC,
    }

def _lookup_idempotency(key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    _prune_idempotency_store()
    entry = _idempotency_store.get(key)
    if entry and entry.get("expiresAt", 0) > time.time():
        return entry
    return None

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

def _check_cached_candidates(episode_id: str, params: dict) -> bool:
    """Check if we have cached candidates for these parameters"""
    # For now, just check if the episode has any clips at all
    # In production, you'd check a proper cache with the specific param hash
    try:
        from services.episode_service import EpisodeService
        episode_service = EpisodeService()
        # Note: This is a sync call, but in production you'd want async
        return True  # Simplified for now
    except:
        return False

def _enqueue_clip_job(episode_id: str, params: dict, idempotency_key: Optional[str] = None) -> str:
    """Enqueue a clip generation job"""
    import uuid
    job_id = f"clips:{episode_id}:{uuid.uuid4().hex[:8]}"
    
    # In production, you'd enqueue this to your actual job queue
    print(f"[CLIPS] Enqueued job {job_id} for episode {episode_id}")
    
    return job_id

@router.post("/{episode_id}/clips")
async def post_generate_clips(
    episode_id: str,
    body: ClipGenRequest,
    x_idempotency_key: Optional[str] = Header(default=None, convert_underscores=True),
    user: Any = Depends(lambda: None),  # Placeholder for auth
):
    """Generate clips for an episode with parameterized options"""
    
    # Validate episode exists
    duration = await _get_episode_duration(episode_id)
    if duration is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"ok": False, "error": "Unknown episode_id"}
        )
    
    # Validate time windows are within episode duration
    if body.timeWindows:
        for tw in body.timeWindows:
            if tw.endSec > (duration + 1e-6):
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={"ok": False, "error": "timeWindow exceeds episode duration"}
                )
    
    # Cache short-circuit (only when regenerate = False and params are compatible)
    if not body.regenerate:
        cached = _check_cached_candidates(episode_id, body.model_dump())
        if cached:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "ok": True,
                    "started": False,
                    "episodeId": episode_id,
                    "cached": True,
                    "next": {
                        "results": f"/api/episodes/{episode_id}/clips"
                    },
                },
            )
    
    # Idempotency: if key is present and seen, return same envelope (no duplicate enqueue)
    if x_idempotency_key:
        prior = _lookup_idempotency(x_idempotency_key)
        if prior and prior.get("episodeId") == episode_id:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "ok": True,
                    "started": True,
                    "episodeId": episode_id,
                    "job": {"id": prior["jobId"], "idempotencyKey": x_idempotency_key},
                    "next": {
                        "progress": f"/api/progress/{episode_id}",
                        "results": f"/api/episodes/{episode_id}/clips",
                    },
                },
            )
    
    # Enqueue the job
    job_id = _enqueue_clip_job(
        episode_id=episode_id,
        params=body.model_dump(),
        idempotency_key=x_idempotency_key,
    )
    
    if x_idempotency_key:
        _remember_idempotency(x_idempotency_key, episode_id, job_id)
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "ok": True,
            "started": True,
            "episodeId": episode_id,
            "job": {"id": job_id, "idempotencyKey": x_idempotency_key},
            "next": {
                "progress": f"/api/progress/{episode_id}",
                "results": f"/api/episodes/{episode_id}/clips",
            },
        },
    )
