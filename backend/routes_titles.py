from fastapi import APIRouter, HTTPException, Request, Body
from datetime import datetime
from models_titles import TitleGenRequest, TitleGenResponse, TitleSetRequest, SaveTitlesRequest
from services.titles_service import TitlesService
from services.title_service import generate_titles as advanced_generate_titles, _extract_text
from services.title_gen import normalize_platform

router = APIRouter(prefix="/api/clips")

@router.post("/{clip_id}/titles", response_model=TitleGenResponse)
def generate_titles(clip_id: str, body: TitleGenRequest, request: Request):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        svc = TitlesService()
        clip = svc.get_clip(clip_id)
        if not clip:
            logger.warning(f"TITLES_422: clip_id={clip_id} not found")
            raise HTTPException(404, "Clip not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TITLES_422: clip_id={clip_id} validation error: {e}")
        raise HTTPException(422, f"Validation error: {str(e)}")

    # Choose the best available text source and normalize to string
    transcript_obj = (
        (clip.get("transcript") or {}) if isinstance(clip, dict) else getattr(clip, "transcript", {})
    )
    transcript = _extract_text(transcript_obj)
    
    if not transcript.strip():
        # nothing usable; send client error instead of 500
        raise HTTPException(status_code=422, detail="clip_transcript_empty")
    
    # Check for rich API request
    rich = request.query_params.get("rich") in {"1", "true", "yes"}
    
    # Normalize platform
    platform = normalize_platform(body.platform)
    
    # Generate titles using the new unified generator
    titles = advanced_generate_titles(transcript, platform=platform, n=6)
    
    if not titles:
        # Fallback to simple heuristic if advanced fails
        variants, chosen, meta = svc.generate_variants(clip, body)
    else:
        if rich:
            # Rich response with scores and reasons
            chosen = titles[0]["title"]
            variants = [t["title"] for t in titles]
            meta = {
                "generator": "unified_v2",
                "version": 2,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "rich_data": titles  # Include full data with scores
            }
        else:
            # Legacy response format
            chosen = titles[0]["title"]
            variants = [t["title"] for t in titles]
            meta = {
                "generator": "unified_v2",
                "version": 2,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
    
    svc.save_titles(clip_id, body.platform, variants, chosen, meta)
    return TitleGenResponse(platform=body.platform, variants=variants, chosen=chosen, meta=meta)

@router.post("/{clip_id}/titles/save")
def save_titles(clip_id: str, req: SaveTitlesRequest = Body(...)):
    """Save titles for a clip with tolerant schema"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        svc = TitlesService()
        
        # Check if clip exists
        clip = svc.get_clip(clip_id)
        if not clip:
            logger.warning(f"SAVE_TITLES_422: clip_id={clip_id} not found")
            raise HTTPException(404, "Clip not found")
        
        # Log the request details
        logger.info(f"SAVE_TITLES: clip_id={clip_id}, platform={req.platform}, count={len(req.titles or [])}, first_title='{req.titles[0] if req.titles else 'none'}'")
        
        # De-dupe and validate titles
        titles = list(dict.fromkeys(req.titles))  # de-dupe, keep order
        if not titles:
            # Graceful app-level guard (return 200 with a helpful message instead of hard 422s)
            logger.warning(f"SAVE_TITLES_422: clip_id={clip_id} no titles provided")
            return {"ok": False, "reason": "no_titles", "message": "No titles provided"}
        
        # Save titles using the existing service
        svc.save_titles(clip_id, req.platform, titles, titles[0] if titles else "", {})
        logger.info(f"SAVE_TITLES_SUCCESS: clip_id={clip_id} saved {len(titles)} titles")
        return {"ok": True, "count": len(titles), "platform": req.platform}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SAVE_TITLES_422: clip_id={clip_id} error: {e}")
        raise HTTPException(422, f"Failed to save titles: {str(e)}")

@router.put("/{clip_id}/title", status_code=204)
def set_title(clip_id: str, body: TitleSetRequest):
    svc = TitlesService()
    if not svc.set_chosen_title(clip_id, body.platform, body.title):
        raise HTTPException(404, "Clip not found")
