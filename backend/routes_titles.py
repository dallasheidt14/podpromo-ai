from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
from models_titles import TitleGenRequest, TitleGenResponse, TitleSetRequest
from services.titles_service import TitlesService
from services.title_service import generate_titles as advanced_generate_titles
from services.title_gen import normalize_platform

router = APIRouter(prefix="/api/clips")

@router.post("/{clip_id}/titles", response_model=TitleGenResponse)
def generate_titles(clip_id: str, body: TitleGenRequest, request: Request):
    svc = TitlesService()
    clip = svc.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "Clip not found")

    # Use the advanced title generator
    transcript = clip.get("transcript") or clip.get("text") or ""
    if not transcript:
        raise HTTPException(400, "Clip has no transcript")
    
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

@router.put("/{clip_id}/title", status_code=204)
def set_title(clip_id: str, body: TitleSetRequest):
    svc = TitlesService()
    if not svc.set_chosen_title(clip_id, body.platform, body.title):
        raise HTTPException(404, "Clip not found")
