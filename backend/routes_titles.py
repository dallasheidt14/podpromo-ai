from fastapi import APIRouter, HTTPException
from datetime import datetime
from models_titles import TitleGenRequest, TitleGenResponse, TitleSetRequest
from services.titles_service import TitlesService
from services.title_service import generate_titles as advanced_generate_titles

router = APIRouter(prefix="/api/clips")

@router.post("/{clip_id}/titles", response_model=TitleGenResponse)
def generate_titles(clip_id: str, body: TitleGenRequest):
    svc = TitlesService()
    clip = svc.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "Clip not found")

    # Use the advanced title generator
    transcript = clip.get("transcript") or clip.get("text") or ""
    if not transcript:
        raise HTTPException(400, "Clip has no transcript")
    
    # Extract features for better title generation
    features = {
        "hook_score": clip.get("hook_score", 0.0),
        "arousal_score": clip.get("arousal_score", 0.0),
        "final_score": clip.get("score", 0.0),
    }
    
    # Generate titles using the advanced service
    variants = advanced_generate_titles(
        transcript=transcript,
        features=features,
        max_len=80
    )
    
    if not variants:
        # Fallback to simple heuristic if advanced fails
        variants, chosen, meta = svc.generate_variants(clip, body)
    else:
        chosen = variants[0]
        meta = {
            "generator": "advanced",
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
