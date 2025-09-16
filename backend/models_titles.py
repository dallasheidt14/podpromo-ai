from pydantic import BaseModel, Field, conlist, constr, validator
from typing import List, Dict, Literal, Optional

Platform = Literal["shorts", "tiktok", "reels", "youtube", "neutral"]

# Title string with length constraints
TitleStr = constr(min_length=3, max_length=96, strip_whitespace=True)

class TitleGenRequest(BaseModel):
    platform: Platform = "shorts"
    n: int = Field(default=6, ge=1, le=8)
    seed: int | None = None
    allow_emoji: bool = True

class TitleGenResponse(BaseModel):
    platform: Platform
    variants: List[str]
    chosen: str
    meta: Dict[str, str | int]

class TitleSetRequest(BaseModel):
    platform: Platform = "shorts"
    title: str

class SaveTitlesRequest(BaseModel):
    """Tolerant schema for saving titles - accepts either 'titles' or 'variants'"""
    # Accept either {"titles": [...]} or {"variants": [...]} from the client
    titles: Optional[List[TitleStr]] = Field(None, alias="variants")
    platform: Platform = "shorts"

    @validator("titles", pre=True, always=True)
    def coerce_titles(cls, v):
        # Support both 'titles' and 'variants' keys and fall back to empty list
        if not v:
            return []
        # Ensure it's a list and limit length
        if isinstance(v, list):
            return v[:12]  # max 12 titles
        return []

    @validator("titles", each_item=True)
    def sanitize_title(cls, t: str) -> str:
        # Flatten whitespace/newlines and trim
        return " ".join((t or "").replace("\n", " ").split())

    class Config:
        populate_by_name = True  # allow 'titles' or alias 'variants'
