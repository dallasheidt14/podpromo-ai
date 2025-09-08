from pydantic import BaseModel, Field
from typing import List, Dict, Literal

Platform = Literal["shorts", "tiktok", "reels", "youtube"]

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
