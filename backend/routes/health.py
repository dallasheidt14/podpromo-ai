"""
Ultra-fast health check - liveness only, never touches disk/network
"""
from fastapi import APIRouter, Response

router = APIRouter()

# Pre-baked response bytes for zero JSON encoding overhead
_HEALTH_BYTES = b'{"ok":true}'

@router.get("/health")
def health():
    """Liveness check - sub-millisecond, pre-baked response"""
    return Response(
        content=_HEALTH_BYTES,
        media_type="application/json",
        headers={"Cache-Control": "no-store"}
    )
