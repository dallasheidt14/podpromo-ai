"""
Secure download endpoints with signed URLs.
Replaces public static file mounts with authenticated, time-limited access.
"""

import os
import time
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from security.signed_urls import verify, sign_path, generate_signed_url
from security.auth import require_user_dev
from config.settings import OUTPUT_DIR, UPLOAD_DIR, SECRET_KEY
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["downloads"])

# Safe base directories
SAFE_BASES = {
    "uploads": Path(UPLOAD_DIR).resolve(),
    "clips": Path(OUTPUT_DIR).resolve(),
}

def _safe_join(base: Path, name: str) -> Path:
    """Safely join path components, preventing directory traversal."""
    p = (base / name).resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p

@router.get("/signed-download")
def get_signed_download(
    area: Literal["uploads", "clips"],
    name: str,
    ttl: int = Query(300, ge=60, le=3600, description="TTL in seconds (60-3600)"),
    user=Depends(require_user_dev),
):
    """
    Generate a signed download URL for a file.
    Requires authentication and returns a time-limited, tamper-proof URL.
    """
    if area not in SAFE_BASES:
        raise HTTPException(status_code=400, detail="Unknown area")
    
    # Validate file exists
    file_path = _safe_join(SAFE_BASES[area], name)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Generate signed URL
    download_path = f"/api/download/{area}/{name}"
    signed_data = generate_signed_url(SECRET_KEY, download_path, ttl)
    
    logger.info(f"Generated signed download for {area}/{name}, expires in {ttl}s")
    
    return signed_data

@router.get("/download/{area}/{name:path}")
def download_file(
    area: Literal["uploads", "clips"],
    name: str,
    exp: int = Query(..., description="Expiration timestamp"),
    sig: str = Query(..., description="HMAC signature"),
):
    """
    Download a file using a signed URL.
    Validates the signature and expiration before serving the file.
    """
    if area not in SAFE_BASES:
        raise HTTPException(status_code=404, detail="Not found")
    
    # Verify signature
    download_path = f"/api/download/{area}/{name}"
    if not verify(SECRET_KEY, download_path, exp, sig):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Link expired or invalid signature"
        )
    
    # Get file path
    file_path = _safe_join(SAFE_BASES[area], name)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type based on file extension
    content_type = "application/octet-stream"
    if file_path.suffix.lower() in [".mp4", ".mov", ".avi"]:
        content_type = "video/mp4"
    elif file_path.suffix.lower() in [".mp3", ".wav", ".m4a", ".aac", ".flac"]:
        content_type = "audio/mpeg"
    elif file_path.suffix.lower() in [".json"]:
        content_type = "application/json"
    
    logger.info(f"Serving file {area}/{name} via signed URL")
    
    return FileResponse(
        str(file_path),
        filename=file_path.name,
        media_type=content_type
    )
