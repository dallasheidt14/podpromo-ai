"""
Secure upload endpoints with strict validation.
Enforces MIME types, file extensions, and size limits.
"""

import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from security.auth import require_user_dev
from config.settings import UPLOAD_DIR
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/secure", tags=["secure-uploads"])

# Strict validation rules
ALLOWED_MIME_TYPES = {
    "audio/mpeg",      # MP3
    "audio/wav",       # WAV
    "audio/x-wav",     # WAV (alternative)
    "audio/aac",       # AAC
    "audio/flac",      # FLAC
    "audio/mp4",       # M4A
    "audio/m4a",       # M4A (alternative)
    "video/mp4",       # MP4
}

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".aac", ".flac", ".mp4", ".m4a"
}

# Get max file size from environment (default 500MB)
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", "524288000"))

def _validate_file_extension(filename: str) -> bool:
    """Validate file extension against allowlist."""
    if not filename:
        return False
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def _validate_mime_type(content_type: str) -> bool:
    """Validate MIME type against allowlist."""
    if not content_type:
        return False
    return content_type.lower() in ALLOWED_MIME_TYPES

async def _validate_file_size(file: UploadFile) -> None:
    """Validate file size by streaming and counting bytes."""
    bytes_read = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    # Reset file pointer
    await file.seek(0)
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_BYTES} bytes ({MAX_FILE_BYTES // 1024 // 1024}MB)"
            )
    
    # Reset file pointer for actual processing
    await file.seek(0)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user=Depends(require_user_dev),
):
    """
    Upload a file with strict validation.
    Validates MIME type, file extension, and size before processing.
    """
    logger.info(f"Upload attempt: {file.filename} ({file.content_type})")
    
    # 1. Validate file extension
    if not _validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # 2. Validate MIME type
    if not _validate_mime_type(file.content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported MIME type: {file.content_type}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    # 3. Validate file size
    await _validate_file_size(file)
    
    # 4. Create safe filename (prevent path traversal)
    safe_filename = Path(file.filename).name  # Remove any path components
    if not safe_filename or safe_filename in [".", ".."]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )
    
    # 5. Save file to upload directory
    upload_path = Path(UPLOAD_DIR) / safe_filename
    try:
        with open(upload_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to save file {safe_filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file"
        )
    
    logger.info(f"Successfully uploaded: {safe_filename}")
    
    return {
        "filename": safe_filename,
        "size_bytes": upload_path.stat().st_size,
        "content_type": file.content_type,
        "message": "File uploaded successfully"
    }

@router.get("/upload/validate")
async def validate_upload_rules():
    """Get current upload validation rules for frontend display."""
    return {
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "allowed_mime_types": list(ALLOWED_MIME_TYPES),
        "max_file_bytes": MAX_FILE_BYTES,
        "max_file_mb": MAX_FILE_BYTES // 1024 // 1024
    }
