"""
Secure preview serving with signed URLs and Range support for video seeking
"""
import base64
import hashlib
import hmac
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from config.settings import PREVIEW_FS_DIR, PREVIEW_SIGNING_KEY, PREVIEW_URL_TTL_SECONDS

router = APIRouter(prefix="/api/previews", tags=["previews"])

def _b64u(x: bytes) -> str:
    """Base64 URL-safe encode without padding"""
    return base64.urlsafe_b64encode(x).decode().rstrip("=")

def _b64u_dec(s: str) -> bytes:
    """Base64 URL-safe decode with padding restoration"""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def sign_preview_name(name: str, *, ttl_s: int | None = None) -> str:
    """Generate a signed URL for a preview file"""
    ttl = ttl_s or PREVIEW_URL_TTL_SECONDS
    exp = int(time.time()) + ttl
    msg = f"{name}|{exp}".encode()
    sig = hmac.new(PREVIEW_SIGNING_KEY.encode(), msg, hashlib.sha256).digest()
    return f"/api/previews/get?name={quote(name)}&exp={exp}&sig={_b64u(sig)}"

@router.get("/sign")
def sign_endpoint(name: str, ttl: int | None = None):
    """Sign a preview filename and return a time-limited URL"""
    # name must be relative, e.g. "filename.mp4"
    name = unquote(name)
    if name.startswith("/") or ".." in name or "/" in name:
        raise HTTPException(400, "invalid name - must be filename only")
    return {"url": sign_preview_name(name, ttl_s=ttl)}

def _fs_path_for(name: str) -> Path:
    """Get filesystem path for a preview filename (with security checks)"""
    # Ensure name is just a filename, no path traversal
    filename = Path(name).name
    p = (Path(PREVIEW_FS_DIR) / filename).resolve()
    
    # Security: ensure the resolved path is within PREVIEW_FS_DIR
    if not str(p).startswith(str(Path(PREVIEW_FS_DIR).resolve())):
        raise HTTPException(400, "invalid path")
    return p

def _verify(name: str, exp: int, sig: str) -> Path:
    """Verify signature and return filesystem path if valid"""
    if exp < int(time.time()):
        raise HTTPException(403, "expired")
    
    msg = f"{name}|{exp}".encode()
    expected = hmac.new(PREVIEW_SIGNING_KEY.encode(), msg, hashlib.sha256).digest()
    
    try:
        given = _b64u_dec(sig)
    except Exception:
        raise HTTPException(400, "bad signature")
    
    if not hmac.compare_digest(expected, given):
        raise HTTPException(403, "signature mismatch")
    
    return _fs_path_for(name)

def iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 512):
    """Stream file content in chunks for Range requests"""
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            yield chunk
            remaining -= len(chunk)

@router.get("/get")
def get_preview(request: Request, name: str, exp: int, sig: str):
    """Serve a preview file with Range support for video seeking"""
    name = unquote(name)
    path = _verify(name, exp, sig)
    
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")

    file_size = path.stat().st_size
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if range_header:
        # Handle Range requests (for video seeking)
        try:
            _, rng = range_header.split("=")
            start_str, end_str = rng.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            start = max(0, start)
            end = min(end, file_size - 1)
            if start > end:
                raise ValueError
        except Exception:
            raise HTTPException(416, "invalid range")
        
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(end - start + 1),
            "Cache-Control": "private, max-age=60",
        }
        return StreamingResponse(
            iter_file_range(path, start, end), 
            status_code=206, 
            media_type=ctype, 
            headers=headers
        )

    # Full file request
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Cache-Control": "private, max-age=60",
    }
    return StreamingResponse(
        iter_file_range(path, 0, file_size - 1), 
        media_type=ctype, 
        headers=headers
    )
