"""
HMAC-based signed URLs for secure file downloads.
Provides time-limited, tamper-proof access to files.
"""

import hmac
import hashlib
import time
import base64
from typing import Optional


def _b64url(data: bytes) -> str:
    """Base64 URL-safe encoding without padding."""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def sign_path(secret: str, path: str, exp_ts: int) -> str:
    """Generate HMAC signature for a path with expiration timestamp."""
    msg = f"{path}|{exp_ts}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    return _b64url(sig)


def verify(secret: str, path: str, exp_ts: int, sig: str) -> bool:
    """Verify HMAC signature and check expiration."""
    if time.time() > exp_ts:
        return False
    try:
        raw_sig = base64.urlsafe_b64decode(sig + "==")
    except Exception:
        return False
    expected = hmac.new(secret.encode(), f"{path}|{exp_ts}".encode(), hashlib.sha256).digest()
    return hmac.compare_digest(raw_sig, expected)


def generate_signed_url(secret: str, path: str, ttl_seconds: int = 300) -> dict:
    """Generate a complete signed URL with expiration."""
    exp_ts = int(time.time()) + ttl_seconds
    sig = sign_path(secret, path, exp_ts)
    return {
        "url": f"{path}?exp={exp_ts}&sig={sig}",
        "expires_at": exp_ts,
        "expires_in": ttl_seconds
    }
