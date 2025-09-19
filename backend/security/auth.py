"""
JWT-based authentication for API endpoints.
Supports both Supabase Auth and custom JWT validation.
"""

import os
import jwt
from fastapi import Depends, HTTPException, status, Header
from typing import Optional, Dict, Any

# JWT Configuration
JWT_AUDIENCE = os.getenv("JWT_AUD", None)
JWT_ISSUER = os.getenv("JWT_ISS", None)
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", None)  # PEM string
JWT_ALGORITHMS = ["RS256", "ES256", "HS256"]


def require_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Require valid JWT bearer token for protected endpoints.
    Returns decoded JWT claims if valid.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ", 1)[1]
    
    try:
        decoded = jwt.decode(
            token,
            JWT_PUBLIC_KEY,
            algorithms=JWT_ALGORITHMS,
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat"]},
        )
        return decoded
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Invalid token: {str(e)}"
        )


def optional_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """
    Optional JWT validation - returns claims if valid, None if missing/invalid.
    Useful for endpoints that work with or without auth.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    
    token = authorization.split(" ", 1)[1]
    
    try:
        decoded = jwt.decode(
            token,
            JWT_PUBLIC_KEY,
            algorithms=JWT_ALGORITHMS,
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat"]},
        )
        return decoded
    except jwt.PyJWTError:
        return None


# For development/testing - bypass auth when JWT_PUBLIC_KEY is not set
def require_user_dev(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Development version that bypasses auth when JWT_PUBLIC_KEY is not configured."""
    if not JWT_PUBLIC_KEY:
        return {"user_id": "dev_user", "sub": "dev_user", "role": "admin"}
    return require_user(authorization)