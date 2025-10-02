"""
Safe I/O utilities for path sanitization and atomic writes.
Prevents path traversal attacks and ensures atomic file operations.
"""

import json
import os
import re
import tempfile
import logging

logger = logging.getLogger(__name__)

# Safe path segment pattern - only allow alphanumeric, dots, underscores, hyphens
SAFE_SEG = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_seg(s: str) -> str:
    """
    Sanitize a path segment to prevent directory traversal attacks.
    
    Args:
        s: Input string to sanitize
        
    Returns:
        Sanitized string safe for use in file paths
    """
    if not s:
        return "untitled"
    
    # Replace path separators with underscores
    s = s.strip().replace(os.path.sep, "_").replace(os.path.altsep or "", "_")
    
    # Remove any unsafe characters
    s = SAFE_SEG.sub("_", s)
    
    # Ensure we don't return empty string
    return s or "untitled"

def atomic_write_json(path: str, obj: dict, *, indent: int = 2) -> None:
    """
    Atomically write JSON data to a file.
    
    This prevents partial writes from crashes and ensures data integrity.
    
    Args:
        path: Target file path
        obj: Dictionary to serialize as JSON
        indent: JSON indentation level
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            "w", 
            delete=False, 
            dir=os.path.dirname(path), 
            encoding="utf-8"
        ) as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=indent)
            tmp.flush()
            os.fsync(tmp.fileno())  # Force write to disk
            tmp_path = tmp.name
        
        # Atomic move to final location
        os.replace(tmp_path, path)
        
    except Exception as e:
        # Clean up temp file on failure
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception as cleanup_e:
            logger.debug("IGNORED_ERROR[%s]: %s", cleanup_e.__class__.__name__, cleanup_e)
        
        logger.error(f"Failed to write JSON to {path}: {e}")
        raise

def atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """
    Atomically write text content to a file.
    
    Args:
        path: Target file path
        content: Text content to write
        encoding: Text encoding
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            "w", 
            delete=False, 
            dir=os.path.dirname(path), 
            encoding=encoding
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())  # Force write to disk
            tmp_path = tmp.name
        
        # Atomic move to final location
        os.replace(tmp_path, path)
        
    except Exception as e:
        # Clean up temp file on failure
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception as cleanup_e:
            logger.debug("IGNORED_ERROR[%s]: %s", cleanup_e.__class__.__name__, cleanup_e)
        
        logger.error(f"Failed to write text to {path}: {e}")
        raise
