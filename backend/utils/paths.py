"""Safe path utilities for file operations."""
from pathlib import Path

class PathTraversalError(Exception):
    """Raised when a path would escape the base directory."""
    pass

def safe_join(base: Path, *parts: str) -> Path:
    """Safely join paths, preventing directory traversal attacks."""
    base = base.resolve()
    p = base.joinpath(*parts).resolve()
    if not str(p).startswith(str(base)):
        raise PathTraversalError(f"Unsafe path outside base: {p}")
    return p

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
