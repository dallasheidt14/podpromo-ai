# backend/services/titles_service.py
# DEPRECATED shim to maintain imports

from .title_service import generate_titles, normalize_platform
__all__ = ["generate_titles", "normalize_platform"]

# Legacy TitlesService class for backward compatibility
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import re
import pathlib
import time
import tempfile
import threading
import random
from datetime import datetime, timezone
from contextlib import contextmanager
from config.settings import UPLOAD_DIR, TITLES_INDEX_TTL_SEC
import os

logger = logging.getLogger(__name__)

# Regex to extract episode ID from clip ID (supports both UUID and test formats)
CLIP_ID_RE = re.compile(r"^clip_([0-9a-f\-]{36}|[a-z0-9\-]+)_")

# In-memory clip index with TTL (fast path)
_CLIP_INDEX = {}          # clip_id -> Path(clips.json)
_CLIP_INDEX_BUILT_AT = 0

def _build_clip_index(uploads_dir: pathlib.Path):
    """Build in-memory index of clip_id -> clips.json file path"""
    global _CLIP_INDEX, _CLIP_INDEX_BUILT_AT
    now = time.time()
    if _CLIP_INDEX and (now - _CLIP_INDEX_BUILT_AT) < TITLES_INDEX_TTL_SEC:
        return
    idx = {}
    for p in uploads_dir.glob("*/clips.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for c in data.get("clips", []):
                cid = c.get("id")
                if cid:
                    idx[cid] = p
        except Exception:
            continue
    _CLIP_INDEX = idx
    _CLIP_INDEX_BUILT_AT = now

@contextmanager
def _file_lock(lock_path: str, tries: int = 40, delay: float = 0.05):
    """Context manager for file-based locking on Windows"""
    for i in range(tries):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                yield
            finally:
                os.close(fd)
                os.unlink(lock_path)
            return
        except FileExistsError:
            time.sleep(delay)
    # proceed without lock after retries
    logger.warning(f"Could not acquire lock {lock_path} after {tries} tries, proceeding without lock")
    yield


def _atomic_write_json(out_path: str, data: Dict[str, Any]) -> None:
    """Windows-safe atomic JSON write with unique temp names and retries"""
    d = os.path.dirname(out_path)
    os.makedirs(d, exist_ok=True)

    # Unique temp name to avoid collisions
    tmp_path = f"{out_path}.{os.getpid()}.{threading.get_ident()}.{int(time.time()*1e6)}.tmp"
    
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Windows may hold a handle briefly; retry replace on WinError 32
        for i in range(8):
            try:
                os.replace(tmp_path, out_path)  # atomic when it succeeds
                return
            except PermissionError as e:
                # [WinError 32] - file is being used by another process
                time.sleep(0.05 * (i + 1) + random.random() * 0.02)
        
        # Last resort: write to a sidecar to avoid temp leakage
        sidecar = f"{out_path}.race.{int(time.time()*1000)}.json"
        os.replace(tmp_path, sidecar)
        logger.warning(f"Could not replace {out_path}, saved to sidecar: {sidecar}")
        
    except Exception as e:
        logger.exception(f"Failed to write {out_path}: {e}")
        # Clean up temp file if it exists
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass


def _titles_path_for(clip_id: str) -> str:
    """Get the titles file path for a clip"""
    try:
        # clip_<episode_uuid>_<index> format
        parts = clip_id.split("_", 2)
        if len(parts) >= 2:
            episode_id = parts[1]
            episode_dir = os.path.join(UPLOAD_DIR, episode_id)
            titles_dir = os.path.join(episode_dir, "titles")
            return os.path.join(titles_dir, f"{clip_id}.json")
        else:
            raise ValueError(f"Invalid clip_id format: {clip_id}")
    except Exception as e:
        logger.error(f"Failed to get titles path for {clip_id}: {e}")
        raise


def _load_clip(clip_id: str, uploads_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load real clip data from clips.json file"""
    m = CLIP_ID_RE.match(clip_id)
    if not m:
        logger.warning("Invalid clip_id format: %s", clip_id)
        return None
    
    episode_id = m.group(1)
    clips_file = uploads_dir / episode_id / "clips.json"
    
    if not clips_file.exists():
        logger.warning("clips.json missing for episode %s", episode_id)
        return None
    
    try:
        data = json.loads(clips_file.read_text(encoding="utf-8"))
        
        # Handle both formats: direct array or wrapped in object
        if isinstance(data, list):
            clips = data
        else:
            clips = data.get("clips", [])
        
        # Find the specific clip
        for clip in clips:
            if clip.get("id") == clip_id:
                return clip
        
        logger.warning("Clip %s not found in clips.json", clip_id)
        return None
        
    except Exception as e:
        logger.error("Failed to load clips.json for episode %s: %s", episode_id, e)
        return None

class TitlesService:
    """Service class for managing clip titles - provides backward compatibility"""
    
    def __init__(self):
        self.clips_cache = {}  # Simple in-memory cache
    
    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get clip data by ID - loads real data from clips.json"""
        uploads_dir = pathlib.Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))
        
        # Try to load real clip data
        clip = _load_clip(clip_id, uploads_dir)
        if clip:
            logger.info(f"TitlesService.get_clip({clip_id}) - loaded real clip data")
            return clip
        
        # Fallback to minimal data if clip not found
        logger.warning(f"TitlesService.get_clip({clip_id}) - clip not found, using minimal fallback")
        return {
            "id": clip_id,
            "text": "",
            "transcript": "",
            "start": 0.0,
            "end": 0.0
        }
    
    def generate_variants(self, clip: Dict[str, Any], body: Any) -> Tuple[List[str], str, Dict[str, Any]]:
        """Generate title variants using legacy method"""
        text = clip.get("transcript") or clip.get("text") or ""
        platform = getattr(body, 'platform', 'default')
        
        # Use new generator as fallback
        titles = generate_titles(text, platform=platform, n=6)
        variants = [t["title"] for t in titles]
        chosen = variants[0] if variants else "Most Leaders Solve the Wrong Problem"
        
        meta = {
            "generator": "legacy_fallback",
            "version": 1,
            "generated_at": "2024-01-01T00:00:00Z",
        }
        
        return variants, chosen, meta
    
    def _read_titles_with_compat(self, titles_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Read titles entry with backward compatibility for missing fields"""
        # Ensure backward compatibility for old entries
        titles_entry.setdefault("meta", {})
        titles_entry.setdefault("generated_at", None)
        titles_entry.setdefault("engine", "v1")  # Default for old entries
        return titles_entry
    
    def save_titles(
        self,
        clip_id: str,
        platform: str,
        variants: List[str],
        chosen: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Persist titles into uploads/<episode_id>/clips.json under the matching clip entry.

        Structure added:
          clip["titles"][<platform>] = {
              "variants": [...],
              "chosen": "<string or None>",
              "generated_at": iso8601,
              "engine": "v2",
              "meta": {...}
          }
        """
        uploads_dir = pathlib.Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Validate variants before persist
        if not variants:
            logger.warning("TitlesService.save_titles: empty variants for %s; refusing to persist", clip_id)
            return False
        
        # Validate chosen index if provided
        if chosen is not None:
            if isinstance(chosen, str):
                try:
                    chosen = variants.index(chosen)
                except ValueError:
                    logger.warning("TitlesService.save_titles: chosen string '%s' not found in variants; resetting to None", chosen)
                    chosen = None
            elif isinstance(chosen, int):
                if not (0 <= chosen < len(variants)):
                    logger.warning("TitlesService.save_titles: chosen index %d out of range [0, %d); resetting to None", chosen, len(variants))
                    chosen = None

        # Build clip index for fast lookup
        _build_clip_index(uploads_dir)
        target_file = _CLIP_INDEX.get(clip_id)
        
        # Log path resolution for debugging
        logger.debug(f"TitlesService.save_titles: looking for clip_id={clip_id} in uploads_dir={uploads_dir}")
        if target_file:
            logger.debug(f"TitlesService.save_titles: found target_file={target_file}")
        else:
            logger.debug(f"TitlesService.save_titles: clip_id={clip_id} not in index, will scan")
        
        if not target_file:
            # Slow path: one scan, then update index for next time
            for p in uploads_dir.glob("*/clips.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for c in data.get("clips", []):
                        if c.get("id") == clip_id:
                            target_file = p
                            _CLIP_INDEX[clip_id] = p
                            break
                    if target_file:
                        break
                except Exception:
                    continue

        if not target_file:
            # Fallback: create dedicated titles file when clip not found in index
            logger.warning("TitlesService.save_titles: clip_id=%s not found in uploads; creating fallback titles file", clip_id)
            return self._save_titles_fallback(clip_id, platform, variants, chosen, meta, uploads_dir)

        # Load, mutate, write atomically
        try:
            data = json.loads(target_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("TitlesService.save_titles: failed to read %s: %s", target_file, e)
            return False

        updated = False
        now = datetime.now(timezone.utc).isoformat()
        for c in data.get("clips", []):
            if c.get("id") != clip_id:
                continue
            titles_obj = c.get("titles") or {}
            titles_obj[platform] = {
                "variants": variants,
                "chosen": chosen if chosen else (variants[0] if variants else None),
                "generated_at": now,
                "engine": "v2",
                "meta": meta or {},
            }
            c["titles"] = titles_obj
            updated = True
            break

        if not updated:
            logger.warning("TitlesService.save_titles: clip_id=%s not present in %s after reload", clip_id, target_file)
            return False

        # Use Windows-safe atomic write with lock file
        lock_path = str(target_file) + ".lock"
        try:
            with _file_lock(lock_path):
                _atomic_write_json(str(target_file), data)
            logger.info("TitlesService.save_titles: persisted titles for %s -> %s", clip_id, target_file)
            return True
        except Exception as e:
            logger.exception("TitlesService.save_titles: failed to write %s: %s", target_file, e)
            return False
    
    def _episode_dir_from_clip_id(self, clip_id: str) -> str:
        """Extract episode directory from clip_id with better error handling"""
        try:
            # clip_<episode_uuid>_<index> format
            parts = clip_id.split("_", 2)
            if len(parts) >= 2:
                episode_id = parts[1]
                return os.path.join(UPLOAD_DIR, episode_id)
            else:
                logger.error(f"Bad clip_id format: {clip_id}")
                raise ValueError(f"Invalid clip_id format: {clip_id}")
        except Exception as e:
            logger.error(f"Failed to extract episode dir from clip_id {clip_id}: {e}")
            raise

    def _save_titles_fallback(self, clip_id: str, platform: str, variants: List[str], 
                            chosen: Optional[str], meta: Optional[Dict[str, Any]], 
                            uploads_dir: pathlib.Path) -> bool:
        """Save titles to a dedicated file when clip not found in index"""
        try:
            # Get the titles file path
            out_path = _titles_path_for(clip_id)
            
            titles_data = {
                "clip_id": clip_id,
                "platform": platform,
                "variants": variants,
                "chosen": chosen if chosen else (variants[0] if variants else None),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "engine": "v2",
                "meta": meta or {},
            }
            
            # Use lock file for mutual exclusion
            lock_path = out_path + ".lock"
            with _file_lock(lock_path):
                _atomic_write_json(out_path, titles_data)
            
            logger.info(f"TitlesService.save_titles: saved titles to fallback file {out_path}")
            return True
            
        except Exception as e:
            logger.exception(f"TitlesService.save_titles: failed to save fallback titles for {clip_id}: {e}")
            return False

    def set_chosen_title(self, clip_id: str, platform: str, title: str) -> bool:
        """Set the chosen title for a clip"""
        logger.info(f"TitlesService.set_chosen_title({clip_id}, {platform}, {title})")
        # In a real app, this would update the database
        return True