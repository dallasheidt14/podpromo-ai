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
import time as _time
import tempfile
import threading
import random
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import lru_cache
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
    now = _time.time()
    if _CLIP_INDEX and (now - _CLIP_INDEX_BUILT_AT) < TITLES_INDEX_TTL_SEC:
        return
    idx = {}
    for p in uploads_dir.glob("*/clips.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Handle both formats: direct array or wrapped in object
            if isinstance(data, list):
                clips = data
            else:
                clips = data.get("clips", [])
            for c in clips:
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
            _time.sleep(delay)
    # proceed without lock after retries
    logger.warning(f"Could not acquire lock {lock_path} after {tries} tries, proceeding without lock")
    yield


def _atomic_write_json(out_path: str, data: Dict[str, Any]) -> None:
    """Windows-safe atomic JSON write with unique temp names and retries"""
    d = os.path.dirname(out_path)
    os.makedirs(d, exist_ok=True)

    # Unique temp name to avoid collisions
    tmp_path = f"{out_path}.{os.getpid()}.{threading.get_ident()}.{int(_time.time()*1e6)}.tmp"
    
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
                _time.sleep(0.05 * (i + 1) + random.random() * 0.02)
        
        # Last resort: write to a sidecar to avoid temp leakage
        sidecar = f"{out_path}.race.{int(_time.time()*1000)}.json"
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


def _episode_id_from_clip_id(clip_id: str) -> str:
    """Extract episode ID from clip ID using robust parsing."""
    # 'clip_' prefix sanity check
    if not clip_id.startswith("clip_"):
        raise ValueError(f"Unexpected clip_id format: {clip_id}")
    # split from the right: last '_' separates index
    base, _, _ = clip_id.rpartition("_")
    if not base:
        raise ValueError(f"Missing index in clip_id: {clip_id}")
    # remove 'clip_' prefix to get episode_id
    if not base.startswith("clip_"):
        raise ValueError(f"Unexpected base in clip_id: {clip_id}")
    return base[len("clip_"):]


def _titles_path_for(clip_id: str) -> str:
    """Get the titles file path for a clip - consistent with clips.json location"""
    try:
        episode_id = _episode_id_from_clip_id(clip_id)
        # Use same root as clips.json: uploads/<episode_id>/titles/
        episode_dir = os.path.join(UPLOAD_DIR, episode_id)
        titles_dir = os.path.join(episode_dir, "titles")
        os.makedirs(titles_dir, exist_ok=True)  # Ensure directory exists
        return os.path.join(titles_dir, f"{clip_id}.json")
    except Exception as e:
        logger.error(f"Failed to get titles path for {clip_id}: {e}")
        raise


def _read_or_migrate_clips_json(target_file: pathlib.Path) -> Dict[str, Any]:
    """
    Read clips.json with self-healing migration from legacy list format to object format.
    Returns dict with 'clips' key and 'version' field.
    """
    try:
        raw = json.loads(target_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("TitlesService: unreadable JSON at %s", target_file)
        raise

    # migrate legacy list -> object
    if isinstance(raw, list):
        logger.warning("TitlesService: migrating legacy list clips.json at %s", target_file)
        data = {"version": 2, "clips": raw}
        try:
            _atomic_write_json(str(target_file), data)
        except Exception as e:
            logger.warning("TitlesService: migration write failed, using in-memory data: %s", e)
        return data

    if not isinstance(raw, dict):
        raise ValueError("clips.json not a dict after read")

    if "clips" not in raw or not isinstance(raw["clips"], list):
        raw["clips"] = []
    
    # Ensure version field exists and is at least 2
    if "version" not in raw or raw["version"] < 2:
        raw["version"] = 2
        try:
            _atomic_write_json(str(target_file), raw)
        except Exception as e:
            logger.warning("TitlesService: version update write failed: %s", e)
    
    return raw

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
        data = _read_or_migrate_clips_json(clips_file)
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
        self._skip_warnings = set()  # Track warned clip_ids to prevent spam
        self._skip_lock = threading.Lock()  # Thread safety for skip warnings
    
    # In-process debounce cache
    _debounce: Dict[str, Tuple[float, str]] = {}
    _debounce_window_sec = 60.0
    
    @lru_cache(maxsize=256)
    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get clip data by ID - loads real data from clips.json (cached)"""
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
        
        # Debounce duplicate bodies for 60s
        import hashlib
        body_hash = hashlib.sha1("|".join([platform, chosen or "", *variants]).encode("utf-8")).hexdigest()
        last = self._debounce.get(clip_id)
        now = _time.time()
        if last and last[1] == body_hash and (now - last[0]) < self._debounce_window_sec:
            logger.info("TitlesService.save_titles: debounced duplicate for %s", clip_id)
            return True

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

        # Parse episode ID early for consistent path resolution
        try:
            episode_id = _episode_id_from_clip_id(clip_id)
            episode_dir = uploads_dir / episode_id
            clips_json_path = episode_dir / "clips.json"
        except Exception as e:
            logger.error(f"TitlesService.save_titles: failed to parse episode_id from clip_id {clip_id}: {e}")
            return self._save_titles_fallback(clip_id, platform, variants, chosen, meta, uploads_dir)
        
        # Try to find the clip in the expected location first
        target_file = None
        if clips_json_path.exists():
            try:
                data = json.loads(clips_json_path.read_text(encoding="utf-8"))
                # Handle both formats: direct array or wrapped in object
                if isinstance(data, list):
                    clips = data
                else:
                    clips = data.get("clips", [])
                for c in clips:
                    if c.get("id") == clip_id:
                        target_file = clips_json_path
                        _CLIP_INDEX[clip_id] = clips_json_path
                        break
            except Exception as e:
                logger.warning(f"TitlesService.save_titles: failed to read {clips_json_path}: {e}")
        
        # If not found in expected location, try the global index
        if not target_file:
            _build_clip_index(uploads_dir)
            target_file = _CLIP_INDEX.get(clip_id)
            
            # Log path resolution for debugging
            logger.debug(f"TitlesService.save_titles: looking for clip_id={clip_id} in uploads_dir={uploads_dir}")
            if target_file:
                logger.debug(f"TitlesService.save_titles: found target_file={target_file}")
            else:
                logger.debug(f"TitlesService.save_titles: clip_id={clip_id} not in index, will scan")
        
        # If still not found, do a broader scan with retry
        if not target_file:
            for attempt in range(3):  # Try up to 3 times with small delay
                for p in uploads_dir.glob("*/clips.json"):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        # Handle both formats: direct array or wrapped in object
                        if isinstance(data, list):
                            clips = data
                        else:
                            clips = data.get("clips", [])
                        for c in clips:
                            if c.get("id") == clip_id:
                                target_file = p
                                _CLIP_INDEX[clip_id] = p
                                break
                        if target_file:
                            break
                    except Exception:
                        continue
                if target_file:
                    break
                if attempt < 2:  # Don't sleep on last attempt
                    _time.sleep(0.1)  # Small delay for race conditions

        if not target_file:
            # Fallback: create dedicated titles file when clip not found in index
            logger.warning("TitlesService.save_titles: clip_id=%s not found in uploads; creating fallback titles file", clip_id)
            return self._save_titles_fallback(clip_id, platform, variants, chosen, meta, uploads_dir)

        # Load, mutate, write atomically with self-healing migration
        try:
            data = _read_or_migrate_clips_json(target_file)
        except Exception as e:
            logger.exception("TitlesService.save_titles: failed to read %s: %s", target_file, e)
            return False

        # Extract clips from migrated data
        clips = data.get("clips", [])

        updated = False
        now = datetime.now(timezone.utc).isoformat()
        for c in clips:
            if c.get("id") != clip_id:
                continue
            titles_obj = c.get("titles") or {}
            
            # Convert old variants format to new schema if needed
            if isinstance(variants, list) and variants and isinstance(variants[0], str):
                # Old format: convert to new schema
                from .title_service import generate_title_pack
                text = c.get("transcript") or c.get("text") or ""
                title_pack = generate_title_pack(text=text, platform=platform)
                titles_obj[platform] = title_pack
                if chosen:
                    titles_obj[platform]["chosen"] = chosen
                    titles_obj[platform]["chosen_at"] = now
            else:
                # New format: use as-is
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

        # Always save in new object format
        save_data = data
        save_data["clips"] = clips

        # Use Windows-safe atomic write with lock file
        lock_path = str(target_file) + ".lock"
        try:
            with _file_lock(lock_path):
                _atomic_write_json(str(target_file), save_data)
            logger.info("TitlesService.save_titles: persisted titles for %s -> %s", clip_id, target_file)
            
            # Write-through to dedicated per-clip file (Windows-safe)
            try:
                episode_id = _episode_id_from_clip_id(clip_id)
                titles_dir = uploads_dir / episode_id / "titles"
                titles_dir.mkdir(parents=True, exist_ok=True)
                out_path = titles_dir / f"{clip_id}.json"
                
                with _file_lock(str(out_path) + ".lock"):
                    _atomic_write_json(str(out_path), {
                        "clip_id": clip_id,
                        "platform": platform,
                        "variants": variants,
                        "chosen": chosen if chosen else (variants[0] if variants else None),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "engine": "v2",
                        "meta": meta or {},
                    })
                logger.debug("TitlesService.save_titles: write-through completed for %s", clip_id)
            except Exception as e:
                logger.warning("TitlesService.save_titles: write-through failed for %s: %s", clip_id, e)
            
            # Update debounce cache (use timestamp, not datetime string)
            self._debounce[clip_id] = (_time.time(), body_hash)
            return True
        except Exception as e:
            logger.exception("TitlesService.save_titles: failed to write %s: %s", target_file, e)
            return False
    
    def _episode_dir_from_clip_id(self, clip_id: str) -> str:
        """Extract episode directory from clip_id with better error handling"""
        try:
            # clip_<episode_uuid>_<index> format
            episode_id = _episode_id_from_clip_id(clip_id)
            return os.path.join(UPLOAD_DIR, episode_id)
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
            
            # Ensure directory exists before creating lock file
            titles_dir = os.path.dirname(out_path)
            os.makedirs(titles_dir, exist_ok=True)
            
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
        """Set the chosen title for a clip and persist it to clips.json"""
        uploads_dir = pathlib.Path(os.getenv("UPLOADS_DIR", UPLOAD_DIR))
        
        try:
            episode_id = _episode_id_from_clip_id(clip_id)
            target_file = uploads_dir / episode_id / "clips.json"
        except Exception as e:
            logger.error(f"TitlesService.set_chosen_title: failed to parse episode_id from clip_id {clip_id}: {e}")
            return False

        if not target_file.exists():
            logger.warning("TitlesService.set_chosen_title: clips.json not found for %s", clip_id)
            return False

        try:
            data = _read_or_migrate_clips_json(target_file)
        except Exception as e:
            logger.exception("TitlesService.set_chosen_title: failed to read %s: %s", target_file, e)
            return False

        now = datetime.now(timezone.utc).isoformat()
        updated = False
        
        for c in data["clips"]:
            if c.get("id") == clip_id:
                titles = c.setdefault("titles", {})
                entry = titles.setdefault(platform, {
                    "variants": [], 
                    "chosen": None, 
                    "engine": "v2", 
                    "generated_at": now,
                    "meta": {}
                })
                entry["chosen"] = title
                entry["chosen_at"] = now
                
                # Update analytics placeholder
                meta = entry.setdefault("meta", {})
                analytics = meta.setdefault("analytics", {
                    "impressions": 0,
                    "clicks": 0,
                    "ctr": 0.0,
                    "last_updated": now
                })
                analytics["last_updated"] = now
                
                updated = True
                break

        if not updated:
            logger.warning("TitlesService.set_chosen_title: clip not found: %s", clip_id)
            return False

        # Write back atomically
        lock_path = str(target_file) + ".lock"
        try:
            with _file_lock(lock_path):
                _atomic_write_json(str(target_file), data)
            logger.info("TitlesService.set_chosen_title: persisted for %s/%s", clip_id, platform)
            return True
        except Exception as e:
            logger.exception("TitlesService.set_chosen_title: failed to write %s: %s", target_file, e)
            return False

    def ensure_titles_for_clip(self, clip_or_id, *, force=False, platform="shorts"):
        """
        Idempotent: if clip already has titles for this platform, do nothing.
        Otherwise, generate + persist.
        Accepts either a clip dict or a clip_id string.
        """
        # Normalize input: turn clip_or_id into a dict via get_clip() if it's a string
        clip = clip_or_id
        if isinstance(clip_or_id, str):
            clip = self.get_clip(clip_or_id)
        
        if not isinstance(clip, dict):
            # Throttle warnings to once per clip (thread-safe)
            with self._skip_lock:
                if clip_or_id not in self._skip_warnings:
                    logger.warning("AUTO_TITLE: load_failed %r", clip_or_id)
                    self._skip_warnings.add(clip_or_id)
                else:
                    logger.info("AUTO_TITLE: throttled load_failed %r", clip_or_id)
            return None  # skip cleanly

        cid = clip.get("id")
        if not cid:
            # Throttle warnings to once per clip (thread-safe)
            with self._skip_lock:
                if clip_or_id not in self._skip_warnings:
                    logger.warning("AUTO_TITLE: no_id %r", clip_or_id)
                    self._skip_warnings.add(clip_or_id)
                else:
                    logger.info("AUTO_TITLE: throttled no_id %r", clip_or_id)
            return None

        # Skip if already generated for this platform (unless force=True)
        if not force and clip.get("titles", {}).get(platform):
            return clip["titles"][platform]

        # Build prompt inputs robustly
        text = clip.get("text") or ""
        if not text and isinstance(clip.get("transcript"), dict):
            text = clip.get("transcript", {}).get("text", "")
        elif not text and isinstance(clip.get("transcript"), str):
            text = clip.get("transcript", "")
        
        if not text:
            # Fallback: try to extract from words
            words = clip.get("words") or []
            if isinstance(words, list) and words:
                text = " ".join(w.get("word", "") for w in words if isinstance(w, dict) and w.get("word"))
        
        if not text:
            # Throttle warnings to once per clip (thread-safe)
            with self._skip_lock:
                if cid not in self._skip_warnings:
                    logger.warning("AUTO_TITLE: no_text %s", cid)
                    self._skip_warnings.add(cid)
                else:
                    logger.info("AUTO_TITLE: throttled no_text %s", cid)
            return None

        # Create snippet for title generation (cheap, safe)
        words = text.split()
        snippet = " ".join(words[:80])  # First 80 words
        
        # Normalize whitespace and trim trailing punctuation
        snippet = re.sub(r'\s+', ' ', snippet.strip())  # Normalize whitespace
        snippet = re.sub(r'[:;\.]{2,}$', '.', snippet)  # Remove multiple colons/semicolons/periods
        snippet = re.sub(r'\.{3,}$', '...', snippet)    # Normalize ellipses
        snippet = snippet.rstrip('.,;:')                # Remove trailing punctuation
        
        try:
            # Check for ad content
            features = clip.get("features", {})
            is_ad = features.get("is_advertisement", False)
            if is_ad:
                logger.info("AUTO_TITLE: skipping ad content %s", cid)
                return None
            
            # Get language hint if available
            language = features.get("language", "en")
            
            # Generate title pack using v1 API
            from .title_service import generate_title_pack
            pack = generate_title_pack(text=snippet, platform=platform)
            
            # Extract variants from the pack with fallback
            variants = [v["title"] for v in pack.get("variants", [])] if pack.get("variants") else []
            chosen = pack.get("overlay", "")
            
            # Fallback if no variants generated
            if not variants:
                logger.warning("AUTO_TITLE: no variants generated for %s", cid)
                variants = [snippet[:50] + "..." if len(snippet) > 50 else snippet]  # Use snippet as fallback
                chosen = variants[0]
            
            # Save titles
            self.save_titles(clip_id=cid, platform=platform, variants=variants, chosen=chosen, meta={"autogen": True})
            
            return {
                "variants": variants,
                "chosen": chosen,
                "overlay": chosen,
                "meta": {"autogen": True}
            }
        except Exception as e:
            # Throttle warnings to once per clip (thread-safe)
            with self._skip_lock:
                if cid not in self._skip_warnings:
                    logger.warning("AUTO_TITLE: generation_failed %s (%s)", cid, e)
                    self._skip_warnings.add(cid)
                else:
                    logger.info("AUTO_TITLE: throttled generation_failed %s", cid)
            return None

    def ensure_titles_for_clip_legacy(self, episode: dict, clip: dict, platform: str = "shorts") -> None:
        """
        Deprecated: Legacy wrapper for backward compatibility.
        TODO: Remove once all callers pass ids or dicts intentionally.
        """
        return self.ensure_titles_for_clip(clip, platform=platform)