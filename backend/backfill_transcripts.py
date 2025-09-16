#!/usr/bin/env python3
"""
One-time backfill script to add full transcripts to existing clips
"""
import json
import os
import glob
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOADS_DIR = r"C:\Users\Dallas Heidt\Desktop\podpromo\backend\uploads"

def load_episode_payload(folder):
    """Load episode data from various possible files in the folder"""
    # Try common filenames your service might have used
    candidates = [
        "episode.json", "transcript.json",
        "segments.json", "words.json"
    ]
    payload = {"words": [], "segments": []}
    
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            try:
                data = json.load(open(p, "r", encoding="utf-8"))
                # tolerate different shapes
                if isinstance(data, dict):
                    payload["words"] += data.get("words", [])
                    payload["segments"] += data.get("segments", data.get("items", []))
                elif isinstance(data, list):
                    # guess by shape
                    if data and "start" in data[0] and "text" in data[0]:
                        # unknown whether words or segments; keep in both
                        payload["words"] += data
                        payload["segments"] += data
            except Exception as e:
                logger.debug(f"Failed to load {p}: {e}")
                pass
    
    return payload

def build_clip_transcript_from_words(words, start, end):
    """Wrapper function for backfill script that takes words directly"""
    from services.transcript_builder import build_clip_transcript_exact
    text, source, meta = build_clip_transcript_exact(words, start, end)
    return text

def backfill_episode(ep_folder):
    """Backfill transcripts for all clips in an episode folder"""
    clips_path = os.path.join(ep_folder, "clips.json")
    if not os.path.exists(clips_path):
        return 0, 0

    try:
        clips = json.load(open(clips_path, "r", encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load clips from {clips_path}: {e}")
        return 0, 0

    payload = load_episode_payload(ep_folder)
    words = payload.get("words", [])

    updated = 0
    for c in clips:
        # Skip if already has transcript
        if c.get("transcript"):
            continue
            
        start = c.get("start")
        end = c.get("end")
        
        # Skip clips with invalid timing
        if start is None or end is None:
            logger.warning(f"Skipping clip {c.get('id', 'unknown')} with invalid timing: start={start}, end={end}")
            continue
            
        start = float(start)
        end = float(end)
        txt = build_clip_transcript_from_words(words, start, end)
        
        c["transcript"] = txt
        c["transcript_source"] = "word_slice"
        c["transcript_char_count"] = len(txt)
        updated += 1

    if updated:
        try:
            with open(clips_path, "w", encoding="utf-8") as f:
                json.dump(clips, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated {updated} clip(s) in {os.path.basename(ep_folder)}")
        except Exception as e:
            logger.error(f"Failed to save clips to {clips_path}: {e}")
            return len(clips), 0

    return len(clips), updated

def main():
    """Main backfill function"""
    total_clips = total_updated = 0
    
    if not os.path.exists(UPLOADS_DIR):
        logger.error(f"Uploads directory not found: {UPLOADS_DIR}")
        return
    
    logger.info(f"Starting backfill in {UPLOADS_DIR}")
    
    for ep_folder in glob.glob(os.path.join(UPLOADS_DIR, "*")):
        if not os.path.isdir(ep_folder):
            continue
            
        clips_count, updated = backfill_episode(ep_folder)
        total_clips += clips_count
        total_updated += updated
    
    logger.info(f"Backfill complete. {total_updated} / {total_clips} clips updated.")

if __name__ == "__main__":
    main()
