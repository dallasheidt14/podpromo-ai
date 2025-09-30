#!/usr/bin/env python3
"""
Run the latest uploaded transcript through EpisodeService and print a brief summary.
"""

import os
import json
import glob
import asyncio
from pathlib import Path


async def main() -> int:
    base = Path(__file__).resolve().parent
    trans_dir = base / "uploads" / "transcripts"
    if not trans_dir.exists():
        print(f"❌ transcripts dir not found: {trans_dir}")
        return 1

    files = sorted(
        glob.glob(str(trans_dir / "*/words.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        print("❌ No transcript JSON files found in uploads/transcripts")
        return 1

    latest = files[0]
    print(f"Latest transcript: {latest}")

    # Load transcript JSON (robust to large files)
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try a known episode ID first, then fall back to directory name
    known_episodes = ["00a5f473-26f6-4f0b-83d0-9eb740087da0", "0420d390-ecd1-4aba-93a8-605191eb5ac2"]
    episode_id = known_episodes[0]  # Use the first known episode
    print(f"Using episode_id: {episode_id}")

    from services.episode_service import EpisodeService

    svc = EpisodeService()
    try:
        result = await svc.process_episode(str(episode_id))
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1

    if not result:
        print("❌ No result returned")
        return 1

    clips = result.get("clips") or []
    print(f"Clips generated: {len(clips)}")
    if not clips:
        return 0

    seed = sum(1 for c in clips if c.get("seed_idx") is not None)
    with_punct_end = sum(
        1 for c in clips if (c.get("text", " ").strip().endswith(tuple(".!?")))
    )
    top = clips[0]
    start = float(top.get("start", 0.0))
    end = float(top.get("end", start))
    title = top.get("title") or top.get("display_title") or "No title"
    payoff_type = (top.get("features") or {}).get("payoff_type", "n/a")

    print(f"   Seed clips: {seed}")
    print(f"   End with punctuation: {with_punct_end}/{len(clips)}")
    print(f"   Top clip: {title}")
    print(f"     {start:.2f}s-{end:.2f}s  ({max(0.0, end - start):.1f}s)")
    print(f"     SeedIdx={top.get('seed_idx')}  PayoffType={payoff_type}")
    
    # Debug: Show actual score values
    print(f"   Score debug:")
    print(f"     virality: {top.get('virality', 'missing')}")
    print(f"     final_score: {top.get('final_score', 'missing')}")
    print(f"     display_score: {top.get('display_score', 'missing')}")
    print(f"     hook_score: {top.get('hook_score', 'missing')}")
    print(f"     features: {top.get('features', {})}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


