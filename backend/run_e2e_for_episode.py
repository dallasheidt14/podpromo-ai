import os
import sys
import asyncio
import logging
from typing import Tuple, List, Dict

# Ensure backend root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from services.clip_score import ClipScoreService
from services.episode_service import EpisodeService

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("e2e_runner")

async def run_one(episode_id: str, platform: str = "tiktok_reels") -> None:
    episode_service = EpisodeService()
    svc = ClipScoreService(episode_service)

    logger.info(f"E2E_START: episode_id={episode_id} platform={platform}")
    finals, meta = await svc.get_candidates(episode_id, platform=platform)

    shipped = len(finals)
    ids = [c.get("id") for c in finals]
    spans = [(c.get("start"), c.get("end")) for c in finals]
    ft_stats = [c.get("ft_status") for c in finals]

    logger.info(f"E2E_DONE: shipped={shipped} ids={ids[:5]} spans={spans[:3]} ft_statuses_sample={ft_stats[:5]}")

    print("\n=== E2E RESULT ===")
    print(f"episode_id: {episode_id}")
    print(f"platform:   {platform}")
    print(f"finals:     {shipped}")
    if finals:
        print("sample:")
        for c in finals[:3]:
            print(f" - [{c.get('start'):.2f},{c.get('end'):.2f}] {c.get('id')} ft={c.get('ft_status')} score={c.get('final_score', 0):.3f}")

if __name__ == "__main__":
    ep = sys.argv[1] if len(sys.argv) > 1 else ""
    plat = sys.argv[2] if len(sys.argv) > 2 else "tiktok_reels"
    if not ep:
        print("Usage: python run_e2e_for_episode.py <episode_id> [platform]")
        sys.exit(2)
    asyncio.run(run_one(ep, plat))
