"""
Semantic deduplication utilities with safe fallback.
"""

from __future__ import annotations
from typing import List, Tuple
import math
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMB = True
    _MODEL = None
except Exception:
    HAS_EMB = False
    _MODEL = None

def _embed(texts: List[str]):
    # Lazy init to avoid cold-start penalty
    global _MODEL
    if not HAS_EMB:
        return [[hash(t) % 997 / 997.0] for t in texts]
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SEMANTIC_DEDUPE: lazy-loaded sentence-transformers model")
    return _MODEL.encode(texts, normalize_embeddings=True)

def cosine(a, b):
    """Calculate cosine similarity between two vectors"""
    if len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def semantic_dedupe(clips: List[dict], sim_thresh: float = 0.92) -> List[dict]:
    """
    Remove semantically similar clips, keeping the highest-scored one.
    
    Args:
        clips: List of clip dictionaries
        sim_thresh: Similarity threshold (0.0 to 1.0)
        
    Returns:
        Deduplicated list of clips
    """
    if not clips:
        return clips
    
    if not HAS_EMB:
        logger.warning("SEMANTIC_DEDUPE: disabled_fallback - embeddings not available")
        return clips
    
    texts = [c.get("text","") for c in clips]
    vecs  = _embed(texts)
    kept: List[int] = []
    
    for i, vi in enumerate(vecs):
        dup = False
        for j in kept:
            if cosine(vi, vecs[j]) >= sim_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    
    result = [clips[i] for i in kept]
    logger.info(f"SEMANTIC_DEDUPE: kept={len(result)} dropped={len(clips) - len(result)}")
    return result
