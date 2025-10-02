"""
Semantic deduplication utilities with safe fallback.
"""

from __future__ import annotations
from typing import List, Tuple
import math
import logging

logger = logging.getLogger(__name__)

try:
    # If you already have an embedding service, import that instead
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    def _embed(texts: List[str]):
        return _MODEL.encode(texts, normalize_embeddings=True)
    HAS_EMB = True
    logger.info("SEMANTIC_DEDUPE: using sentence-transformers model")
except Exception as e:
    HAS_EMB = False
    logger.warning(f"SEMANTIC_DEDUPE: sentence-transformers not available ({e}), using fallback")
    def _embed(texts: List[str]):
        # Fallback: crude hashing vector to keep API stable (low quality)
        return [[hash(t) % 997 / 997.0] for t in texts]

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
