"""
Word structure normalization utilities
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def normalize_words(words: List[Dict]) -> List[Dict]:
    """Normalize word structure to consistent schema"""
    if not words:
        return []
    
    out = []
    for w in words or []:
        if not isinstance(w, dict):
            continue
            
        word = w.get("word") or w.get("text") or w.get("token") or w.get("w")
        start = w.get("start") or w.get("ts") or w.get("begin") or w.get("s")
        end = w.get("end") or w.get("te") or w.get("until") or w.get("e")
        
        if word and start is not None and end is not None:
            out.append({
                "word": str(word), 
                "start": float(start), 
                "end": float(end)
            })
    
    return out
