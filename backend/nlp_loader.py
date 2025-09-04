import os
import logging

log = logging.getLogger(__name__)

_SPACY_AVAILABLE = None
_NLP = None
_SPACY_ERR = None

def spacy_enabled_by_config(cfg) -> bool:
    """Check if spaCy should be enabled based on config and environment"""
    return bool(cfg.get("nlp", {}).get("spacy_enabled", True)) and os.getenv("SPACY_DISABLED", "0") != "1"

def get_spacy(model="en_core_web_sm"):
    """Lazy load spaCy with graceful fallback"""
    global _SPACY_AVAILABLE, _NLP, _SPACY_ERR
    
    if _SPACY_AVAILABLE is not None:
        return _NLP
    
    try:
        import spacy  # type: ignore
        # Keep it light - disable parser and tagger, keep NER
        _NLP = spacy.load(model, disable=["parser", "tagger"])
        _SPACY_AVAILABLE = True
        log.info("spaCy loaded successfully: %s", model)
    except Exception as e:
        _SPACY_ERR = str(e)
        _SPACY_AVAILABLE = False
        _NLP = None
        log.warning("spaCy unavailable: %s", _SPACY_ERR)
    
    return _NLP

def spacy_status():
    """Return spaCy availability status for health checks"""
    return {
        "available": bool(_SPACY_AVAILABLE), 
        "error": _SPACY_ERR,
        "model": "en_core_web_sm" if _SPACY_AVAILABLE else None
    }

def maybe_ner(text, cfg, max_chars=2000):
    """Extract named entities with graceful fallback"""
    if not spacy_enabled_by_config(cfg):
        return []
    
    nlp = get_spacy(cfg.get("nlp", {}).get("model", "en_core_web_sm"))
    if not nlp:
        return []  # graceful fallback
    
    try:
        # Cap text length to keep latency predictable
        text = text[:max_chars]
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        log.debug("NER extraction failed: %s", e)
        return []
