# backend/services/trending_provider.py
import os, json, time, math, re, logging, hashlib
from typing import List, Dict, Optional
from datetime import datetime, timezone

from config.settings import (
    TRENDING_TERMS_FILE, TREND_GLOBAL_WEIGHT, TREND_CATEGORY_WEIGHT,
    TREND_HALF_LIFE_DAYS, TREND_AB_PCT, TREND_CONTROL_SCALE
)

logger = logging.getLogger(__name__)

_CACHE = {"path": None, "mtime": 0.0, "doc": None}
WORD = re.compile(r"\b[#$]?[A-Za-z][A-Za-z0-9\-]{1,30}\b", re.I)

CATEGORY_KEYWORDS = {
    "tech": {"ai","openai","agents","apple","google","microsoft","android","iphone","llm","chip","gpu"},
    "business": {"inflation","earnings","revenue","market","fed","rate","economy","jobs","startup","valuation"},
    "crypto": {"bitcoin","btc","ethereum","eth","solana","sol","binance","defi","etf","halving","onchain"},
    "sports": {"nfl","nba","soccer","mlb","ufc","olympics","tennis","golf","formula","f1"},
    "entertainment": {"movie","tv","series","celebrity","trailer","box","album"},
    "health": {"fitness","nutrition","mental","wellness","sleep","diet","workout","meditation"},
    "politics": {"election","policy","congress","president","campaign","parliament","senate","house"},
    "general": set()
}

def _load_doc():
    path = TRENDING_TERMS_FILE
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _CACHE["path"] == path and _CACHE["mtime"] == mtime:
        return _CACHE["doc"]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            doc = json.load(f)
        _CACHE.update({"path": path, "mtime": mtime, "doc": doc})
        return doc
    except Exception as e:
        logger.warning("trend.load_failed: %s", e)
        return None

def _half_life_weight(days: float, half_life: float) -> float:
    lam = math.log(2.0) / max(1e-6, half_life)
    return math.exp(-lam * max(0.0, days))

def _now_utc():
    return datetime.now(timezone.utc)

def _normalize(text: str) -> List[str]:
    text = (text or "").lower()
    return WORD.findall(text)

def detect_categories(text: str, max_k: int = 2) -> List[str]:
    toks = set(_normalize(text))
    scores = []
    for cat, keys in CATEGORY_KEYWORDS.items():
        if not keys:
            continue
        overlap = sum(1 for k in keys if k in toks)
        if overlap > 0:
            scores.append((overlap, cat))
    scores.sort(reverse=True)
    cats = [c for _, c in scores[:max_k]]
    return cats or ["general"]

def _score_terms(text_tokens: List[str], items: List[Dict], half_life_days: float) -> float:
    if not items:
        return 0.0
    now = _now_utc()
    toks = set(text_tokens)
    score = 0.0
    joined = " ".join(text_tokens)
    for it in items:
        term = (it.get("term") or "").lower()
        if not term:
            continue
        aliases = [a.lower() for a in (it.get("aliases") or [])]
        hit = (term in toks) or any(a in toks for a in aliases)
        if not hit and " " in term and term in joined:
            hit = True
        if hit:
            base = float(it.get("weight", 0.0))
            ds = it.get("decay_start")
            try:
                ds_dt = datetime.fromisoformat(ds.replace("Z","+00:00")) if ds else now
            except Exception:
                ds_dt = now
            days = (now - ds_dt).total_seconds() / 86400.0
            w = base * _half_life_weight(days, half_life_days)
            score += w
    return min(1.0, score)

def collect_hashtags(seg_text: str, episode_meta: Optional[Dict]) -> List[str]:
    """Extract hashtags from segment text and episode metadata"""
    import re
    
    # Text hashtags like #Productivity
    text_tags = [f"#{m.lower()}" for m in re.findall(r"#([A-Za-z0-9_]{2,40})", seg_text or "")]
    
    # Episode metadata hashtags
    meta_tags = []
    if episode_meta:
        for k in ("hashtags", "tags", "social_hashtags", "topics"):
            v = episode_meta.get(k)
            if isinstance(v, (list, tuple)):
                meta_tags.extend([f"#{str(x).lower().lstrip('#')}" for x in v])
    
    # Dedupe and limit to top 8
    tags = list(dict.fromkeys(text_tags + meta_tags))[:8]
    return tags

def _ab_scale(key: Optional[str]) -> float:
    """
    Continuous AB scaling:
    - Treatment bucket: 1.0 (full weight)
    - Control bucket: TREND_CONTROL_SCALE (reduced but non-zero)
    """
    if TREND_AB_PCT >= 100:
        return 1.0
    if not key:
        return 0.0
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(h[:6], 16) % 100
    return 1.0 if bucket < TREND_AB_PCT else TREND_CONTROL_SCALE

def trend_match_score(text: str, hashtags: Optional[List[str]] = None,
                      categories: Optional[List[str]] = None,
                      ab_key: Optional[str] = None) -> float:
    scale = _ab_scale(ab_key)
    if scale <= 0.0:
        return 0.0
    doc = _load_doc()
    if not doc:
        return 0.0

    # Check for stale file
    STALE_DAYS = 3
    try:
        age_days = (time.time() - os.path.getmtime(TRENDING_TERMS_FILE))/86400.0
        if age_days > STALE_DAYS:
            logger.warning("trend.file_stale", extra={"age_days": round(age_days,2)})
    except OSError:
        logger.warning("trend.file_missing", extra={"path": TRENDING_TERMS_FILE})

    half_life_days = float(doc.get("half_life_days", TREND_HALF_LIFE_DAYS))
    tokens = _normalize(" ".join(filter(None, [text, " ".join(hashtags or [])])))

    global_items = doc.get("global", [])
    by_cat = doc.get("by_category", {})

    cats = categories or detect_categories(text, max_k=2)

    s_global = _score_terms(tokens, global_items, half_life_days)
    s_cats = 0.0
    for c in cats:
        s_cats = max(s_cats, _score_terms(tokens, by_cat.get(c, []), half_life_days))

    combined = TREND_GLOBAL_WEIGHT * s_global + TREND_CATEGORY_WEIGHT * s_cats
    return float(min(1.0, combined * scale))