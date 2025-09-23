# backend/jobs/trends_collector.py
import os, json, re, math, argparse
from datetime import datetime, timezone
from collections import Counter, defaultdict
from urllib.request import urlopen, Request

import os
from config.settings import (
    TREND_CATEGORIES, TREND_HALF_LIFE_DAYS
)

# Read file path dynamically to allow test overrides
def get_trending_terms_file():
    return os.getenv("TRENDING_TERMS_FILE", os.getenv("TRENDS_FILE", "backend/data/trending_terms.json"))

# Optional Google Trends
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except Exception:
    HAS_PYTRENDS = False

if not HAS_PYTRENDS:
    print("[trends_collector] pytrends not installed; skipping Google Trends (RSS only).")

STOP = set("""
a an the of to in on for and or if with from is are was were be being been this that
these those there here it its it's i you he she we they them our your will should
""".split())

TOKEN = re.compile(r"\$?[A-Za-z][A-Za-z0-9\-]{1,30}")

RSS_BY_CAT = {
    "general": [
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/all/.rss",
    ],
    "tech": [
        "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/technology/.rss",
        "https://www.reddit.com/r/MachineLearning/.rss",
    ],
    "business": [
        "https://news.google.com/rss/search?q=business&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/business/.rss",
        "https://www.reddit.com/r/investing/.rss",
    ],
    "crypto": [
        "https://news.google.com/rss/search?q=crypto%20OR%20bitcoin%20OR%20ethereum&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/CryptoCurrency/.rss",
    ],
    "sports": [
        "https://news.google.com/rss/search?q=sports&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/sports/.rss",
    ],
    "entertainment": [
        "https://news.google.com/rss/search?q=entertainment&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/entertainment/.rss",
    ],
    "health": [
        "https://news.google.com/rss/search?q=health&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/Health/.rss",
    ],
    "politics": [
        "https://news.google.com/rss/search?q=politics&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/politics/.rss",
    ],
}

def tokenize(text: str):
    text = (text or "").lower()
    for m in TOKEN.finditer(text):
        w = m.group(0)
        if w in STOP or len(w) < 3:
            continue
        yield w

def fetch_rss_titles(url: str, limit=80):
    import time
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urlopen(req, timeout=10).read().decode("utf-8", "ignore")
            titles = re.findall(r"<title>(.*?)</title>", data, flags=re.I | re.S)
            return [re.sub(r"<.*?>", "", t).strip() for t in titles[1:limit+1]]
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return []

def collect_rss_terms():
    cat_terms = defaultdict(Counter)
    errs = 0
    for cat in TREND_CATEGORIES:
        for u in RSS_BY_CAT.get(cat, []):
            try:
                for title in fetch_rss_titles(u):
                    for w in tokenize(title):
                        cat_terms[cat][w] += 1
            except Exception as e:
                errs += 1
                print(f"[trends_collector] fetch_error: {u} - {e}")
    return cat_terms, errs

def collect_pytrends():
    if not HAS_PYTRENDS:
        return {}
    py = TrendReq(hl="en-US", tz=360)
    seeds_by_cat = {
        "general": ["news", "trending"],
        "tech": ["ai", "apple", "openai", "agents", "semiconductor"],
        "business": ["inflation", "earnings", "economy"],
        "crypto": ["bitcoin", "ethereum", "solana", "fed rate"],
        "sports": ["nfl", "nba", "soccer"],
        "entertainment": ["movie", "tv", "celebrity"],
        "health": ["nutrition", "mental health", "fitness"],
        "politics": ["election", "policy", "congress"],
    }
    out = defaultdict(Counter)
    for cat, seeds in seeds_by_cat.items():
        for s in seeds:
            try:
                py.build_payload([s], geo="US")
                rq = py.related_queries()
                if s in rq and rq[s].get("rising") is not None:
                    for row in rq[s]["rising"].to_dict("records"):
                        q = (row.get("query") or "").lower().strip()
                        if q and q not in STOP:
                            out[cat][q] += 1 + int(row.get("value", 0) > 0)
            except Exception:
                pass
    return out

def to_items(counter, topk=150):
    items = []
    # Handle both Counter and dict
    if hasattr(counter, 'most_common'):
        items_data = counter.most_common(topk)
    else:
        # Convert dict to sorted list
        items_data = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topk]
    
    for term, cnt in items_data:
        w = min(1.0, 0.25 + 0.05 * cnt)  # bounded 0.25..1.0
        items.append({
            "term": term,
            "weight": round(w, 3),
            "aliases": [],
            "decay_start": None,
            "source": "collector"
        })
    return items

def main():
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rss, errs = collect_rss_terms()
    gtr = collect_pytrends()

    per_cat = {}
    for cat in TREND_CATEGORIES:
        bag = Counter()
        bag.update(rss.get(cat, Counter()))
        bag.update(gtr.get(cat, Counter()))
        per_cat[cat] = to_items(bag)

    global_bag = Counter()
    for c in per_cat.values():
        for it in c:
            global_bag[it["term"]] += 1
    global_items = to_items(global_bag)

    for it in global_items:
        it["decay_start"] = now
    for cat in per_cat:
        for it in per_cat[cat]:
            it["decay_start"] = now

    doc = {
        "updated_at": now,
        "half_life_days": TREND_HALF_LIFE_DAYS,
        "global": global_items,
        "by_category": per_cat
    }

    trending_file = get_trending_terms_file()
    os.makedirs(os.path.dirname(trending_file), exist_ok=True)
    with open(trending_file, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"[trends_collector] wrote {trending_file} "
          f"(global={len(global_items)}, " +
          ", ".join(f"{k}:{len(v)}" for k,v in per_cat.items()) + f", errors={errs})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="Print resulting JSON to stdout")
    args = ap.parse_args()
    main()