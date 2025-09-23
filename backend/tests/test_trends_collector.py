import os, json, tempfile
from collections import Counter
import jobs.trends_collector as col

def test_to_items_bounds():
    c = Counter()
    for i in range(30):
        c[f"term{i}"] = i+1
    items = col.to_items(c, topk=10)
    assert 1 <= len(items) <= 10
    for it in items:
        assert 0.25 <= it["weight"] <= 1.0

def test_collect_and_write(monkeypatch):
    # make RSS deterministic
    def fake_titles(url):
        return [
            "Fed rate cut expected after FOMC meeting",
            "Solana hits new highs as crypto rallies",
            "OpenAI showcases new AI agents at developer day"
        ]
    monkeypatch.setattr(col, "fetch_rss_titles", fake_titles)
    monkeypatch.setattr(col, "collect_pytrends", lambda: {})  # skip pytrends

    tmp = tempfile.TemporaryDirectory()
    out = os.path.join(tmp.name, "trends.json")
    monkeypatch.setenv("TRENDING_TERMS_FILE", out)
    # Also set the old env var to ensure it uses our path
    monkeypatch.setenv("TRENDS_FILE", out)

    col.main()

    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert "global" in doc and "by_category" in doc
    assert len(doc["global"]) > 0
