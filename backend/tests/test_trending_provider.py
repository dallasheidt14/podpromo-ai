import json, os, tempfile, time
from services import trending_provider as tp

SAMPLE = {
  "updated_at": "2025-09-23T00:00:00Z",
  "half_life_days": 7,
  "global": [
    {"term":"fed rate cut","weight":0.9,"aliases":["fomc","interest rate cut"],"decay_start":"2025-09-23T00:00:00Z"},
    {"term":"ai agents","weight":0.8,"aliases":["autonomous agents"],"decay_start":"2025-09-23T00:00:00Z"}
  ],
  "by_category": {
    "crypto": [
      {"term":"solana","weight":0.9,"aliases":["$sol"],"decay_start":"2025-09-23T00:00:00Z"}
    ],
    "tech": [
      {"term":"openai","weight":0.8,"aliases":[],"decay_start":"2025-09-23T00:00:00Z"}
    ]
  }
}

def write_tmp(doc):
    tmpdir = tempfile.TemporaryDirectory()
    path = os.path.join(tmpdir.name, "trends.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return tmpdir, path

def test_detect_categories_basic():
    text = "OpenAI releases new agents for developers"
    cats = tp.detect_categories(text)
    assert "tech" in cats

def test_trend_match_global_and_category(monkeypatch):
    tmpdir, path = write_tmp(SAMPLE)
    monkeypatch.setenv("TRENDING_TERMS_FILE", path)
    # force reload cache
    tp._CACHE.update({"path": None, "mtime": 0.0, "doc": None})
    # ab 100% for test
    monkeypatch.setenv("TREND_AB_PCT", "100")

    text = "FOMC meets this week to discuss a fed rate cut; solana rallies."
    score = tp.trend_match_score(text, hashtags=["#crypto"], categories=["crypto"], ab_key="episode123")
    assert score > 0.0

def test_half_life_decay(monkeypatch):
    # Test the half-life weight function directly
    weight_recent = tp._half_life_weight(0.0, 7.0)  # 0 days old
    weight_old = tp._half_life_weight(1000.0, 7.0)   # ~3 years old
    
    # Recent should have higher weight than old
    assert weight_recent > weight_old
    assert weight_recent == 1.0  # No decay for recent
    assert weight_old < 0.1      # Significant decay for old

def test_missing_file_returns_zero(monkeypatch):
    monkeypatch.setenv("TRENDING_TERMS_FILE", "does/not/exist.json")
    tp._CACHE.update({"path": None, "mtime": 0.0, "doc": None})
    score = tp.trend_match_score("anything", ab_key="e1")
    assert score == 0.0