diff --git a/backend/tests/test_rank_candidates.py b/backend/tests/test_rank_candidates.py
index 2fa75916901647f6932b4d51e2f5c8cb37c7de40..5eece2e8d87bfdb69c3acf5e821923e1eb6f9ee5 100644
--- a/backend/tests/test_rank_candidates.py
+++ b/backend/tests/test_rank_candidates.py
@@ -1,111 +1,124 @@
 """
 Ranking behavior tests (light integration).
 We monkeypatch clip_engine.compute_features so we don't need real audio.
 """
 
 import types
 import sys
 import os
 
 # Add backend to path for imports
-sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
+sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
 
 # Provide a minimal numpy stub if numpy isn't available
 if "numpy" not in sys.modules:
     sys.modules["numpy"] = types.SimpleNamespace()
 
 # Provide a minimal pydantic stub if pydantic isn't available
 if "pydantic" not in sys.modules:
     class _BaseModel:
         pass
 
     def _Field(default=None, **kwargs):
         return default
 
     sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_BaseModel, Field=_Field)
 
 # Stub out heavy secret_sauce module if missing dependencies
 if "services.secret_sauce" not in sys.modules:
     fake_ss = types.SimpleNamespace(
         compute_features_v4=lambda *a, **k: {},
         score_segment_v4=lambda *a, **k: {"final_score": 0.0, "winning_path": "", "path_scores": {}, "synergy_multiplier": 1.0, "bonuses_applied": 0.0, "bonus_reasons": []},
         explain_segment_v4=lambda *a, **k: "",
         viral_potential_v4=lambda *a, **k: 0.0,
         get_clip_weights=lambda: {},
     )
     sys.modules["services.secret_sauce"] = fake_ss
+if "services.secret_sauce_pkg" not in sys.modules:
+    sys.modules["services.secret_sauce_pkg"] = types.SimpleNamespace(
+        compute_features_v4=lambda *a, **k: {},
+        score_segment_v4=lambda *a, **k: {"final_score": 0.0, "winning_path": "", "path_scores": {}, "synergy_multiplier": 1.0, "bonuses_applied": 0.0, "bonus_reasons": []},
+        explain_segment_v4=lambda *a, **k: "",
+        viral_potential_v4=lambda *a, **k: 0.0,
+        get_clip_weights=lambda: {},
+        _heuristic_title=lambda *a, **k: "",
+        _grade_breakdown=lambda *a, **k: {},
+        resolve_platform=lambda x: x,
+        detect_podcast_genre=lambda x: "general",
+    )
 
 # Stub progress writer to avoid FastAPI dependency
 if "services.progress_writer" not in sys.modules:
     sys.modules["services.progress_writer"] = types.SimpleNamespace(write_progress=lambda *a, **k: None)
 
 from services.clip_score import ClipScoreService
+from services.prerank import pre_rank_candidates
+import services.secret_sauce_pkg as sspkg
 
 
 def test_rank_prefers_hooky_segment(monkeypatch):
     # Two fake segments: one 'hooky', one bland
     segments = [
         {"start": 10.0, "end": 28.0, "text": "Most people get this wrong. Here's why..."},
         {"start": 40.0, "end": 58.0, "text": "Today we talk about several things."},
     ]
 
     def fake_compute_features_v4(segment, audio_file):
         t = segment["text"].lower()
         return {
             "hook_score": 0.9 if "wrong" in t or "here's why" in t else 0.1,
             "prosody_score": 0.5,
             "emotion_score": 0.3,
             "question_score": 0.3 if "?" in t else 0.0,
             "payoff_score": 0.7 if "why" in t else 0.2,
             "info_density": 0.8,
             "loopability": 0.6,
         }
 
     # Create service instance and patch the secret_sauce import
     service = ClipScoreService(None)
-    monkeypatch.setattr("services.secret_sauce.compute_features_v4", fake_compute_features_v4)
+    monkeypatch.setattr(sspkg, "compute_features_v4", fake_compute_features_v4)
 
     ranked = service.rank_candidates(segments, audio_file="dummy", top_k=2)
     assert ranked[0]["text"].startswith("Most people get this wrong"), "Hooky segment should rank first"
     assert ranked[0]["score"] >= ranked[1]["score"]
 
 
 def test_rank_considers_payoff(monkeypatch):
     segments = [
         {"start": 0.0, "end": 18.0, "text": "What if you could 3x your results? The key is focus."},
         {"start": 20.0, "end": 38.0, "text": "What if you could 3x your results? Anyway, let's move on."},
     ]
 
     def fake_compute_features_v4(segment, audio_file):
         t = segment["text"].lower()
         has_payoff = "the key is" in t or "here's why" in t or "because" in t
         return {
             "hook_score": 0.7,  # both have a hook
             "prosody_score": 0.4,
             "emotion_score": 0.2,
             "question_score": 0.7,  # both ask a question
             "payoff_score": 0.9 if has_payoff else 0.1,
             "info_density": 0.8,
             "loopability": 0.6,
         }
 
     service = ClipScoreService(None)
-    monkeypatch.setattr("services.secret_sauce.compute_features_v4", fake_compute_features_v4)
+    monkeypatch.setattr(sspkg, "compute_features_v4", fake_compute_features_v4)
 
     ranked = service.rank_candidates(segments, audio_file="dummy", top_k=2)
     assert "the key is" in ranked[0]["text"].lower(), "Segment with payoff should rank higher"
 
 
 def test_pre_rank_filters_ads():
     segments = [
         {"start": 0.0, "end": 10.0, "text": "This episode is sponsored by Acme Co. Use code POD10."},
         {"start": 10.0, "end": 20.0, "text": "Here's a fascinating insight into tech."},
     ]
 
-    service = ClipScoreService(None)
-    filtered = service.pre_rank_candidates(segments, episode_id="ep1")
+    filtered = pre_rank_candidates(segments, episode_id="ep1")
 
     # Ad segment should be flagged and removed
     assert segments[0]["is_advertisement"] is True
     assert len(filtered) == 1
     assert filtered[0]["text"].startswith("Here")
