from typing import Dict, List

from config_loader import get_config
from services.secret_sauce_pkg import _grade_breakdown, _heuristic_title


def format_candidates(
    ranked_segments: List[Dict],
    final_genre: str,
    backend_platform: str,
    episode_id: str,
) -> List[Dict]:
    """Convert ranked segments into candidate dictionaries"""
    candidates: List[Dict] = []
    config = get_config()
    for i, seg in enumerate(ranked_segments):
        features = seg.get("features", {})
        enhanced_features = {**features, "final_score": seg.get("raw_score", 0.0)}
        title = _heuristic_title(seg["text"], enhanced_features, config, rank=i + 1)
        grades = _grade_breakdown(enhanced_features)
        candidate = {
            "id": f"clip_{episode_id}_{i}",
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "title": title,
            "features": features,
            "grades": grades,
            "score": seg.get("raw_score", 0.0),
            "raw_score": seg.get("raw_score", 0),
            "display_score": seg.get("display_score", 0),
            "clip_score_100": seg.get("clip_score_100", 0),
            "confidence": seg.get("confidence", "Low"),
            "confidence_color": seg.get("confidence_color", "gray"),
            "synergy_mult": seg.get("synergy_multiplier", 1.0),
            "winning_path": seg.get("winning_path", "unknown"),
            "path_scores": seg.get("path_scores", {}),
            "genre": final_genre,
            "platform": backend_platform,
            "moment_type": seg.get("type", "general"),
            "moment_confidence": seg.get("confidence", 0.5),
            "status": "completed",
            "hook_score": features.get("hook_score", 0.0),
            "arousal_score": features.get("arousal_score", 0.0),
            "emotion_score": features.get("emotion_score", 0.0),
            "payoff_score": features.get("payoff_score", 0.0),
            "question_score": features.get("question_score", 0.0),
            "info_density": features.get("info_density", 0.0),
            "loopability": features.get("loopability", 0.0),
            "platform_length_match": features.get("platform_len_match", features.get("platform_length_match", 0.0)),
            "Viral Potential": seg.get("display_score", 0),
            "Hook Power": features.get("hook_score", 0.0) * 100,
            "Energy Level": features.get("arousal_score", 0.0) * 100,
            "Payoff Strength": features.get("payoff_score", 0.0) * 100,
            "Emotion Impact": features.get("emotion_score", 0.0) * 100,
            "Question Engagement": features.get("question_score", 0.0) * 100,
            "Information Density": features.get("info_density", 0.0) * 100,
            "Loop Potential": features.get("loopability", 0.0) * 100,
            "Platform Match": features.get("platform_len_match", 0.0) * 100,
        }
        candidates.append(candidate)
    return candidates
