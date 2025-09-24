"""
Scoring module for viral detection system.
Contains scoring functions, weights, and viral potential calculations.
"""

from typing import Dict, List, Tuple, Any
import logging
import os
from config_loader import get_config

logger = logging.getLogger(__name__)

# Scoring version and feature flags
SCORING_VERSION = "v4.8-unified-2025-09"
USE_PL_V2 = True     # read platform_length_score_v2 if present
USE_Q_LIST = True    # include q_list_score with a small weight

# --- New: env-configurable trend boost (privacy-safe, from features["trend_match_score"]) ---
ENABLE_TREND_BOOST = os.getenv("ENABLE_TREND_BOOST", "1") == "1"
try:
    TREND_BOOST_WEIGHT = float(os.getenv("TREND_BOOST_WEIGHT", "0.05"))  # recommended 0.03–0.10
except Exception:
    TREND_BOOST_WEIGHT = 0.05

# Module-level flag to log normalization only once
_NORMALIZE_LOGGED_ONCE = False


def get_clip_weights():
    """Get normalized clip weights (sum to 1.0)"""
    global _NORMALIZE_LOGGED_ONCE
    weights = dict(get_config()["weights"])
    ws = sum(weights.values()) or 1.0
    if abs(ws - 1.0) > 1e-6:
        weights = {k: v / ws for k, v in weights.items()}
        if not _NORMALIZE_LOGGED_ONCE:
            logger.debug("Weights normalized from %.2f to 1.00", ws)
            _NORMALIZE_LOGGED_ONCE = True
    return weights


def score_segment_v4(features: Dict, apply_penalties: bool = True, genre: str = 'general', platform: str = None) -> Dict:
    """V4 Multi-path scoring system with genre awareness and platform multipliers.
       Prefers arousal_score_v2 if present; includes optional trend boost."""
    from config.settings import ENABLE_TREND_BOOST, TREND_BOOST_WEIGHT

    f = features

    # --- prefer v2 arousal if present ---
    AROUS = float(f.get("arousal_score_v2", f.get("arousal_score", 0.0)))
    f.setdefault("_arousal_src", "v2" if "arousal_score_v2" in f else "v1")

    # Get genre-specific scoring paths
    if genre != 'general':
        from .genres import GenreAwareScorer
        genre_scorer = GenreAwareScorer()
        genre_profile = genre_scorer.genres.get(genre, genre_scorer.genres['general'])
        path_scores = genre_profile.get_scoring_paths(features)
    else:
        # Platform length v2 + q_list_score integration
        pl = (
            f.get("platform_length_score_v2", f.get("platform_len_match", 0.0))
            if USE_PL_V2 else
            f.get("platform_len_match", 0.0)
        )
        ql = (f.get("q_list_score", 0.0) if USE_Q_LIST else 0.0)

    # Apply ASR quality-based feature weight adjustments
    from config.settings import ENABLE_ASR_QUALITY_WEIGHTING
    
    if ENABLE_ASR_QUALITY_WEIGHTING:
        # Get ASR quality metrics from episode metadata
        aq = f.get("asr_quality") or {}
        punct_ok = aq.get("punct_ok", 1.0)
        low_asr = punct_ok < 0.20
        
        if low_asr:
            # De-emphasize text-fragile features, lean more on arousal_v2
            if "hook_score" in f:
                f["hook_score"] *= 0.85
            if "question_score" in f:
                f["question_score"] *= 0.85
            if "payoff_score" in f:
                f["payoff_score"] *= 0.90
            if "info_density" in f:
                f["info_density"] *= 0.95
            
            # Blend arousal scores when ASR quality is low
            if "arousal_score_v2" in f:
                f["arousal_score"] = 0.4 * f.get("arousal_score", 0.0) + 0.6 * f.get("arousal_score_v2", 0.0)
            
            logger.debug("ASR quality weighting applied: punct_ok=%.2f, low_asr=%s", punct_ok, low_asr)

    # ---- use AROUS (v2 if present) in all paths ----
    path_a = (0.42 * f.get("hook_score", 0.0) + 0.15 * AROUS +
              0.10 * f.get("payoff_score", 0.0) + 0.12 * f.get("info_density", 0.0) +
              0.18 * pl + 0.03 * f.get("loopability", 0.0) +
              0.02 * f.get("insight_score", 0.0) + 0.02 * ql)

    path_b = (0.30 * f.get("payoff_score", 0.0) + 0.25 * f.get("info_density", 0.0) +
              0.15 * f.get("emotion_score", 0.0) + 0.10 * f.get("hook_score", 0.0) +
              0.10 * pl + 0.05 * AROUS +
              0.05 * f.get("insight_score", 0.0))

    path_c = (0.30 * AROUS + 0.20 * f.get("emotion_score", 0.0) +
              0.20 * f.get("hook_score", 0.0) + 0.10 * f.get("loopability", 0.0) +
              0.10 * pl + 0.05 * f.get("question_score", 0.0) +
              0.05 * f.get("insight_score", 0.0))

    path_d = (0.25 * f.get("question_score", 0.0) + 0.25 * f.get("info_density", 0.0) +
              0.20 * f.get("payoff_score", 0.0) + 0.20 * f.get("hook_score", 0.0) +
              0.10 * pl)

    path_scores = {"hook": path_a, "payoff": path_b, "energy": path_c, "structured": path_d}

    # Find winning path
    winning_path = max(path_scores.items(), key=lambda x: x[1])[0]
    base_score = path_scores[winning_path]

    # Apply synergy (additive, bounded) – use AROUS here too
    synergy_boost = _synergy_v4(
        f.get("hook_score", 0.0),
        AROUS,
        f.get("payoff_score", 0.0)
    )

    # Apply bonuses
    bonuses_applied = 0.0
    bonus_reasons: List[str] = []

    # Insight bonus
    insight_score = f.get("insight_score", 0.0)
    if insight_score >= 0.7:
        insight_bonus = 0.15
        bonuses_applied += insight_bonus
        bonus_reasons.append(f"insight_high_{insight_bonus:.2f}")

    # Question/List bonus
    question_score = f.get("question_score", 0.0)
    if question_score >= 0.6:
        ql_bonus = 0.10
        bonuses_applied += ql_bonus
        bonus_reasons.append(f"question_list_{ql_bonus:.2f}")

    # Platform length match bonus
    platform_len = f.get("platform_len_match", 0.0)
    if platform_len >= 0.8:
        platform_bonus = 0.05
        bonuses_applied += platform_bonus
        bonus_reasons.append(f"platform_length_{platform_bonus:.2f}")

    # Optional: trend boost (privacy-safe; small, capped)
    trend = float(f.get("trend_match_score", 0.0) or 0.0)
    if trend > 0.0 and 'ENABLE_TREND_BOOST' in globals():
        # If imported settings exist, use them
        try:
            if ENABLE_TREND_BOOST:
                boost = min(0.10, float(TREND_BOOST_WEIGHT) * trend)  # cap at +0.10 absolute
                if boost > 0:
                    bonuses_applied += boost
                    bonus_reasons.append(f"trend_boost_{boost:.2f}")
        except Exception:
            pass

    # Apply ad penalty if needed
    ad_penalty = f.get("_ad_penalty", 0.0)
    if apply_penalties and ad_penalty > 0:
        base_score -= ad_penalty

    # CAP: prevent double-counting of high-energy chatter
    # Allow up to +25% uplift vs the strongest single component.
    hook = f.get("hook_score", 0.0)
    payoff = f.get("payoff_score", 0.0)
    triad = hook + AROUS + payoff + synergy_boost
    triad_cap = 1.25 * max(hook, AROUS, payoff)
    triad_adj = min(triad, triad_cap)
    
    # Calculate final score with capped triad
    final_score = base_score + (triad_adj - (hook + AROUS + payoff)) + bonuses_applied

    # Platform-fit boost for excellent length matches
    pl_v2 = f.get("platform_length_score_v2", f.get("platform_len_match", 0.0))
    if pl_v2 >= 0.90:
        platform_fit_boost = 0.02
        final_score += platform_fit_boost
        bonus_reasons.append(f"platform_fit_boost_{platform_fit_boost:.2f}")

    final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
    viral_score_100 = int(final_score * 100)

    # legacy/telemetry fields
    f["_synergy_boost"] = synergy_boost
    f["_synergy_multiplier"] = 1.0 + synergy_boost

    return {
        "final_score": final_score,
        "viral_score_100": viral_score_100,
        "winning_path": winning_path,
        "path_scores": path_scores,
        "synergy_multiplier": 1.0 + synergy_boost,
        "bonuses_applied": bonuses_applied,
        "bonus_reasons": bonus_reasons,
    }


def explain_segment_v4(features: Dict, genre: str = 'general', scoring_result: Dict = None) -> Dict:
    """Explain V4 scoring results"""
    if scoring_result is None:
        scoring_result = score_segment_v4(features, genre=genre)
    f = features

    strengths = []
    improvements = []

    hook_score = f.get("hook_score", 0.0)
    if hook_score >= 0.8:
        strengths.append("**Killer Hook**: Opens with attention-grabbing content")
    elif hook_score < 0.4:
        improvements.append("**Weak Hook**: Needs compelling opening")

    viral_score = scoring_result["viral_score_100"]
    if viral_score >= 70:
        overall = "**High Viral Potential** - Strong fundamentals"
    elif viral_score >= 50:
        overall = "**Good Potential** - Solid foundation"
    else:
        overall = "**Needs Work** - Multiple issues to address"

    return {
        "overall_assessment": overall,
        "viral_score": viral_score,
        "winning_strategy": scoring_result["winning_path"],
        "strengths": strengths,
        "improvements": improvements
    }


def viral_potential_v4(features: dict, length_s: float, platform: str = "general", genre: str = 'general') -> dict:
    """Calculate viral potential using V4 scoring"""
    scoring_result = score_segment_v4(features, genre=genre, platform=platform)
    viral_score = scoring_result["viral_score_100"]

    # Platform recommendations based on score and features
    platforms = []
    if viral_score >= 70:
        platforms = ["TikTok", "Instagram Reels", "YouTube Shorts"]
    elif viral_score >= 50:
        platforms = ["TikTok", "Instagram Reels"]
    else:
        platforms = ["TikTok"]  # Default fallback

    return {
        "viral_score": viral_score,
        "platforms": platforms,
        "confidence": "High" if viral_score >= 70 else "Medium" if viral_score >= 50 else "Low"
    }


def score_segment(features: Dict, weights: Dict = None, version: str = "v4", genre: str = 'general') -> float:
    """Legacy scoring function for backward compatibility"""
    if version == "v4":
        result = score_segment_v4(features, weights, genre=genre)
        return result["final_score"]

    if weights is None:
        weights = get_clip_weights()

    # New: prefer v2 arousal in legacy path as well
    arousal_legacy = float(features.get("arousal_score_v2", features.get("arousal_score", 0.0)))

    return float(
        weights["hook"] * features.get("hook_score", 0.0) +
        weights["arousal"] * arousal_legacy +
        weights["emotion"] * features.get("emotion_score", 0.0) +
        weights["q_or_list"] * features.get("question_score", 0.0) +
        weights["payoff"] * features.get("payoff_score", 0.0) +
        weights["info"] * features.get("info_density", 0.0) +
        weights["loop"] * features.get("loopability", 0.0)
    )


def viral_potential(features: dict, length_s: float, platform: str = "general", version: str = "v4", genre: str = 'general') -> dict:
    """Legacy viral potential function for backward compatibility"""
    if version == "v4":
        return viral_potential_v4(features, length_s, platform, genre)

    # New: include v2 arousal when present
    f = {
        "hook_score": float(features.get("hook_score", 0.0)),
        "arousal_score": float(features.get("arousal_score_v2", features.get("arousal_score", 0.0))),
        "emotion_score": float(features.get("emotion_score", 0.0)),
        "question_score": float(features.get("question_score", 0.0)),
        "payoff_score": float(features.get("payoff_score", 0.0)),
        "info_density": float(features.get("info_density", 0.0)),
        "loopability": float(features.get("loopability", 0.0)),
    }

    base = (
        0.37 * f["hook_score"] + 0.17 * f["emotion_score"] + 0.16 * f["arousal_score"] +
        0.14 * f["payoff_score"] + 0.08 * f["loopability"] + 0.05 * f["info_density"] +
        0.03 * f["question_score"]
    )

    viral_0_100 = int(max(0.0, min(1.0, base)) * 100)
    platforms = ["TikTok"] if f["hook_score"] >= 0.6 else []

    return {"viral_score": viral_0_100, "platforms": platforms}


def explain_segment(features: Dict, weights: Dict = None, version: str = "v4", genre: str = 'general') -> Dict:
    """Legacy explain function for backward compatibility"""
    if version == "v4":
        return explain_segment_v4(features, weights, genre=genre)

    if weights is None:
        weights = get_clip_weights()

    # New: include v2 arousal when present
    arousal_legacy = float(features.get("arousal_score_v2", features.get("arousal_score", 0.0)))

    contributions = {
        "Hook": weights["hook"] * features.get("hook_score", 0.0),
        "Arousal": weights["arousal"] * arousal_legacy,
        "Emotion": weights["emotion"] * features.get("emotion_score", 0.0),
        "Q/List": weights["q_or_list"] * features.get("question_score", 0.0),
        "Payoff": weights["payoff"] * features.get("payoff_score", 0.0),
        "Info": weights["info"] * features.get("info_density", 0.0),
        "Loop": weights["loop"] * features.get("loopability", 0.0)
    }

    reasons = []
    if features.get("hook_score", 0.0) >= 0.8:
        reasons.append("Starts with a strong hook that will grab attention")
    else:
        reasons.append("Weak opening - viewers might scroll past this")

    return {
        "contributions": contributions,
        "reasons": reasons[:6]
    }


def viral_potential_from_segment(segment: Dict, audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """Calculate viral potential from a segment"""
    from .features import compute_features_v4
    features = compute_features_v4(segment, audio_file, genre=genre, platform=platform)
    length_s = float(segment["end"] - segment["start"])
    return viral_potential_v4(features, length_s, platform, genre)


def explain_segment_from_segment(segment: Dict, audio_file: str, genre: str = 'general', platform: str = 'tiktok') -> Dict:
    """Explain scoring from a segment"""
    from .features import compute_features_v4
    features = compute_features_v4(segment, audio_file, genre=genre, platform=platform)
    return explain_segment_v4(features, genre=genre)


def _synergy_v4(hook: float, arousal: float, payoff: float) -> float:
    """
    Additive synergy boost (0.00..0.12).
    High triad → +0.12, moderate → +0.06, else +0.00.
    """
    if hook >= 0.60 and arousal >= 0.50 and payoff >= 0.40:
        return 0.12
    if hook >= 0.40 and (arousal >= 0.40 or payoff >= 0.30):
        return 0.06
    return 0.0
