"""
Platform recommendation service for post-selection platform optimization.
Takes clip characteristics and recommends optimal platforms with fit scores.
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlatformRecommendation:
    platform: str
    fit_score: float
    reason: str

class PlatformRecommender:
    """Recommends platforms based on clip duration and content characteristics."""
    
    def __init__(self):
        # Duration-based platform mapping
        self.duration_mapping = {
            (0, 11): ["shorts", "tiktok", "reels"],
            (12, 22): ["shorts", "tiktok", "reels"],
            (23, 30): ["shorts", "tiktok", "reels"],
            (31, 45): ["reels", "tiktok", "shorts"],  # shorts with retention risk
            (46, 60): ["reels", "shorts", "tiktok"],  # 60s cap for shorts
            (61, 90): ["reels", "tiktok"]  # shorts excluded (over 60s)
        }
        
        # Content-aware nudges
        self.content_weights = {
            "shorts": {
                "high_info": 0.05,  # Educational content does well
                "advice": 0.05,
                "stats": 0.05
            },
            "tiktok": {
                "humor": 0.05,  # Humor/skits do well
                "shock": 0.05,
                "skits": 0.05
            },
            "reels": {
                "narrative": 0.05,  # Narrative content
                "heartfelt": 0.05,
                "aesthetic": 0.05
            }
        }
    
    def recommend_platforms(self, 
                          duration_s: float, 
                          content_signals: Dict[str, float] = None,
                          archetype: str = None) -> List[PlatformRecommendation]:
        """
        Recommend platforms based on duration and content characteristics.
        
        Args:
            duration_s: Clip duration in seconds
            content_signals: Dict of content characteristics (info_density, humor, etc.)
            archetype: Content archetype (confession, hot_take, etc.)
        
        Returns:
            List of PlatformRecommendation objects sorted by fit_score
        """
        content_signals = content_signals or {}
        archetype = archetype or ""
        
        # Get base platforms from duration
        base_platforms = self._get_duration_platforms(duration_s)
        
        # Calculate fit scores
        recommendations = []
        for platform in base_platforms:
            fit_score = self._calculate_fit_score(
                platform, duration_s, content_signals, archetype
            )
            reason = self._generate_reason(platform, duration_s, content_signals, archetype)
            
            recommendations.append(PlatformRecommendation(
                platform=platform,
                fit_score=fit_score,
                reason=reason
            ))
        
        # Sort by fit score (highest first)
        recommendations.sort(key=lambda x: x.fit_score, reverse=True)
        
        return recommendations
    
    def _get_duration_platforms(self, duration_s: float) -> List[str]:
        """Get base platforms based on duration."""
        for (min_dur, max_dur), platforms in self.duration_mapping.items():
            if min_dur <= duration_s <= max_dur:
                return platforms
        
        # Fallback for extreme durations
        if duration_s < 0:
            return ["shorts", "tiktok", "reels"]
        else:  # > 90s
            return ["reels", "tiktok"]
    
    def _calculate_fit_score(self, 
                           platform: str, 
                           duration_s: float, 
                           content_signals: Dict[str, float],
                           archetype: str) -> float:
        """Calculate platform fit score (0.0 to 1.0)."""
        base_score = 0.8  # Base fit for duration-appropriate platforms
        
        # Apply content-aware nudges
        if platform in self.content_weights:
            for signal, weight in self.content_weights[platform].items():
                if signal in content_signals:
                    base_score += weight * content_signals[signal]
        
        # Archetype-based adjustments
        archetype_boosts = {
            "confession": {"reels": 0.05, "tiktok": 0.03},
            "hot_take": {"tiktok": 0.05, "shorts": 0.03},
            "reveal": {"tiktok": 0.05, "reels": 0.03},
            "insider": {"shorts": 0.05, "reels": 0.03},
            "shock": {"tiktok": 0.05},
            "humor": {"tiktok": 0.05},
            "advice": {"shorts": 0.05, "reels": 0.03},
            "stats": {"shorts": 0.05, "reels": 0.03},
            "mystery": {"tiktok": 0.05, "reels": 0.03}
        }
        
        if archetype in archetype_boosts and platform in archetype_boosts[archetype]:
            base_score += archetype_boosts[archetype][platform]
        
        # Duration-specific adjustments
        if platform == "shorts" and duration_s > 45:
            base_score -= 0.1  # Retention risk for long shorts
        elif platform == "tiktok" and duration_s > 60:
            base_score -= 0.05  # TikTok prefers shorter content
        
        # Cap at 1.0
        return min(1.0, max(0.0, base_score))
    
    def _generate_reason(self, 
                        platform: str, 
                        duration_s: float, 
                        content_signals: Dict[str, float],
                        archetype: str) -> str:
        """Generate human-readable reason for platform recommendation."""
        reasons = []
        
        # Duration-based reason
        if 0 <= duration_s <= 11:
            reasons.append("universal short format")
        elif 12 <= duration_s <= 22:
            reasons.append("optimal short format")
        elif 23 <= duration_s <= 30:
            reasons.append("strong short format")
        elif 31 <= duration_s <= 45:
            reasons.append("medium format with retention considerations")
        elif 46 <= duration_s <= 60:
            reasons.append("long format within platform limits")
        else:
            reasons.append("extended format")
        
        # Content-based reasons
        if platform == "shorts" and content_signals.get("info_density", 0) > 0.7:
            reasons.append("high info density suits educational platform")
        elif platform == "tiktok" and content_signals.get("humor", 0) > 0.6:
            reasons.append("humor content performs well")
        elif platform == "reels" and content_signals.get("narrative", 0) > 0.6:
            reasons.append("narrative content fits platform style")
        
        # Archetype-based reasons
        if archetype in ["confession", "reveal"] and platform == "reels":
            reasons.append("personal content suits platform")
        elif archetype in ["hot_take", "shock"] and platform == "tiktok":
            reasons.append("controversial content performs well")
        elif archetype in ["advice", "stats"] and platform == "shorts":
            reasons.append("educational content ideal for platform")
        
        return "; ".join(reasons) if reasons else "duration-appropriate platform"

# Global instance
platform_recommender = PlatformRecommender()

def recommend_platforms_for_clip(clip: Dict) -> List[Dict]:
    """
    Recommend platforms for a single clip.
    
    Args:
        clip: Clip dictionary with duration, content signals, etc.
    
    Returns:
        List of platform recommendation dictionaries
    """
    duration_s = clip.get("duration_s", clip.get("dur", 0.0))
    
    # Extract content signals
    content_signals = {
        "info_density": clip.get("info_density", 0.0),
        "humor": clip.get("humor_score", 0.0),
        "shock": clip.get("shock_score", 0.0),
        "narrative": clip.get("narrative_score", 0.0),
        "advice": clip.get("advice_score", 0.0),
        "stats": clip.get("stats_score", 0.0)
    }
    
    # Get archetype if available
    archetype = clip.get("archetype", "")
    
    # Get recommendations
    recommendations = platform_recommender.recommend_platforms(
        duration_s, content_signals, archetype
    )
    
    # Convert to dictionary format
    return [
        {
            "platform": rec.platform,
            "fit_score": round(rec.fit_score, 2),
            "reason": rec.reason
        }
        for rec in recommendations
    ]

def add_platform_recommendations_to_clips(clips: List[Dict]) -> List[Dict]:
    """
    Add platform recommendations to a list of clips.
    
    Args:
        clips: List of clip dictionaries
    
    Returns:
        List of clips with added platform_recommendations field
    """
    for clip in clips:
        clip["platform_recommendations"] = recommend_platforms_for_clip(clip)
    
    return clips
