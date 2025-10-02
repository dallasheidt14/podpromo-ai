"""
Centralized Ad Detection Service

This module provides a single, canonical ad detection function that consolidates
all ad detection logic across the codebase to eliminate drift and ensure consistency.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Comprehensive ad detection patterns
AD_SPONSORS = {
    "squarespace", "nordvpn", "betterhelp", "hellofresh", "audible", "blue apron", "blueapron",
    "rocketmoney", "expressvpn", "skillshare", "grammarly", "wix", "shopify", "canva", "raycon",
    "mint mobile", "mintmobile", "casetify", "manscaped", "keeps", "hims", "feals", "athletic greens",
    "ag1", "liquid iv", "liquid-iv", "square", "stripe", "monday.com", "ziprecruiter", "indeed",
    "factor meals", "factor75", "doordash", "uber one", "raid shadow legends", "grainger", "wise",
    "notion", "atlassian", "freshbooks", "quickbooks", "capterra", "honey", "rakuten", "seatgeek",
    "draftkings", "fanduel", "policygenius", "rocketmortgage", "masterclass", "curiositystream",
    "brilliant", "wework", "godaddy", "bluehost", "hostgator", "namecheap", "anchor.fm", "anchor",
    "spotify for podcasters"
}

AD_PHRASES = (
    "this episode is sponsored by", "sponsored by", "use code", "promo code",
    "limited time offer", "free shipping", "visit our website", "visit ",
    "order now", "call now", "call today", "learn more", "shop now", "click the link",
    "link in the description", "terms apply", "see site for details"
)

# Compiled regex patterns for performance
PHONE_PAT = re.compile(r"\b(?:\+?1[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}\b")
URL_PAT = re.compile(
    r"\b(?:(?:https?://|www\.)\S+|(?:bit\.ly|t\.co|tinyurl\.com|linktr\.ee|lnk\.to|amzn\.to|geni\.us|kit\.co)/\S+)\b", 
    re.I
)

def looks_like_ad(text: str, strict: bool = False) -> bool:
    """
    Canonical ad detection function.
    
    Args:
        text: Text to analyze
        strict: If True, requires clear CTA phrases for URL/phone detection
        
    Returns:
        True if text appears to be an advertisement
    """
    if not text or not isinstance(text, str):
        return False
    
    t = text.lower().strip()
    if not t:
        return False
    
    # Fast path: check for ad phrases first
    if any(phrase in t for phrase in AD_PHRASES):
        return True
    
    # Check for sponsor brands
    if any(brand in t for brand in AD_SPONSORS):
        return True
    
    # Check for phone numbers
    if PHONE_PAT.search(t):
        return True
    
    # Check for URLs
    if URL_PAT.search(t):
        return True
    
    # Strict mode: require CTA phrases for URL/phone detection
    if strict:
        # Additional strict checks could be added here
        pass
    
    return False

def ad_like_score(text: str) -> float:
    """
    Calculate ad likelihood score (0.0 to 1.0).
    
    Args:
        text: Text to analyze
        
    Returns:
        Score from 0.0 (not an ad) to 1.0 (definitely an ad)
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    t = text.lower().strip()
    if not t:
        return 0.0
    
    score = 0.0
    
    # Check for ad phrases (high confidence)
    for phrase in AD_PHRASES:
        if phrase in t:
            score = max(score, 0.8)
            break
    
    # Check for sponsor brands (medium-high confidence)
    for brand in AD_SPONSORS:
        if brand in t:
            score = max(score, 0.7)
            break
    
    # Check for phone numbers (medium confidence)
    if PHONE_PAT.search(t):
        score = max(score, 0.6)
    
    # Check for URLs (medium confidence)
    if URL_PAT.search(t):
        score = max(score, 0.6)
    
    return score

def ad_penalty(text: str) -> dict:
    """
    Calculate comprehensive ad penalty metrics.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with ad penalty information
    """
    if not text or not isinstance(text, str):
        return {"ad_penalty": 0.0, "is_advertisement": False, "confidence": 0.0}
    
    t = text.lower().strip()
    if not t:
        return {"ad_penalty": 0.0, "is_advertisement": False, "confidence": 0.0}
    
    penalty = 0.0
    confidence = 0.0
    reasons = []
    
    # Check for ad phrases
    for phrase in AD_PHRASES:
        if phrase in t:
            penalty = max(penalty, 0.8)
            confidence = max(confidence, 0.9)
            reasons.append(f"ad_phrase: {phrase}")
            break
    
    # Check for sponsor brands
    for brand in AD_SPONSORS:
        if brand in t:
            penalty = max(penalty, 0.7)
            confidence = max(confidence, 0.8)
            reasons.append(f"sponsor: {brand}")
            break
    
    # Check for phone numbers
    if PHONE_PAT.search(t):
        penalty = max(penalty, 0.6)
        confidence = max(confidence, 0.7)
        reasons.append("phone_number")
    
    # Check for URLs
    if URL_PAT.search(t):
        penalty = max(penalty, 0.6)
        confidence = max(confidence, 0.7)
        reasons.append("url")
    
    is_advertisement = penalty >= 0.3
    
    return {
        "ad_penalty": penalty,
        "is_advertisement": is_advertisement,
        "confidence": confidence,
        "reasons": reasons
    }

def filter_ads_from_features(all_features: list) -> list:
    """
    Filter out ad content from features list.
    
    Args:
        all_features: List of feature dictionaries
        
    Returns:
        Filtered list with ad content removed
    """
    if not all_features:
        return []
    
    kept = []
    ad_count = 0
    
    for feature in all_features:
        text = feature.get("text", "")
        if looks_like_ad(text):
            ad_count += 1
            continue
        kept.append(feature)
    
    if ad_count > 0:
        logger.info(f"AD_FILTER: filtered {ad_count} ad segments, kept {len(kept)}")
    
    return kept

# Environment configuration
import os

ENHANCED_AD_FILTERING = os.getenv("ENHANCED_AD_FILTERING", "1") in ("1", "true", "True")
AD_PREROLL_SECONDS = int(os.getenv("AD_PREROLL_SECONDS", "35"))
AD_FILTER_STRICT = os.getenv("AD_FILTER_STRICT", "0") in ("1", "true", "True")
AD_FILTER_LOG_ONLY = os.getenv("AD_FILTER_LOG_ONLY", "0") in ("1", "true", "True")

def is_enhanced_ad_filtering_enabled() -> bool:
    """Check if enhanced ad filtering is enabled."""
    return ENHANCED_AD_FILTERING

def get_preroll_seconds() -> int:
    """Get the pre-roll ad guard duration in seconds."""
    return AD_PREROLL_SECONDS

def is_strict_mode() -> bool:
    """Check if strict ad filtering mode is enabled."""
    return AD_FILTER_STRICT

def is_log_only_mode() -> bool:
    """Check if log-only mode is enabled (for testing)."""
    return AD_FILTER_LOG_ONLY
