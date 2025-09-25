"""
Payoff Rescue System - Boosts clips with strong CTAs, punchlines, or insights.
Designed to be safe and non-leapfrogging to maintain ranking integrity.
"""

import logging

logger = logging.getLogger(__name__)

# Configuration constants
PAYOFF_TYPES = {"cta", "punchline", "insight"}  # tune as you like
PAYOFF_CONF_MIN = 0.60   # require decent confidence
INSIGHT_MIN      = 0.20  # allow rescue if insight is present even if payoff label is missing
BUMP_LOW         = 0.03  # +3%
BUMP_HIGH        = 0.05  # +5%
TOP_MARGIN       = 0.02  # cannot leapfrog a better clip unless within 2pts (on 0..1 scale)


def _eligible_for_payoff_rescue(f):
    """Check if a clip is eligible for payoff rescue based on features."""
    # Primary signal: explicit payoff label/span + confidence
    has_payoff = (f.get("payoff_label") in PAYOFF_TYPES) or bool(f.get("payoff_span"))
    payoff_conf = float(f.get("insight_conf", 0.0))  # if you have an explicit payoff_conf, use it instead
    # Secondary signal: strong insight score without explicit payoff labeling
    strong_insight = float(f.get("insight_score", 0.0)) >= INSIGHT_MIN
    return (has_payoff and payoff_conf >= PAYOFF_CONF_MIN) or (strong_insight and payoff_conf >= PAYOFF_CONF_MIN)


def _bump_amount(f):
    """Calculate the bump amount based on payoff/insight strength."""
    # Scale bump by strength
    payoff = float(f.get("payoff_score", 0.0))
    insight = float(f.get("insight_score", 0.0))
    strength = max(payoff, insight)
    return BUMP_HIGH if strength >= 0.35 else BUMP_LOW


def apply_payoff_rescue(sorted_items):
    """
    Apply payoff rescue to boost clips with strong CTAs, punchlines, or insights.
    
    Args:
        sorted_items: list of dicts already carrying:
          - 'calibrated' (0..1)
          - 'features' (dict)
        Must be sorted DESC by 'calibrated' before calling.
    
    Returns:
        The same list with possibly bumped 'calibrated_rescued', preserving order unless within TOP_MARGIN.
    """
    if not sorted_items:
        return sorted_items
    
    # Compute an "upper bound" each item is allowed to reach without unfair leapfrogging.
    # Bound is min(next_higher - TOP_MARGIN, calibrated + 0.08)  # extra safety cap of +8 pts
    for idx, item in enumerate(sorted_items):
        f = item.get("features", {})
        original = float(item.get("calibrated", 0.0))
        
        if not _eligible_for_payoff_rescue(f):
            item["calibrated_rescued"] = original
            item["rescue_reason"] = None
            continue

        bump = _bump_amount(f)
        proposed = original * (1.0 + bump)

        # Non-leapfrog cap
        if idx > 0:
            # previous item is strictly better; we can't overtake it unless within margin
            prev = float(sorted_items[idx-1].get("calibrated_rescued", sorted_items[idx-1].get("calibrated", 0.0)))
            upper = max(prev - TOP_MARGIN, original)  # don't go past prev - margin
        else:
            upper = proposed  # top item can take full bump

        # Extra absolute cap to avoid wild jumps on low scores
        hard_cap = original + 0.08  # at most +8 points on a 0..1 scale
        final = min(proposed, upper, hard_cap)

        item["calibrated_rescued"] = final
        item["rescue_reason"] = "payoff_rescue(+{:.1f}%)".format(bump * 100.0)
        
        # Log the rescue
        if final > original:
            clip_id = item.get("clip", {}).get("id", "unknown")
            duration = item.get("clip", {}).get("end", 0) - item.get("clip", {}).get("start", 0)
            logger.info("PICK_TRACE: virality=%.3f→%.3f (+%.0f%%) reason=%s dur=%.1fs clip=%s",
                       original, final, bump * 100, item["rescue_reason"], duration, clip_id)

    # Keep ranking stable unless changes dictate a swap *within* the allowed margin
    sorted_items.sort(key=lambda x: x.get("calibrated_rescued", x.get("calibrated", 0.0)), reverse=True)
    return sorted_items


def is_length_agnostic_mode():
    """Check if length-agnostic mode is enabled via environment variables."""
    import os
    platform_protect = os.getenv("PLATFORM_PROTECT", "true").lower() == "false"
    pl_v2_weight = float(os.getenv("PL_V2_WEIGHT", "0.5")) == 0.0
    length_cap = os.getenv("LENGTH_CAP_ENABLED", "true").lower() == "false"
    return platform_protect and pl_v2_weight and length_cap
