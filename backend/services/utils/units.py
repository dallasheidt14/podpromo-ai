"""
Unit conversion utilities for consistent rate/density calculations.
"""

def per_min(count: float, duration_s: float) -> float:
    """Convert count to rate per minute"""
    dur_min = max(1e-6, duration_s / 60.0)
    return float(count) / dur_min

def per_sec(count: float, duration_s: float) -> float:
    """Convert count to rate per second"""
    dur_s = max(1e-6, duration_s)
    return float(count) / dur_s
