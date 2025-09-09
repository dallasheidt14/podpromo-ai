"""
Typed containers for scoring system with validation and clamping.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def _c01(x: float) -> float:
    """Clamp to [0,1] range"""
    return max(0.0, min(1.0, float(x)))

def quantize(x: float, q: float = 0.01) -> float:
    """Quantize to reduce UI flicker and make scores stable"""
    return round(_c01(x), int(abs(q)).bit_length() or 2)

@dataclass
class Features:
    """Typed container for the 8 core scoring features"""
    hook: float
    arousal: float
    emotion: float
    payoff: float
    info_density: float
    q_list: float
    loopability: float
    platform_length: float
    meta: Dict[str, float] = field(default_factory=dict)
    
    def validate(self, strict: bool = False) -> 'Features':
        """Validate and clamp feature values"""
        vals = [
            self.hook, self.arousal, self.emotion, self.payoff,
            self.info_density, self.q_list, self.loopability, self.platform_length
        ]
        
        if strict and any(not (0 <= v <= 1) for v in vals):
            raise ValueError(f"Feature outside [0,1]: {dict(zip(['hook', 'arousal', 'emotion', 'payoff', 'info_density', 'q_list', 'loopability', 'platform_length'], vals))}")
        
        # Production safety: clamp all values
        self.hook = _c01(self.hook)
        self.arousal = _c01(self.arousal)
        self.emotion = _c01(self.emotion)
        self.payoff = _c01(self.payoff)
        self.info_density = _c01(self.info_density)
        self.q_list = _c01(self.q_list)
        self.loopability = _c01(self.loopability)
        self.platform_length = _c01(self.platform_length)
        
        # Log anomalies in production (rate-limited)
        if not strict:
            for name, val in zip(['hook', 'arousal', 'emotion', 'payoff', 'info_density', 'q_list', 'loopability', 'platform_length'], vals):
                if not (0 <= val <= 1):
                    logger.warning(f"Feature {name} clamped from {val} to {_c01(val)}")
        
        return self
    
    def quantize(self, q: float = 0.01) -> 'Features':
        """Quantize all features to reduce UI flicker"""
        self.hook = quantize(self.hook, q)
        self.arousal = quantize(self.arousal, q)
        self.emotion = quantize(self.emotion, q)
        self.payoff = quantize(self.payoff, q)
        self.info_density = quantize(self.info_density, q)
        self.q_list = quantize(self.q_list, q)
        self.loopability = quantize(self.loopability, q)
        self.platform_length = quantize(self.platform_length, q)
        return self
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backward compatibility"""
        return {
            'hook': self.hook,
            'arousal': self.arousal,
            'emotion': self.emotion,
            'payoff': self.payoff,
            'info_density': self.info_density,
            'q_list': self.q_list,
            'loopability': self.loopability,
            'platform_length': self.platform_length,
            **self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Features':
        """Create from dictionary"""
        return cls(
            hook=data.get('hook', 0.0),
            arousal=data.get('arousal', 0.0),
            emotion=data.get('emotion', 0.0),
            payoff=data.get('payoff', 0.0),
            info_density=data.get('info_density', 0.0),
            q_list=data.get('q_list', 0.0),
            loopability=data.get('loopability', 0.0),
            platform_length=data.get('platform_length', 0.0),
            meta={k: v for k, v in data.items() if k not in ['hook', 'arousal', 'emotion', 'payoff', 'info_density', 'q_list', 'loopability', 'platform_length']}
        )

@dataclass
class Scores:
    """Typed container for scoring results"""
    raw_paths: Dict[str, float]
    whitened_paths: Optional[Dict[str, float]] = None
    synergy_bonus: float = 0.0
    final: float = 0.0
    grade: str = "F"
    winning_path: str = "balanced"
    debug: Dict[str, any] = field(default_factory=dict)
    
    def validate(self) -> 'Scores':
        """Validate and clamp score values"""
        self.synergy_bonus = _c01(self.synergy_bonus)
        self.final = _c01(self.final)
        return self
    
    def quantize(self, q: float = 0.01) -> 'Scores':
        """Quantize all scores"""
        self.synergy_bonus = quantize(self.synergy_bonus, q)
        self.final = quantize(self.final, q)
        if self.whitened_paths:
            self.whitened_paths = {k: quantize(v, q) for k, v in self.whitened_paths.items()}
        return self

# Feature flag constants
FEATURE_TYPES = True  # Enable typed containers
SYNERGY_MODE = "unified"  # unified vs legacy
PLATFORM_LEN_V = 2  # Platform length version
WHITEN_PATHS = True  # Phase 2 - Enable path whitening
GENRE_BLEND = True  # Phase 2 - Enable genre confidence blending
BOUNDARY_HYSTERESIS = True  # Phase 2 - Enable boundary stability
PROSODY_AROUSAL = True  # Phase 3 - Enable prosody-aware arousal
PAYOFF_GUARD = True  # Phase 3 - Enable payoff evidence guard
CALIBRATION_V = "2025.09.3"  # Phase 3 - Mild calibration version for better mid-range contrast

# Phase 1: Segment caps and filtering
MIN_WORDS = 5
MAX_WORDS = 100
MIN_SEC = 8
MAX_SEC = 60

def _keep(seg: Dict) -> bool:
    """
    Filter segments based on word count and duration caps.
    Applied after segmentation and after any extend/merge/snap step.
    """
    word_count = len(seg.get('text', '').split())
    duration = seg.get('end', 0) - seg.get('start', 0)
    
    return (MIN_WORDS <= word_count <= MAX_WORDS) and (MIN_SEC <= duration <= MAX_SEC)
