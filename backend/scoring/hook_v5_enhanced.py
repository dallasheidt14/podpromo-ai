# backend/scoring/hook_v5_enhanced.py
import re, numpy as np
from typing import Dict, List, Tuple
from config_loader import get_config
from utils.hooks import get_hook_families_and_meta

FAMILY_PATTERNS: Dict[str, re.Pattern] = {}
FAMILY_PATTERNS_VERSION: str | None = None

def _compile_family_patterns(cfg: Dict):
    global FAMILY_PATTERNS, FAMILY_PATTERNS_VERSION
    # Create a hashable snapshot to detect changes
    fam_bins, _ = get_hook_families_and_meta(cfg)
    snapshot = tuple(sorted((k, tuple(patterns)) for k, patterns in fam_bins.items()))
    version = str(hash(snapshot))
    if version == FAMILY_PATTERNS_VERSION and FAMILY_PATTERNS:
        return
    compiled: Dict[str, re.Pattern] = {}
    for name, patterns in fam_bins.items():
        if not patterns:
            continue
        union = "|".join(f"(?:{p})" for p in patterns)
        compiled[name] = re.compile(union, re.I)
    FAMILY_PATTERNS = compiled
    FAMILY_PATTERNS_VERSION = version

TOKS_EARLY = re.compile(r"[A-Za-z0-9%$''-\.]+")
ANTI_INTRO = re.compile(r"\b(hey|hi|hello|welcome|thanks for|what's up|good (morning|afternoon|evening))\b", re.I)
HEDGE_RE   = re.compile(r"\b(maybe|probably|kinda|sort of|i think|i guess)\b", re.I)

EVIDENCE_PATTERNS = {
    "numbers": re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b|\$)?\b", re.I),
    "comparisons": re.compile(r"\b(?:vs\.?|versus|more than|less than|bigger than|smaller than|better than|worse than)\b", re.I),
    "contrasts": re.compile(r"\b(?:but actually|however|on the contrary|surprisingly|unexpectedly|turns out)\b", re.I),
    "causal": re.compile(r"\b(?:because|therefore|so|which means|as a result|consequently)\b", re.I),
    "imperatives": re.compile(r"\b(?:must|should|need to|have to|don't|stop|start|always|never)\b", re.I),
    "specifics": re.compile(r"\b(?:specifically|exactly|precisely|particularly)\b", re.I),
    "transitions": re.compile(r"\b(first|second|third|then|next|finally|additionally)\b", re.I),
    "definitions": re.compile(r"\b(means|defined as|refers to|is when|called)\b", re.I),
}

def _normalize_quotes(s: str) -> str:
    return (s
        .replace(""", '"').replace(""", '"').replace("„", '"').replace("«", '"').replace("»", '"')
        .replace("'", "'").replace("'", "'").replace("‛", "'").replace("‚", "'")
    )

def _pos_decay(idx: int, k: float) -> float:
    return float(np.exp(-idx / max(1e-6, k)))

def _scan_evidence(text: str) -> tuple[float, List[str]]:
    score = 0.0
    kinds: List[str] = []
    for k, rx in EVIDENCE_PATTERNS.items():
        if rx.search(text):
            score += 0.15
            kinds.append(k)
    return min(score, 1.0), kinds

def _family_hits_weighted(text_tokens: List[str], k: float) -> tuple[Dict[str, float], Dict[str, List[str]]]:
    fam_scores: Dict[str, float] = {}
    fam_spans: Dict[str, List[str]] = {}
    for name, rx in FAMILY_PATTERNS.items():
        acc = 0.0
        spans: List[str] = []
        for j in range(len(text_tokens)):
            win = " ".join(text_tokens[max(0, j-2): j+3])
            m = rx.search(win)
            if m:
                acc += 0.25 * _pos_decay(j, k)
                if len(spans) < 8:
                    spans.append(m.group(0))
        fam_scores[name] = acc / (1.0 + 0.6 * acc)  # saturate
        if spans:
            fam_spans[name] = spans
    return fam_scores, fam_spans

def score_hook_v5_enhanced(
    text: str,
    arousal: float = 0.0,
    q_or_list: float = 0.0,
    genre: str = "general",
    audio_data=None, sr=None, start_time: float = 0.0,
) -> Tuple[float, str, Dict]:
    """
    Returns (hook_score_0_1, reason_str, debug_dict)
    """
    if not text or len(text.strip()) < 8:
        return 0.0, "text_too_short", {}

    text = _normalize_quotes(text)
    t_lower = text.lower()

    cfg = get_config().get("hook_v5", {})
    _compile_family_patterns(cfg)

    k = float(cfg.get("proximity_k_words", 5.0))
    maxw = int(cfg.get("max_words_considered", 40))
    syn_cfg = cfg.get("synergy", {})
    anti_pen = float(cfg.get("anti_intro", {}).get("penalty", 0.05))
    need_after = int(cfg.get("evidence", {}).get("require_after_words", 12))

    toks = TOKS_EARLY.findall(t_lower)[:maxw]
    early = " ".join(toks[:need_after])

    evidence_score, evidence_types = _scan_evidence(text)
    has_evidence = (
        evidence_score >= 0.30
        or bool(re.search(r"\b(how|why|what)\b", early))
        or bool(re.search(r"\b\d+(?:\.\d+)?%?\b", early))
        or bool(re.search(r"\b(?:vs\.?|versus|more than|less than)\b", early))
    )
    evidence_rich = (evidence_score >= 0.45) or (len(evidence_types) >= 3)
    evidence_ok = evidence_rich or has_evidence

    fam_scores, fam_spans = _family_hits_weighted(toks, k)
    family_weights = cfg.get(
        "family_weights",
        {"curiosity": 0.20, "contrarian": 0.15, "howto_list": 0.15, "stakes_risk": 0.15, "authority": 0.10},
    )

    base = 0.0
    for name, val in fam_scores.items():
        base += float(family_weights.get(name, 0.15)) * float(val)

    reasons: List[str] = []
    dbg: Dict = {
        "fam_scores": {k: round(float(v), 4) for k, v in fam_scores.items()},
        "fam_hits": fam_spans,
        "family_weights": family_weights,
        "evidence_score": round(evidence_score, 4),
        "evidence_types": evidence_types,
        "early_token_count": len(toks[:need_after]),
    }

    first_clause = re.split(r"[.!?]\s+", text, maxsplit=1)[0][:180]
    if ANTI_INTRO.search(first_clause):
        base -= anti_pen
        reasons.append("anti_intro")

    if not evidence_ok:
        base = min(base, 0.20)   # evidence guard cap
        reasons.append("evidence_cap_0.20")
    dbg["evidence_ok"] = evidence_ok

    if HEDGE_RE.search(t_lower):
        base *= 0.95
        reasons.append("hedge_soften")
        dbg["hedge_soften"] = True
    else:
        dbg["hedge_soften"] = False

    syn_bonus = 0.0
    if arousal >= float(syn_cfg.get("arousal_gate", 0.60)):
        syn_bonus += float(syn_cfg.get("bonus_each", 0.02)) * min(1.5, 1.0 + (arousal - 0.60) / 0.40)
        reasons.append(f"arousal_synergy_{arousal:.2f}")
    if q_or_list >= float(syn_cfg.get("q_or_list_gate", 0.60)):
        syn_bonus += float(syn_cfg.get("bonus_each", 0.02)) * min(1.5, 1.0 + (q_or_list - 0.60) / 0.40)
        reasons.append(f"q_list_synergy_{q_or_list:.2f}")
    syn_bonus = min(syn_bonus, float(syn_cfg.get("cap_total", 0.08)))
    if syn_bonus:
        base += syn_bonus
    dbg["synergy"] = {"arousal": round(arousal, 3), "q_or_list": round(q_or_list, 3), "bonus": round(syn_bonus, 4)}

    if audio_data is not None and sr is not None:
        try:
            from services.secret_sauce import compute_audio_hook_modifier
            audio_mod = float(compute_audio_hook_modifier(audio_data, sr, start_time))
        except Exception:
            audio_mod = 0.0
        if audio_mod > 0:
            base = min(1.0, base + audio_mod)
            reasons.append(f"audio_{audio_mod:.2f}")
            dbg["audio_modifier"] = round(audio_mod, 4)

    raw01 = float(np.clip(base, 0.0, 1.0))
    dbg["raw"] = round(raw01, 4)
    dbg["base_after_all"] = round(base, 4)
    reason_str = ",".join(reasons) if reasons else "no_modifiers"
    return raw01, reason_str, dbg
