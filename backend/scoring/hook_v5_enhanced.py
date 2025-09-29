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

    # Import hook constants
    from services.secret_sauce_pkg.features import (
        HOOK_FIRST_CLAUSE_WINDOW,
        HOOK_MICRO_RETRIM_MAX_TOKENS, HOOK_MICRO_RETRIM_STEP_BONUS, HOOK_MICRO_RETRIM_MAX_BONUS,
        HOOK_HEDGE_LOOKAHEAD_TOKENS, HOOK_HEDGE_SOFTEN_FACTOR,
        HOOK_START_FILLERS, HOOK_STRONG_STARTERS,
    )

    k = float(cfg.get("proximity_k_words", 5.0))
    maxw = int(cfg.get("max_words_considered", 40))
    syn_cfg = cfg.get("synergy", {})
    anti_pen = float(cfg.get("anti_intro", {}).get("penalty", 0.05))
    need_after = int(cfg.get("evidence", {}).get("require_after_words", 12))
    
    # Use the new first clause window
    clause_window = cfg.get("first_clause_window", HOOK_FIRST_CLAUSE_WINDOW)
    print(f"HOOK: clause_window={clause_window}")

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

    # Micro re-trim: find peak hook cue within first N tokens
    peak_j = None
    max_j = min(HOOK_MICRO_RETRIM_MAX_TOKENS, max(0, len(toks) - 1))
    for j in range(0, max_j + 1):
        t = toks[j].lower()
        # allow matching multi-word phrases cheaply by also peeking at j+1
        t2 = (t + " " + toks[j+1].lower()) if j + 1 < len(toks) else t
        if (t in HOOK_STRONG_STARTERS) or ("truth" in t2) or ("need" in t2) or ("nobody" in t2):
            peak_j = j
            break

    micro_retrim_bonus = 0.0
    if peak_j is not None and peak_j > 0:
        micro_retrim_bonus = min(HOOK_MICRO_RETRIM_MAX_BONUS, HOOK_MICRO_RETRIM_STEP_BONUS * peak_j)
        base += micro_retrim_bonus
        print(f"HOOK: micro_retrim j={peak_j} +{micro_retrim_bonus:.3f}")

    reasons: List[str] = []
    
    # Hedge penalty softening: if a hedge appears among the first 3 tokens but the next few tokens
    # contain a strong, non-filler word, reduce the penalty.
    try:
        first_tokens = [t.lower() for t in toks[:3]]
        lookahead = [t.lower() for t in toks[3:3 + HOOK_HEDGE_LOOKAHEAD_TOKENS]]
        has_early_hedge = any(t in HOOK_START_FILLERS for t in first_tokens)
        has_strong_follow = any(
            (len(t) > 2 and t not in HOOK_START_FILLERS) or (t in HOOK_STRONG_STARTERS)
            for t in lookahead
        )
        
        hedge_pen = float(cfg.get("anti_intro", {}).get("hedge_penalty", 0.03))
        if has_early_hedge and has_strong_follow and hedge_pen > 0:
            hedge_pen *= HOOK_HEDGE_SOFTEN_FACTOR
            reasons.append("hedge_softened")
            print(f"HOOK: hedge_soften applied (lookahead strong) "
                  f"-{hedge_pen/HOOK_HEDGE_SOFTEN_FACTOR:.3f}→-{hedge_pen:.3f}")
        elif has_early_hedge:
            reasons.append("hedge_penalty")
            print(f"HOOK: hedge_penalty applied")
        
        if has_early_hedge:
            base = max(0.0, base - hedge_pen)
    except Exception:
        # fail-safe: don't break scoring if tokenization differs
        pass
    
    dbg: Dict = {
        "fam_scores": {k: round(float(v), 4) for k, v in fam_scores.items()},
        "fam_hits": fam_spans,
        "family_weights": family_weights,
        "evidence_score": round(evidence_score, 4),
        "evidence_types": evidence_types,
        "early_token_count": len(toks[:need_after]),
        "micro_retrim_bonus": round(micro_retrim_bonus, 4),
        "peak_j": peak_j,
    }

    first_clause = re.split(r"[.!?]\s+", text, maxsplit=1)[0][:180]
    if ANTI_INTRO.search(first_clause):
        base -= anti_pen
        reasons.append("anti_intro")

    if not evidence_ok:
        base = min(base, 0.20)   # evidence guard cap
        reasons.append("evidence_cap_0.20")
    dbg["evidence_ok"] = evidence_ok

    # Hedge logic moved to earlier in the function (hedge softening)

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
        print(f"HOOK: synergy bonus q_or_list={q_or_list:.3f} arousal={arousal:.3f} +{syn_bonus:.3f}")
    dbg["synergy"] = {"arousal": round(arousal, 3), "q_or_list": round(q_or_list, 3), "bonus": round(syn_bonus, 4)}

    if audio_data is not None and sr is not None:
        try:
            from services.secret_sauce_pkg import compute_audio_hook_modifier
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
