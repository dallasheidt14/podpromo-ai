"""
Hook V5 clustering utilities for organizing legacy HOOK_CUES into families
and building evidence-aware hook patterns.
"""

import re
import json
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

# 1) Rule keywords for clustering (keep short/high-precision)
ANTI_INTRO_KEYS = [
    r"\blet me (tell|show) you\b",
    r"\bi want to (share|talk about)\b",
    r"\blisten up\b",
    r"\b(check this out|so today|hey (guys|everyone)|story time)\b",
]

FAMILY_KEYS: Dict[str, List[str]] = {
    "curiosity": [
        r"\bwhat (nobody|no one) (tells|told) you\b",
        r"\bwhat they don'?t want you to know\b",
        r"\bthe (real|hidden) reason\b",
        r"\bbefore you (buy|refi|sell|sign)\b",
        r"\bplot twist\b", r"\bhere'?s the catch\b", r"\byou won'?t believe (how|what)\b",
        r"\bhere'?s why\b", r"\bthe truth (about|is)\b", r"\bthe real reason\b", r"\bwhat if\b",
    ],
    "contrarian": [
        r"\beveryone (thinks|believes)\b.*\bbut\b", r"\bconventional wisdom is wrong\b",
        r"\beveryone (thinks|believes)\b", r"\bbut (actually|in reality)\b",
        r"\bi was wrong\b", r"\bcontrary to\b", r"\bthe biggest lie (in|about)\b",
        r"\bhere'?s why (that|this) advice fails\b", r"\b(stop|don'?t) follow this popular tip\b",
    ],
    "howto_list": [
        r"\bhere'?s how\b", r"\bhere'?s how to\b", r"\btop\s+\d+\b",
        r"\bthe \d+\s*(rules|steps|mistakes)\b",
        r"\b(step|tip|rule)s?\b", r"\b(use|try) this (trick|hack|method)\b",
        r"\bif you only (remember|do) one thing\b", r"\bstart with this\b",
    ],
    "stakes_risk": [
        r"\bthe (costly|fatal|classic) mistake\b", r"\bwhatever you do, don'?t\b",
        r"\bavoid this (trap|gotcha|pitfall)\b", r"\b(or|else) you'?ll (regret|hate)\b",
        r"\bthis (saves|costs) you (\$\d+[km]?|\d+%)\b", r"\bdo this before (you|they) (increase|change|audit)\b",
    ],
    "authority": [
        r"\bafter (\d+|10|20)\+? years (doing|in)\b", r"\bwe analyzed (\d{2,}|thousands|millions)\b",
        r"\bdata (says|shows|proves)\b", r"\bbacked by (data|numbers|evidence)\b",
        r"\bas a (broker|underwriter|creator|engineer)\b", r"\bstudies? show\b", r"\bstatistically speaking\b",
        r"\bin my experience\b", r"\btrust me\b",
    ],
}

# 2) Extra V5 patterns to merge with clustered families
CURIOUS_EXTRA = [
    r"\bwhat (nobody|no one) (tells|told) you\b",
    r"\bwhat they don'?t want you to know\b",
    r"\bthe (one|real) thing (no one|nobody) talks about\b",
    r"\bbefore you (buy|refi|sell|sign)\b",
    r"\bthe (real|hidden) reason\b",
    r"\b(guess|bet) you didn'?t know\b",
    r"\bplot twist\b",
    r"\bthe truth (no one|nobody) tells you\b",
    r"\bhere'?s the catch\b",
    r"\byou won'?t believe (how|what)\b",
    r"\bhere'?s why\b",
]

CONTRA_EXTRA = [
    r"\beveryone (thinks|believes) (?:this|that),? but\b",
    r"\beveryone (thinks|believes)\b",
    r"\bbut actually\b",
    r"\b(conventional|common) wisdom is wrong\b",
    r"\b(X|this) isn'?t (what|how) you think\b",
    r"\b(do|did) (this|that) and you'?re doing it wrong\b",
    r"\bforget everything you learned about\b",
    r"\bthe biggest lie (in|about)\b",
    r"\b(i|we) was wrong about\b",
    r"\bhere'?s why (that|this) advice fails\b",
    r"\b(stop|don'?t) follow this popular tip\b",
]

HOWTO_EXTRA = [
    r"\bhere'?s how to\b",
    r"\b(do|build|fix|avoid) (it|this) in (\d+|one) (step|minute|rule)\b",
    r"\btop\s+\d+\b",
    r"\bthe \d+\s*(rules|steps|mistakes)\b",
    r"\b(simple|proven) framework\b",
    r"\b(use|try) this (trick|hack|method)\b",
    r"\bif you only (remember|do) one thing\b",
    r"\bthe (3|three) pillars of\b",
    r"\bstart with this\b",
]

STAKES_EXTRA = [
    r"\bthe (costly|fatal|classic) mistake\b",
    r"\bwhatever you do, don'?t\b",
    r"\bI (lost|wasted) (\$\d+[km]?|\d+ (years|months)) (by|because)\b",
    r"\bthis (saves|costs) you (\$\d+[km]?|\d+%)\b",
    r"\bdo this before (you|they) (increase|change|audit)\b",
    r"\bavoid this (trap|gotcha|pitfall)\b",
    r"\b(read|watch) this (first|before)\b",
    r"\b(or|else) you'?ll (regret|hate) it\b",
    r"\bthe deadline (no one|nobody) told you about\b",
]

AUTH_EXTRA = [
    r"\bafter (10|20|\d+)\+? years (doing|in)\b",
    r"\bwe analyzed (\d{2,}|thousands|millions) (of )?(loans|clips|deals|files)\b",
    r"\bdata (says|shows|proves)\b",
    r"\bbacked by (data|numbers|evidence)\b",
    r"\bfrom (inside|behind) the scenes\b",
    r"\bas a (broker|underwriter|creator|engineer)\b",
    r"\bI tested (everything|all the options)\b",
    r"\bstatistically speaking\b",
    r"\bstudies? show\b",
]

# 3) Canonical helpers
_WORDS = re.compile(r"[A-Za-z0-9']+")

def _canon(s: str) -> str:
    """Normalize text to canonical form"""
    return " ".join(_WORDS.findall((s or "").lower())).strip()

def _to_regex_from_literal(cue: str) -> str:
    """Turn a plain cue like 'here's how' into a safe word-boundary regex"""
    parts = [re.escape(p) for p in cue.strip().split()]
    if not parts: 
        return ""
    return r"\b" + r"\s+".join(parts) + r"\b"

def _matches_any(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the patterns"""
    return any(re.search(p, text) for p in patterns)

def cluster_hook_cues(hook_cues: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Cluster legacy HOOK_CUES into families.
    
    Returns (families, meta) where families has keys:
      curiosity/contrarian/howto_list/stakes_risk/authority/anti_intro/other
    and meta includes simple coverage stats.
    """
    fam_bins = defaultdict(list)
    moved = {
        "anti_intro": [], "curiosity": [], "contrarian": [], 
        "howto_list": [], "stakes_risk": [], "authority": [], "other": []
    }

    # Precompile our rule sets
    anti_r = [re.compile(p, re.I) for p in ANTI_INTRO_KEYS]
    fam_r = {k: [re.compile(p, re.I) for p in v] for k, v in FAMILY_KEYS.items()}

    for raw in hook_cues or []:
        if not raw: 
            continue
        canon = _canon(raw)
        if not canon: 
            continue
        text = canon

        # 1) Anti-intro first
        if any(r.search(text) for r in anti_r):
            moved["anti_intro"].append(raw)
            fam_bins["anti_intro"].append(_to_regex_from_literal(canon))
            continue

        # 2) Family match
        placed = False
        for fam_name, regs in fam_r.items():
            if any(r.search(text) for r in regs):
                moved[fam_name].append(raw)
                fam_bins[fam_name].append(_to_regex_from_literal(canon))
                placed = True
                break

        if not placed:
            moved["other"].append(raw)
            fam_bins["other"].append(_to_regex_from_literal(canon))

    # De-dupe & tidy
    for k in list(fam_bins.keys()):
        seen = set()
        uniq = []
        for pat in fam_bins[k]:
            if pat and pat not in seen:
                uniq.append(pat)
                seen.add(pat)
        fam_bins[k] = uniq

    # Quick stats
    total = sum(len(v) for v in moved.values())
    meta = {k: sorted(v) for k, v in moved.items()}
    meta["_coverage"] = {k: len(v) for k, v in moved.items()} | {"total": total}
    return fam_bins, meta

def build_hook_families_from_config(cfg: Dict) -> Tuple[Dict[str, List[str]], Dict]:
    """
    Build hook families from config, merging legacy cues with V5 extras.
    """
    # 1) Pull existing cues from config
    existing = (cfg.get("HOOK_CUES") or []) + (cfg.get("HOOK_CUES_EXTRA") or [])
    fam_bins, meta = cluster_hook_cues(existing)

    # 2) Merge with V5 extras
    fam_bins["curiosity"] = (fam_bins.get("curiosity", []) + CURIOUS_EXTRA)
    fam_bins["contrarian"] = (fam_bins.get("contrarian", []) + CONTRA_EXTRA)
    fam_bins["howto_list"] = (fam_bins.get("howto_list", []) + HOWTO_EXTRA)
    fam_bins["stakes_risk"] = (fam_bins.get("stakes_risk", []) + STAKES_EXTRA)
    fam_bins["authority"] = (fam_bins.get("authority", []) + AUTH_EXTRA)

    # 3) Anti-intro: put into config's hook_v5.anti_intro.phrases (as literals)
    anti_list = cfg.get("hook_v5", {}).get("anti_intro", {}).get("phrases", [])
    anti_from_cluster = [re.sub(r"\\b|\\s\+|\\", "", p) for p in fam_bins.get("anti_intro", [])]
    cfg.setdefault("hook_v5", {}).setdefault("anti_intro", {}).setdefault("phrases", [])
    cfg["hook_v5"]["anti_intro"]["phrases"] = sorted(set(anti_list + anti_from_cluster))

    # 4) Return ready-to-use families (regex lists)
    return fam_bins, meta

def get_hook_families_and_meta(cfg: Dict) -> Tuple[Dict[str, List[str]], Dict]:
    """Hook families builder (cached internally)"""
    # Use a simple module-level cache since cfg is not hashable
    if not hasattr(get_hook_families_and_meta, '_cache'):
        get_hook_families_and_meta._cache = {}
    
    # Create a cache key from the relevant config parts
    cache_key = str(sorted(cfg.get("HOOK_CUES", []) + cfg.get("HOOK_CUES_EXTRA", [])))
    
    if cache_key not in get_hook_families_and_meta._cache:
        fam_bins, meta = build_hook_families_from_config(cfg)
        get_hook_families_and_meta._cache[cache_key] = (fam_bins, meta)
    
    return get_hook_families_and_meta._cache[cache_key]

def clear_hook_family_cache():
    """Clear the hook family cache for testing"""
    try:
        if hasattr(get_hook_families_and_meta, '_cache'):
            get_hook_families_and_meta._cache.clear()
    except Exception:
        pass

def print_cluster_report(cfg: Dict, top_other: int = 20) -> str:
    """
    Returns a human-readable report string of how legacy cues
    clustered into V5 families, which moved to anti-intro, and which remain 'other'.
    """
    fam_bins, meta = get_hook_families_and_meta(cfg)
    cov = meta.get("_coverage", {})
    lines = []
    lines.append("Hook V5 — Clustering Coverage")
    lines.append("-" * 36)
    for k in ("curiosity", "contrarian", "howto_list", "stakes_risk", "authority", "anti_intro", "other"):
        lines.append(f"{k:>12}: {cov.get(k, 0)}")
    lines.append(f"{'total':>12}: {cov.get('total', 0)}")
    lines.append("")
    # Show examples so you can prune
    others = meta.get("other", [])[:top_other]
    if others:
        lines.append(f"Unclassified examples (top {len(others)}):")
        for ex in others:
            lines.append(f"  • {ex}")
    # Return + log-friendly JSON block at the end
    lines.append("")
    lines.append("JSON coverage:")
    lines.append(json.dumps(cov, indent=2, sort_keys=True, ensure_ascii=False))
    report = "\n".join(lines)
    return report
