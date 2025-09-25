// ui/score.ts
export function getViralityPct(clip: Partial<Clip>): number {
  if (typeof clip.virality_pct === 'number') return clip.virality_pct;
  if (typeof clip.virality_calibrated === 'number') return Math.round(clip.virality_calibrated * 100);
  // fall back to legacy scale by de-clamping 45–100 -> ~0–100 guess
  if (typeof clip.display_score === 'number') {
    const x = Math.max(0, Math.min(100, clip.display_score));
    return Math.round(((x - 45) / 55) * 100); // not perfect, but better than showing 50 caps
  }
  if (typeof clip.clip_score_100 === 'number') return clip.clip_score_100;
  return 0;
}

export function getPlatformFitPct(clip: Partial<Clip>): number {
  if (typeof clip.platform_fit_pct === 'number') return clip.platform_fit_pct;
  if (typeof clip.platform_fit === 'number') return Math.round(clip.platform_fit * 100);
  return 0;
}

export function scoreToTone(pct: number): 'ok' | 'good' | 'great' | 'fire' {
  if (pct >= 85) return 'fire';
  if (pct >= 70) return 'great';
  if (pct >= 55) return 'good';
  return 'ok';
}

export function toneClasses(tone: ReturnType<typeof scoreToTone>): string {
  switch (tone) {
    case 'fire':  return 'bg-red-600/10 text-red-600 ring-1 ring-red-600/20';
    case 'great': return 'bg-emerald-600/10 text-emerald-600 ring-1 ring-emerald-600/20';
    case 'good':  return 'bg-amber-600/10 text-amber-700 ring-1 ring-amber-600/20';
    default:      return 'bg-slate-600/10 text-slate-600 ring-1 ring-slate-600/20';
  }
}
