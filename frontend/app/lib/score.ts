// frontend/app/lib/score.ts
export const toPct = (v: unknown, digits = 0): number => {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return 0;
  const base = n <= 1 ? n : n / 100;     // accept 0–1 or 0–100
  return Number(Math.max(0, Math.min(100, base * 100)).toFixed(digits));
};

export const buildScoreBundle = (clip: any) => {
  const viralityRaw =
    clip?.virality ??
    clip?.final_score ??
    clip?.display_score ??
    clip?.score ?? 0;

  return {
    viralityPct: toPct(viralityRaw),
    hookPct: toPct(clip?.hook_score ?? clip?.features?.hook_score ?? 0),
    arousalPct: toPct(clip?.arousal_score ?? clip?.features?.arousal_score ?? 0),
    payoffPct: toPct(clip?.payoff_score ?? clip?.features?.payoff_score ?? 0),
    infoPct: toPct(clip?.info_density ?? clip?.features?.info_density ?? 0),
    qlistPct: toPct(clip?.q_list_score ?? clip?.features?.q_list_score ?? 0),
  };
};
