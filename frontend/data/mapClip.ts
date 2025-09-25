// data/mapClip.ts
export function mapClip(api: any): Clip {
  return {
    id: api.id,
    title: api.title,
    duration: api.actual_duration ?? api.duration,
    display_score: api.display_score ?? api.clip_score_100,
    clip_score_100: api.clip_score_100,

    // NEW (optional)
    virality_calibrated: api.virality_calibrated,
    virality_pct: api.virality_pct,
    platform_fit: api.platform_fit,
    platform_fit_pct: api.platform_fit_pct,

    confidence: api.confidence,
    start_tc: api.start_tc,
    end_tc: api.end_tc,
  };
}
