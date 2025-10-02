// types/Clip.ts
export interface PlatformRecommendation {
  platform: string;
  fit_score: number;
  reason: string;
}

export interface Clip {
  id: string;
  title?: string;
  duration?: number;         // seconds
  // existing:
  display_score?: number;    // legacy 45–100
  clip_score_100?: number;   // legacy alias in some payloads

  // NEW — safe to be optional
  virality_calibrated?: number; // 0–1
  virality_pct?: number;        // 0–100
  platform_fit?: number;        // 0–1
  platform_fit_pct?: number;    // 0–100
  platform_recommendations?: PlatformRecommendation[];

  // nice-to-have if you return them:
  confidence?: string;       // "Low" | "Med" | "High"
  start_tc?: string;
  end_tc?: string;
  // ...rest of your fields
}
