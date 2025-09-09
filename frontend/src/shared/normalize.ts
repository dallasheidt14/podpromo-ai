// frontend/src/shared/normalize.ts
import type { Clip, Progress, ProgressInfo, Stage } from "./types";

const STAGE_MAP: Record<string, Stage> = {
  queued: "queued",
  queue: "queued",
  waiting: "queued",
  preparing: "queued",
  uploading: "uploading",
  upload: "uploading",
  converting: "converting",
  convert: "converting",
  muxing: "converting",
  transcribing: "transcribing",
  transcribe: "transcribing",
  asr: "transcribing",
  scoring: "scoring",
  analyze: "processing",
  analysing: "processing",
  processing: "processing",
  process: "processing",
  generate: "processing",
  generating: "processing",
  completed: "completed",
  complete: "completed",
  done: "completed",
  success: "completed",
  error: "error",
  failed: "error",
  fail: "error",
};

function normalizeStage(raw: any): Stage {
  const s = String(raw ?? "").toLowerCase().trim();
  return STAGE_MAP[s] ?? "processing";
}

function normalizePercent(raw: any): number {
  if (raw == null) return 0;
  if (typeof raw === "number") {
    // handle 0..1 fractions, clamp
    const v = raw <= 1 && raw >= 0 ? raw * 100 : raw;
    return Math.max(0, Math.min(100, Math.round(v)));
  }
  const str = String(raw).trim();
  // "10%" → 10
  const pct = str.endsWith("%") ? parseFloat(str.slice(0, -1)) : parseFloat(str);
  if (Number.isFinite(pct)) {
    const v = pct <= 1 && pct >= 0 ? pct * 100 : pct;
    return Math.max(0, Math.min(100, Math.round(v)));
  }
  return 0;
}

export function normalizeClip(raw: any): Clip {
  const startTime = Number(raw.start_time || raw.startSec || raw.start || 0);
  const endTime = Number(raw.end_time || raw.endSec || raw.end || 0);
  
  // Normalize features with all available fields
  const features = raw.features || {};
  const normalizedFeatures = {
    // Core scoring features
    hook_score: Number(features.hook_score || 0),
    arousal_score: Number(features.arousal_score || 0),
    emotion_score: Number(features.emotion_score || 0),
    question_score: Number(features.question_score || 0),
    payoff_score: Number(features.payoff_score || 0),
    loopability: Number(features.loopability || 0),
    info_density: Number(features.info_density || 0),
    platform_len_match: Number(features.platform_len_match || features.platform_length_match || 0),
    words_per_sec: Number(features.words_per_sec || 0),
    
    // Additional scoring details
    hook_reasons: String(features.hook_reasons || ''),
    payoff_type: String(features.payoff_type || ''),
    type: String(features.type || features.moment_type || 'general'),
    insight_score: Number(features.insight_score || 0),
    
    // Debug and analysis fields
    raw_score: Number(features.raw_score || raw.raw_score || 0),
    display_score: Number(features.display_score || raw.display_score || 0),
    clip_score_100: Number(features.clip_score_100 || raw.clip_score_100 || 0),
    confidence: String(features.confidence || raw.confidence || ''),
    confidence_color: String(features.confidence_color || raw.confidence_color || ''),
    synergy_mult: Number(features.synergy_mult || features.synergy_multiplier || raw.synergy_mult || 1),
    winning_path: String(features.winning_path || raw.winning_path || ''),
    path_scores: features.path_scores || raw.path_scores || {},
    bonuses_applied: Number(features.bonuses_applied || raw.bonuses_applied || 0),
    bonus_reasons: String(features.bonus_reasons || raw.bonus_reasons || ''),
    
    // Ad detection
    _ad_flag: Boolean(features._ad_flag || raw._ad_flag || false),
    _ad_penalty: Number(features._ad_penalty || raw._ad_penalty || 0),
    _ad_reason: String(features._ad_reason || raw._ad_reason || ''),
    
    // Hook components
    hook_components: features.hook_components || {},
    hook_type: String(features.hook_type || ''),
    hook_confidence: Number(features.hook_confidence || 0),
    audio_modifier: Number(features.audio_modifier || 0),
    laughter_boost: Number(features.laughter_boost || 0),
    time_weighted_score: Number(features.time_weighted_score || 0)
  };
  
  // Normalize score to 0-1 range
  const rawScore = Number(raw.score || raw.final_score || raw.composite || raw.raw_score || 0);
  const normalizedScore = rawScore > 1 ? rawScore / 100 : rawScore; // If score > 1, assume it's 0-100 and convert to 0-1

  return {
    id: String(raw.id || raw.clip_id || `clip_${Date.now()}`),
    startTime,
    endTime,
    duration: endTime - startTime,
    score: Math.max(0, Math.min(1, normalizedScore)), // Ensure score is between 0-1
    title: String(raw.title || ''),
    text: String(raw.text || raw.transcript || raw.content || ''),
    status: (raw.status || 'completed') as 'generating' | 'completed' | 'failed',
    features: normalizedFeatures,
    downloadUrl: raw.download_url || raw.downloadUrl || null,
    previewUrl: raw.preview_url || raw.previewUrl ? 
      (raw.preview_url || raw.previewUrl).startsWith('http') ? 
        (raw.preview_url || raw.previewUrl) : 
        `http://localhost:8000${raw.preview_url || raw.previewUrl}` : 
      null,
    error: raw.error || null,
  };
}

export function normalizeProgress(raw: any, last?: { stage: string; percent: number; ts: number }): Progress {
  const now = Date.now();
  const p = raw?.progress ?? raw;
  if (!p) return { stage: "processing", percent: 0, message: "", timestamp: new Date().toISOString() };
  
  // 1) Percent parsing: support percent, percentage; clamp 0..100
  let percent = normalizePercent(p.percent ?? p.percentage ?? p.pct);
  
  // 2) Stage normalization
  const stage = normalizeStage(p.stage ?? p.status ?? p.state);
  
  // 3) Monotonic percent (never go backwards)
  if (last) percent = Math.max(last.percent, percent);
  
  // 4) **Scoring inference**: only infer scoring if we were at high transcribing percent
  // and now see processing with low percent, or if explicitly scoring
  const cameFromHighTranscribe = last?.stage === "transcribing" && last?.percent >= 80;
  const scoringLike = stage === "scoring" ||
    (cameFromHighTranscribe && stage === "processing" && percent < 20);

  let finalStage = stage;
  if (scoringLike && (stage as string) !== "completed" && (stage as string) !== "error") {
    finalStage = "scoring";
    // keep percent at least at 85 if we previously moved past transcribing
    if (last && last.percent >= 80) percent = Math.max(percent, 85);
  }
  
  return {
    stage: finalStage as Stage,
    percent,
    message: p.message ?? "",
    timestamp: p.timestamp ?? p.updated_at ?? new Date().toISOString(),
  };
}

export function normalizeProgressInfo(raw: any): ProgressInfo | null {
  if (!raw) return null;
  
  const p = raw?.progress ?? raw;
  if (!p) return null;
  
  const stage = normalizeStage(p.stage ?? p.status ?? p.state);
  let percentage = normalizePercent(p.percent ?? p.percentage ?? p.pct);
  
  // If stage is completed, ensure percentage is 100
  if (stage === "completed") {
    percentage = 100;
  }
  
  return {
    stage: String(stage),
    percentage,
    message: String(p.message || ""),
    timestamp: String(p.timestamp || p.updated_at || new Date().toISOString()),
  };
}
