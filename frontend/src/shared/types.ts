// frontend/src/shared/types.ts

export type Stage = 
  | "queued"
  | "uploading" 
  | "converting"
  | "transcribing"
  | "scoring"        // NEW
  | "processing"
  | "completed" 
  | "failed" 
  | "error";

export interface Progress {
  stage: Stage;
  percent: number;         // 0..100
  message?: string;
  timestamp?: string;      // ISO
}

export interface ProgressInfo {
  stage: string;
  percentage: number;
  message: string;
  timestamp: string;
}

export interface Episode {
  id: string;
  title: string;
  filename: string;
  originalName: string;
  size: number;
  duration: number;
  status: Stage | "ready";
  uploadedAt: string;
  processedAt?: string;
  error?: string;
}

export interface Clip {
  id: string;
  startTime: number;
  endTime: number;
  start?: number; // Alias for startTime for compatibility
  end?: number;   // Alias for endTime for compatibility
  duration: number;
  score: number;
  title: string;
  text: string;
  raw_text?: string; // Raw text content
  full_transcript?: string; // Full stitched transcript
  transcript?: string; // Alias for text for compatibility
  status: 'generating' | 'completed' | 'failed';
  downloadUrl?: string;
  previewUrl?: string;
  captionsVttUrl?: string;
  error?: string;
  features?: {
    // Core scoring features
    hook_score?: number;
    arousal_score?: number;
    emotion_score?: number;
    question_score?: number;
    payoff_score?: number;
    loopability?: number;
    info_density?: number;
    platform_len_match?: number;
    words_per_sec?: number;
    
    // Additional scoring details
    hook_reasons?: string;
    payoff_type?: string;
    type?: string;
    insight_score?: number;
    
    // Debug and analysis fields
    raw_score?: number;
    display_score?: number;
    clip_score_100?: number;
    confidence?: string;
    confidence_color?: string;
    synergy_mult?: number;
    winning_path?: string;
    path_scores?: Record<string, number>;
    bonuses_applied?: number;
    bonus_reasons?: string;
    
    // Ad detection
    _ad_flag?: boolean;
    _ad_penalty?: number;
    _ad_reason?: string;
    
    // Hook components
    hook_components?: Record<string, any>;
    hook_type?: string;
    hook_confidence?: number;
    audio_modifier?: number;
    laughter_boost?: number;
    time_weighted_score?: number;
  };
  grades?: {
    overall: string;
    hook: string;
    flow: string;
    value: string;
    trend: string;
  };
}

export interface ConfigGet {
  maxFileSize?: number;    // bytes
}

export type ApiOk<T>    = { ok: true; data: T };
export type ApiErr      = { ok: false; error: string };
export type ApiResult<T> = ApiOk<T> | ApiErr;

// --- POST /clips params (mirror of backend ClipGenRequest) ---
export type Strategy = "topk" | "diverse" | "hybrid";

export interface TimeWindow {
  startSec: number;
  endSec: number;
}

export interface LoopSeam {
  enabled: boolean;
  maxGapSec: number;
}

export interface ScoreWeights {
  hook: number;
  emotion: number;
  payoff: number;
  loop: number;
  novelty: number;
}

export interface ClipGenParams {
  targetCount?: number;              // default 12
  minDurationSec?: number;           // default 12
  maxDurationSec?: number;           // default 45
  excludeAds?: boolean;              // default true
  scoreThreshold?: number;           // default 0
  scoreWeights?: ScoreWeights;       // defaults provided by backend
  timeWindows?: TimeWindow[];
  loopSeam?: LoopSeam;               // default enabled, 0.25s
  language?: string;                 // default "en"
  strategy?: Strategy;               // default "topk"
  diversityPenalty?: number;         // default 0.15
  seed?: number | null;
  regenerate?: boolean;              // default false
  notes?: string;
}
