import { Clip, ProgressInfo } from './types';

/**
 * Normalizes a clip object from various backend formats into the Clip shape
 * used by the frontend. Unknown properties are preserved.
 */
export function normalizeClip(raw: any): Clip {
  if (!raw) {
    return {
      id: '',
      start: 0,
      end: 0,
      duration: 0,
    } as Clip;
  }

  const start = Number(raw.start ?? raw.start_time ?? raw.startSec ?? 0);
  const end = Number(
    raw.end ?? raw.end_time ?? raw.endSec ?? (start + (raw.duration ?? raw.length ?? 0))
  );

  const clip: Clip = {
    ...raw,
    id: String(raw.id ?? raw.clip_id ?? raw._id ?? ''),
    start,
    end,
    duration: Number(raw.duration ?? raw.length ?? end - start),
    score: raw.score ?? raw.rank ?? raw.rating ?? raw.probability,
    url: raw.url ?? raw.video_url ?? raw.clip_url,
    transcript: raw.transcript ?? raw.text ?? raw.caption,
    is_advertisement:
      raw.is_advertisement ?? raw._ad_flag ?? raw.features?.is_advertisement ?? false,
  };

  return clip;
}

/**
 * Extracts progress information from an API response. It accepts either a
 * response object containing a `progress` field or a progress object itself.
 */
export function normalizeProgressInfo(resp: any): ProgressInfo {
  const progress = resp?.progress ?? resp;
  return normalizeProgress(progress);
}

/**
 * Normalizes raw progress info into a consistent shape.
 */
export function normalizeProgress(progress: any): ProgressInfo {
  return {
    stage: progress?.stage ?? progress?.status ?? '',
    percentage: Number(
      progress?.percentage ?? progress?.percent ?? progress?.progress ?? 0
    ),
    eta:
      progress?.eta ??
      progress?.eta_seconds ??
      progress?.remaining ??
      progress?.remaining_seconds ??
      null,
    ...progress,
  };
}