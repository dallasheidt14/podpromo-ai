// frontend/src/shared/api.ts
import type {
  ApiResult, ApiOk, ApiErr,
  Progress, Episode, Clip, ConfigGet,
  ClipGenParams
} from "./types";

// Title management types
export interface TitleGenRequest {
  platform: 'shorts' | 'tiktok' | 'reels' | 'youtube';
  n?: number;
  seed?: number;
  allow_emoji?: boolean;
}

export interface TitleGenResponse {
  platform: string;
  variants: string[];
  chosen: string;
  meta: Record<string, any>;
}

export interface TitleSetRequest {
  platform: 'shorts' | 'tiktok' | 'reels' | 'youtube';
  title: string;
}

// API base URL - defaults to same origin, can be overridden with NEXT_PUBLIC_API_BASE
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
export const apiUrl = (path: string) => `${API_BASE}${path}`;

// Get auth token from localStorage (for signed URLs)
const getAuthToken = (): string => {
  if (typeof window === 'undefined') return '';
  return localStorage.getItem('token') || '';
};

// =============================
// Safe Fetch Utilities (global)
// =============================
const ERROR_SLICE = 200; // keep to 200 chars for readable toasts
const DEBUG_FALLBACKS = false; // flip to true when diagnosing payload shapes

export async function getJson<T = any>(input: RequestInfo, init?: RequestInit): Promise<ApiResult<T>> {
  try {
    const res = await fetch(input, init);
    const ct = res.headers.get("content-type") || "";
    let parsed: any = undefined;
    if (ct.includes("application/json")) {
      parsed = await res.json().catch(() => undefined);
    } else {
      // Avoid "Unexpected token 'I'..." crashes on plain text/HTML
      parsed = await res.text().catch(() => undefined);
    }
    const error = !res.ok
      ? (typeof parsed === "object" && parsed
          ? (parsed.detail || parsed.error || parsed.message || res.statusText)
          : (typeof parsed === "string" ? parsed.slice(0, ERROR_SLICE) : res.statusText))
      : undefined;
    return {
      ok: res.ok,
      data: typeof parsed === "object" ? (parsed as T) : undefined,
      error,
      status: res.status,
      requestId: res.headers.get("x-request-id") || undefined,
    };
  } catch (e: any) {
    return { ok: false, error: e?.message || "network_error" };
  }
}

function parseClips<T = any>(payload: any): T[] {
  // Supports: [{...}] OR {clips:[...]} OR {data:{clips:[...]}}
  if (Array.isArray(payload)) return payload as T[];
  if (payload?.clips) return payload.clips as T[];
  const nested = payload?.data?.clips;
  if (Array.isArray(nested)) return nested as T[];
  if (DEBUG_FALLBACKS) {
    // eslint-disable-next-line no-console
    console.warn("[clips] unknown payload shape", { keys: Object.keys(payload || {}) });
  }
  return [];
}

// --- Public typed wrappers ---

export async function getConfig(): Promise<ApiResult<ConfigGet>> {
  return getJson<ConfigGet>(apiUrl("/api/config/get"));
}

export async function getEpisodes(): Promise<ApiResult<{ episodes: Episode[] }>> {
  return getJson<{ episodes: Episode[] }>(apiUrl("/api/episodes"));
}

export async function getProgress(): Promise<ApiResult<{ progress: any }>> {
  return getJson<{ progress: any }>(apiUrl("/api/progress"), { cache: "no-store" });
}

export async function getEpisodeProgress(id: string): Promise<ApiResult<{ progress: any }>> {
  return getJson<{ progress: any }>(apiUrl(`/api/progress/${encodeURIComponent(id)}`), {
    cache: "no-store",
  });
}

export async function getClips(episodeId: string): Promise<ApiResult<{ clips: Clip[] }>> {
  const r = await getJson<any>(
    apiUrl(`/api/episodes/${encodeURIComponent(episodeId)}/clips?ts=${Date.now()}`),
    { cache: "no-store" }
  );
  if (!r.ok) throw new Error(r.error || `clips_http_${r.status ?? "unknown"}`);
  const clips = parseClips<Clip>(r.data);
  return { ok: true, data: { clips } };
}

export async function getClipsSimple(episodeId: string): Promise<Clip[]> {
  const r = await getJson<any>(
    apiUrl(`/api/episodes/${encodeURIComponent(episodeId)}/clips?ts=${Date.now()}`),
    { cache: "no-store" }
  );
  if (!r.ok) throw new Error(r.error || `clips_http_${r.status ?? "unknown"}`);
  const clips = parseClips<Clip>(r.data);
  // Normalize media URLs to absolute backend URLs (handles "/clips/..." etc.)
  const absolutize = (u?: string) =>
    !u ? u : (u.startsWith("http://") || u.startsWith("https://")) ? u : apiUrl(u);
  return clips.map((c: any) => ({
    ...c,
    preview_url: absolutize(c.preview_url || c.previewUrl || c.video_url),
    audio_url: absolutize(c.audio_url || c.audioUrl),
  }));
}

export async function uploadYouTube(url: string): Promise<ApiResult<{ episode_id: string }>> {
  const form = new FormData();
  form.append("url", url);
  return getJson<{ episode_id: string }>(apiUrl("/api/upload-youtube"), {
    method: "POST",
    body: form,
  });
}

// New API helpers for the event-driven system
export type UploadYouTubeError =
  | "invalid_url"
  | "too_short"
  | "too_long"
  | "live_stream_not_supported"
  | "bot_detection"
  | "video_unavailable"
  | "private_video"
  | "download_failed"
  | "audio_conversion_failed"
  | "youtube_disabled"
  | "internal_error";

export async function uploadYouTubeSimple(url: string): Promise<{ episode_id: string }> {
  const form = new FormData();
  form.append("url", url);
  const r = await getJson<{ episode_id: string }>(apiUrl("/api/upload-youtube"), { method: "POST", body: form });
  if (!r.ok) {
    let code: UploadYouTubeError = "internal_error";
    if (r.data?.detail) code = r.data.detail as UploadYouTubeError;
    throw new Error(code);
  }
  return r.data!;
}

export type ProgressStage =
  | "queued" | "fetching_metadata" | "downloading" | "extracting_audio"
  | "transcribing" | "scoring" | "processing" | "completed" | "error";

export interface ProgressResponse {
  stage: ProgressStage;
  percent?: number;
  message?: string;
}

export async function getProgressSimple(episodeId: string): Promise<ProgressResponse> {
  const r = await getJson<any>(apiUrl(`/api/progress/${episodeId}`), { cache: 'no-store' });
  if (!r.ok) throw new Error(r.error || `progress_http_${r.status ?? "unknown"}`);
  // normalize both shapes: {ok:true, progress:{...}} or just {...}
  return (r.data && (r.data.progress || r.data));
}

// Types are additive; keep your existing Clip type/shape untouched elsewhere.
export interface ClipSimple {
  id: string;
  start: number;
  end: number;
  title?: string;
  previewUrl?: string;   // some backends
  preview_url?: string;  // others
  video_url?: string;    // fallback
  raw_text?: string;
  full_transcript?: string;
}

// This function is already updated above with parseClips

export function isTerminalProgress(p?: { stage?: string; percent?: number } | null) {
  if (!p) return false;
  return p.stage === "completed" || (p.stage === "scoring" && (p.percent ?? 0) >= 100);
}

export async function uploadFile(
  file: File, 
  onProgress?: (progress: number) => void
): Promise<ApiResult<{ episodeId: string }>> {
  const formData = new FormData();
  formData.append('file', file);
  
  // Create AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 1800000); // 30 minutes timeout
  
  try {
    // Use XMLHttpRequest for upload progress tracking
    const result = await new Promise<ApiResult<{ episodeId: string }>>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      // Track upload progress
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      });
      
      // Handle completion
      xhr.addEventListener('load', () => {
        clearTimeout(timeoutId);
        try {
          const response = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve({ ok: true, data: response });
          } else {
            resolve({ ok: false, error: response?.error || `HTTP ${xhr.status}` });
          }
        } catch (err) {
          resolve({ ok: false, error: 'Invalid response format' });
        }
      });
      
      // Handle errors
      xhr.addEventListener('error', () => {
        clearTimeout(timeoutId);
        resolve({ ok: false, error: 'Network error during upload' });
      });
      
      // Handle abort
      xhr.addEventListener('abort', () => {
        clearTimeout(timeoutId);
        resolve({ ok: false, error: 'Upload cancelled or timed out' });
      });
      
      // Start upload
      xhr.open('POST', apiUrl('/api/upload'));
      xhr.send(formData);
      
      // Handle timeout
      controller.signal.addEventListener('abort', () => {
        xhr.abort();
      });
    });
    
    return result;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      return { ok: false, error: 'Upload timeout - file may be too large or connection too slow' };
    }
    throw error;
  }
}

export async function postGenerateClips(
  episodeId: string,
  params: ClipGenParams = {},
  idempotencyKey: string = (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`)
): Promise<ApiResult<{
  ok: true;
  started: boolean;
  episodeId: string;
  cached?: boolean;
  job?: { id: string; idempotencyKey?: string };
  next?: { progress?: string; results?: string };
}>> {
  return getJson(apiUrl(`/api/episodes/${encodeURIComponent(episodeId)}/clips`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Idempotency-Key": idempotencyKey
    },
    body: JSON.stringify(params ?? {}),
  }, /*retries*/ 1);
}

// --- Small helpers often handy in UI layers ---

export function ensureOk<T>(res: ApiResult<T>): T {
  if (!res.ok) throw new Error((res as ApiErr).error);
  return res.data;
}

// Utility function for handling API results in components
export function handleApiResult<T>(
  result: ApiResult<T>,
  onSuccess: (data: T) => void,
  onError: (error: string) => void
): void {
  if (result.ok) {
    onSuccess(result.data);
  } else {
    onError((result as ApiErr).error);
  }
}

// Example: polling progress until completed
export async function waitForCompletion(episodeId: string, opts: { intervalMs?: number; timeoutMs?: number } = {}) {
  const start = Date.now();
  const interval = opts.intervalMs ?? 3000;
  const timeout = opts.timeoutMs ?? 15 * 60 * 1000;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const r = await getEpisodeProgress(episodeId);
    if (!r.ok) throw new Error(r.error || "progress_fetch_failed");
    const st = r.data.progress.stage;
    if (st === "completed") return r.data.progress;
    if (st === "failed") throw new Error("Processing failed");
    if (Date.now() - start > timeout) throw new Error("Timed out waiting for completion");
    await sleep(interval);
  }
}

// Title management functions
export async function generateTitles(clipId: string, request: TitleGenRequest): Promise<ApiResult<TitleGenResponse>> {
  return getJson<TitleGenResponse>(apiUrl(`/api/clips/${clipId}/titles`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
}

export async function setClipTitle(clipId: string, request: TitleSetRequest): Promise<ApiResult<void>> {
  const r = await getJson<void>(apiUrl(`/api/clips/${clipId}/title`), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  return r;
}

export async function saveTitles(clipId: string, platform: string, titles: string[]): Promise<ApiResult<{ok: boolean, count: number, platform: string}>> {
  // Clean and dedupe titles
  const cleanTitles = [...new Set(titles.filter(t => t && t.trim().length >= 3).map(t => t.trim()))];
  
  if (cleanTitles.length === 0) {
    return { success: false, error: "No valid titles provided" };
  }
  
  const r = await getJson<{ok: boolean, count: number, platform: string}>(apiUrl(`/api/clips/${clipId}/titles/save`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      platform: platform,
      titles: cleanTitles
    })
  });
  return r;
}

// =============================
// Signed URL Functions (Security)
// =============================

export interface SignedUrlResponse {
  url: string;
  expires_at: number;
  expires_in: number;
}

export async function getSignedDownload(
  area: "uploads" | "clips", 
  name: string, 
  ttl: number = 300
): Promise<ApiResult<SignedUrlResponse>> {
  const token = getAuthToken();
  const r = await getJson<SignedUrlResponse>(apiUrl(`/api/signed-download?area=${area}&name=${encodeURIComponent(name)}&ttl=${ttl}`), {
    headers: { 
      'Authorization': `Bearer ${token}`,
      'Cache-Control': 'no-store'
    }
  });
  return r;
}

export function downloadFile(signedUrl: string): void {
  // Open the signed URL in a new tab/window for download
  window.open(`${API_BASE}${signedUrl}`, '_blank');
}

export async function downloadFileWithSignedUrl(area: "uploads" | "clips", name: string): Promise<void> {
  const result = await getSignedDownload(area, name);
  if (result.success) {
    downloadFile(result.data.url);
  } else {
    throw new Error(`Failed to get signed URL: ${result.error}`);
  }
}
