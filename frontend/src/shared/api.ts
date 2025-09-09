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

// Optional: naive exponential backoff for transient failures
async function sleep(ms: number) { return new Promise(res => setTimeout(res, ms)); }

async function getJson<T>(url: string, init?: RequestInit, retries = 0): Promise<ApiResult<T>> {
  try {
    const res = await fetch(url, init);
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const msg = `Non-JSON response: ${res.status}`;
      if (retries > 0 && res.status >= 500) {
        await sleep(400 * retries);
        return getJson<T>(url, init, retries - 1);
      }
      return { ok: false, error: msg };
    }
    const json = await res.json();
    if (!res.ok || json?.ok === false) {
      const msg = json?.error ?? `HTTP ${res.status}`;
      if (retries > 0 && res.status >= 500) {
        await sleep(400 * retries);
        return getJson<T>(url, init, retries - 1);
      }
      return { ok: false, error: msg };
    }
    // Success
    return { ok: true, data: json as T };
  } catch (err: any) {
    if (retries > 0) {
      await sleep(400 * retries);
      return getJson<T>(url, init, retries - 1);
    }
    return { ok: false, error: err?.message ?? "Network error" };
  }
}

// --- Public typed wrappers ---

export async function getConfig(): Promise<ApiResult<ConfigGet>> {
  return getJson<ConfigGet>(apiUrl("/api/config/get"));
}

export async function getEpisodes(): Promise<ApiResult<{ episodes: Episode[] }>> {
  return getJson<{ episodes: Episode[] }>(apiUrl("/api/episodes"));
}

export async function getProgress(episodeId: string): Promise<ApiResult<{ progress: Progress }>> {
  return getJson<{ progress: Progress }>(apiUrl(`/api/progress/${encodeURIComponent(episodeId)}`));
}

export async function getClips(episodeId: string): Promise<ApiResult<{ clips: Clip[] }>> {
  return getJson<{ clips: Clip[] }>(apiUrl(`/api/episodes/${encodeURIComponent(episodeId)}/clips`));
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
    const r = await getProgress(episodeId);
    if (!r.ok) throw new Error((r as ApiErr).error);
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
  const result = await fetch(apiUrl(`/api/clips/${clipId}/title`), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  
  if (result.ok) {
    return { ok: true, data: undefined };
  } else {
    const error = await result.text();
    return { ok: false, error };
  }
}
