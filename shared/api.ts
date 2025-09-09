export interface ApiResult<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

async function apiFetch<T>(url: string, options: RequestInit = {}): Promise<ApiResult<T>> {
  try {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => undefined);
    if (!res.ok) {
      return { ok: false, error: (data as any)?.error || res.statusText };
    }
    return { ok: true, data };
  } catch (err: any) {
    return { ok: false, error: err?.message || String(err) };
  }
}

export async function getClips(episodeId: string) {
  return apiFetch<any>(`/api/episodes/${episodeId}/clips`);
}

export async function getProgress(episodeId: string) {
  return apiFetch<any>(`/api/progress/${episodeId}`);
}

export async function postGenerateClips(episodeId: string, body: any) {
  return apiFetch<any>(`/api/episodes/${episodeId}/clips`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}

export async function uploadFile(file: File, onProgress?: (percent: number) => void) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/api/upload', {
    method: 'POST',
    body: form
  });
  const data = await res.json().catch(() => undefined);
  return { ok: res.ok, data, error: res.ok ? undefined : data?.error || res.statusText } as ApiResult<any>;
}

export function handleApiResult<T>(result: ApiResult<T>, onSuccess: (data: T) => void, onError: (error: string) => void) {
  if (result.ok && result.data !== undefined) {
    onSuccess(result.data);
  } else {
    onError(result.error || 'Unknown error');
  }
}

export async function ensureOk<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(res.statusText);
  }
  return res.json() as Promise<T>;
}