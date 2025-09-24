// frontend/lib/progressPoller.ts
type Progress = { stage: string; message: string; percent: number };

const etags = new Map<string, string>();
const controllers = new Map<string, AbortController>();
const lastStates = new Map<string, Progress>();
const timers = new Map<string, number>();

const BASE_MS = 1000;
const MAX_MS = 8000;

function eq(a?: Progress, b?: Progress) {
  if (!a || !b) return false;
  return a.stage === b.stage && a.message === b.message && a.percent === b.percent;
}

export function startProgressPoller(
  episodeId: string,
  onUpdate: (p: Progress) => void,
) {
  // De-dupe: only one poller per episode
  if (controllers.has(episodeId)) return;

  let delay = BASE_MS;
  const ctrl = new AbortController();
  controllers.set(episodeId, ctrl);

  const tick = async () => {
    if (!controllers.has(episodeId)) return; // stopped
    try {
      const headers: Record<string, string> = {};
      const prevEtag = etags.get(episodeId);
      if (prevEtag) headers["If-None-Match"] = prevEtag;

      const res = await fetch(`/api/progress/${episodeId}`, {
        method: "GET",
        headers,
        signal: ctrl.signal,
      });

      if (res.status === 304) {
        // no change → backoff up to MAX_MS
        delay = Math.min(MAX_MS, Math.floor(delay * 1.5));
        console.log(`[POLLING] 304 Not Modified - backoff to ${delay}ms`);
      } else if (res.ok) {
        const et = res.headers.get("ETag") || undefined;
        if (et) etags.set(episodeId, et);
        const data = await res.json();
        const p: Progress = data?.progress || data;

        const prev = lastStates.get(episodeId);
        if (!eq(prev, p)) {
          lastStates.set(episodeId, p);
          onUpdate(p);
          // if progress changed, tighten cadence again
          delay = BASE_MS;
        } else {
          // unchanged body despite 200 (rare) → gentle backoff
          delay = Math.min(MAX_MS, Math.floor(delay * 1.5));
        }
      } else {
        // error → backoff but keep trying
        delay = Math.min(MAX_MS, Math.floor(delay * 1.8));
      }
    } catch (e) {
      if ((e as any).name === "AbortError") return;
      delay = Math.min(MAX_MS, Math.floor(delay * 1.8));
    } finally {
      if (controllers.has(episodeId)) {
        const t = window.setTimeout(tick, delay);
        timers.set(episodeId, t);
      }
    }
  };

  tick();
}

export function stopProgressPoller(episodeId: string) {
  const c = controllers.get(episodeId);
  if (c) {
    c.abort();
    controllers.delete(episodeId);
  }
  const t = timers.get(episodeId);
  if (t) {
    clearTimeout(t);
    timers.delete(episodeId);
  }
}
