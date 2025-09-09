export interface PollingOptions {
  intervalMs?: number;
  onSuccess: (data: any) => void;
  onError?: (error: string, retryCount: number) => void;
  on404?: () => void | boolean;
  onComplete?: () => void;
}

export interface Poller {
  start: () => void;
  stop: () => void;
}

export function createProgressPoller(url: string, options: PollingOptions) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;
  let retries = 0;

  const poll = async () => {
    if (stopped) return;
    try {
      const res = await fetch(url);
      if (res.status === 404) {
        const shouldContinue = options.on404?.();
        if (shouldContinue === false) {
          stop();
          return;
        }
      } else {
        const data = await res.json();
        options.onSuccess(data);
      }
    } catch (err: any) {
      retries++;
      options.onError?.(err?.message || String(err), retries);
    }

    timer = setTimeout(poll, options.intervalMs ?? 2000);
  };

  const start = () => {
    stopped = false;
    poll();
  };

  const stop = () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };

  return { start, stop } as Poller;
}
