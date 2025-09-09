export interface Poller {
  start(): void;
  stop(): void;
}

export interface PollingOptions {
  interval?: number;
  onSuccess?: (data: any) => void;
  onError?: (error: string, retryCount: number) => void;
  on404?: () => boolean | void;
  onComplete?: () => void;
}

/**
 * Creates a simple polling helper that repeatedly fetches the given URL
 * until stopped. Callers can provide callbacks for success, errors and
 * completion. The returned object exposes `start` and `stop` methods.
 */
export function createProgressPoller(
  url: string,
  options: PollingOptions = {}
): Poller {
  const {
    interval = 2000,
    onSuccess,
    onError,
    on404,
    onComplete
  } = options;

  let timer: NodeJS.Timeout | null = null;
  let retryCount = 0;

  const poll = async () => {
    try {
      const response = await fetch(url);
      if (response.status === 404) {
        const shouldContinue = on404?.();
        if (shouldContinue === false) {
          stop();
        }
        return;
      }

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      onSuccess?.(data);

      // If the poller reports completion, stop polling
      if (data?.progress?.stage === 'completed') {
        onComplete?.();
        stop();
      }
    } catch (err) {
      retryCount += 1;
      onError?.((err as Error).message, retryCount);
    }
  };

  const start = () => {
    if (timer) return;
    poll();
    timer = setInterval(poll, interval);
  };

  const stop = () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  };

  return { start, stop };
}