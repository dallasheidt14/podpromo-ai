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
  enableETag?: boolean;  // Enable ETag caching
  maxBackoff?: number;   // Maximum backoff interval in ms
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
    interval = 1000,  // Reduced from 2000ms to 1000ms
    onSuccess,
    onError,
    on404,
    onComplete,
    enableETag = true,
    maxBackoff = 4000
  } = options;

  let timer: NodeJS.Timeout | null = null;
  let retryCount = 0;
  let lastETag: string | null = null;
  let currentInterval = interval;
  let consecutiveUnchanged = 0;

  const poll = async () => {
    try {
      const headers: HeadersInit = {};
      if (enableETag && lastETag) {
        headers['If-None-Match'] = lastETag;
        console.log(`[POLLING] Request with If-None-Match: ${lastETag}`);
      } else {
        console.log(`[POLLING] Request without ETag`);
      }

      const response = await fetch(url, { headers });
      
      if (response.status === 304) {
        // Data unchanged, implement exponential backoff
        consecutiveUnchanged++;
        if (consecutiveUnchanged > 1) {
          currentInterval = Math.min(currentInterval * 1.5, maxBackoff);
        }
        console.log(`[POLLING] 304 Not Modified - backoff to ${currentInterval}ms`);
        return;
      }

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

      // Reset backoff on successful response
      consecutiveUnchanged = 0;
      currentInterval = interval;

      // Update ETag for next request
      if (enableETag) {
        lastETag = response.headers.get('ETag');
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
    
    const scheduleNext = () => {
      timer = setTimeout(() => {
        poll().then(() => {
          scheduleNext();
        });
      }, currentInterval);
    };
    
    // Start immediately, then schedule subsequent polls
    poll().then(() => {
      scheduleNext();
    });
  };

  const stop = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  return { start, stop };
}