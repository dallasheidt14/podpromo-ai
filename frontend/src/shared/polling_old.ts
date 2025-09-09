// frontend/src/shared/polling.ts
// Enhanced polling service with proper 404/500 handling and backoff

// Dev logging helper
const isDev = typeof process !== "undefined" && process.env.NODE_ENV !== "production";
function log(...args: any[]) { if (isDev) console.log(...args); }

// Timeout helper for older browsers
function withTimeout(ms: number, signal?: AbortSignal) {
  if ((AbortSignal as any)?.timeout) return (AbortSignal as any).timeout(ms);
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  const combo = new AbortController();
  function abortBoth() { ctrl.abort(); combo.abort(); clearTimeout(id); }
  if (signal) signal.addEventListener("abort", abortBoth, { once: true });
  return ctrl.signal;
}

export interface PollingOptions {
  maxRetries?: number;
  baseInterval?: number;
  maxInterval?: number;
  backoffMultiplier?: number;
  maxBackoffTime?: number;
  onSuccess?: (data: any) => void;
  onError?: (error: string, retryCount: number) => void;
  onComplete?: () => void;
  on404?: (retryCount: number) => boolean; // Return true to continue polling
  on500?: (retryCount: number) => boolean; // Return true to continue polling
}

export class PollingService {
  private active = false;
  private timer: NodeJS.Timeout | null = null;
  private retryCount = 0;
  private uploadTime: number | null = null;
  private controller: AbortController | null = null;
  private lastPercent = 0;
  private visibilityHandler = () => {
    // Visibility change handled in getIntervalForStage
  };

  constructor(private url: string, private options: PollingOptions = {}) {
    this.options = {
      maxRetries: 10,
      baseInterval: 1000,
      maxInterval: 10000,
      backoffMultiplier: 1.5,
      maxBackoffTime: 60000, // 60 seconds
      ...options
    };
  }

  start(uploadTime?: number) {
    if (this.active) {
      log('[POLLING] Already active, ignoring start request');
      return;
    }

    this.active = true;
    this.retryCount = 0;
    this.uploadTime = uploadTime || Date.now();
    this.lastPercent = 0;
    
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", this.visibilityHandler);
    }
    
    log('[POLLING] Starting polling service for:', this.url);
    this.poll();
  }

  stop() {
    if (!this.active) return;
    
    this.active = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    
    if (this.controller) {
      this.controller.abort();
      this.controller = null;
    }
    
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.visibilityHandler);
    }
    
    log('[POLLING] Stopping polling service');
  }

  private async poll() {
    if (!this.active) return;

    try {
      console.log(`[POLLING] Attempt ${this.retryCount + 1} for:`, this.url);
      
      const response = await fetch(this.url, { 
        signal: this.controller?.signal 
      });
      
      console.log('[POLLING] Response status:', response.status);

      // Handle different response types
      if (response.status === 404) {
        await this.handle404();
        return;
      }
      
      if (response.status >= 500) {
        await this.handle500(response.status);
        return;
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Parse response
      const data = await response.json();
      console.log('[POLLING] Response data:', data);

      // Reset retry count on success
      this.retryCount = 0;
      
      // Call success handler
      this.options.onSuccess?.(data);
      
      // Schedule next poll
      this.scheduleNextPoll(data);

    } catch (error: any) {
      if (!this.active) return;
      
      console.error('[POLLING] Polling error:', error);
      
      if (error.name === 'AbortError') {
        console.log('[POLLING] Request aborted');
        return;
      }

      await this.handleError(error);
    }
  }

  private async handle404() {
    const timeSinceUpload = this.uploadTime ? Date.now() - this.uploadTime : 0;
    const shouldContinue = this.options.on404?.(this.retryCount) ?? 
      (timeSinceUpload < 60000); // Default: continue for 60 seconds after upload

    if (shouldContinue && this.retryCount < (this.options.maxRetries || 10)) {
      console.log(`[POLLING] 404 - Episode not ready yet (${timeSinceUpload}ms since upload), retrying...`);
      this.retryCount++;
      this.scheduleNextPoll(null, 2000); // Quick retry for 404s
    } else {
      console.log('[POLLING] 404 - Episode not found, stopping polling');
      this.options.onError?.('Episode not found', this.retryCount);
      this.stop();
    }
  }

  private async handle500(status: number) {
    const shouldContinue = this.options.on500?.(this.retryCount) ?? 
      (this.retryCount < 3); // Default: retry 500s up to 3 times

    if (shouldContinue && this.retryCount < (this.options.maxRetries || 10)) {
      console.log(`[POLLING] ${status} - Server error, retrying with backoff...`);
      this.retryCount++;
      const backoffTime = this.calculateBackoff();
      this.scheduleNextPoll(null, backoffTime);
    } else {
      console.log(`[POLLING] ${status} - Server error, max retries reached`);
      this.options.onError?.(`Server error (${status})`, this.retryCount);
      this.stop();
    }
  }

  private async handleError(error: any) {
    this.retryCount++;
    
    if (this.retryCount >= (this.options.maxRetries || 10)) {
      console.log('[POLLING] Max retries reached, stopping polling');
      this.options.onError?.(error.message || 'Network error', this.retryCount);
      this.stop();
    } else {
      console.log(`[POLLING] Error, retrying ${this.retryCount}/${this.options.maxRetries}:`, error.message);
      const backoffTime = this.calculateBackoff();
      this.scheduleNextPoll(null, backoffTime);
    }
  }

  private calculateBackoff(): number {
    const baseInterval = this.options.baseInterval || 1000;
    const multiplier = this.options.backoffMultiplier || 1.5;
    const maxInterval = this.options.maxInterval || 10000;
    const maxBackoffTime = this.options.maxBackoffTime || 60000;
    
    // Exponential backoff with jitter
    const backoffTime = Math.min(
      baseInterval * Math.pow(multiplier, this.retryCount - 1),
      maxInterval
    );
    
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.1 * backoffTime;
    const finalTime = Math.min(backoffTime + jitter, maxBackoffTime);
    
    console.log(`[POLLING] Backoff time: ${finalTime}ms (retry ${this.retryCount})`);
    return finalTime;
  }

  private scheduleNextPoll(data: any, customInterval?: number) {
    if (!this.active) return;

    let interval: number;
    
    if (customInterval) {
      interval = customInterval;
    } else if (data?.progress?.stage) {
      interval = this.getAdaptiveInterval(data.progress.stage);
    } else {
      interval = this.options.baseInterval || 1000;
    }

    console.log(`[POLLING] Scheduling next poll in ${interval}ms`);
    
    this.timer = setTimeout(() => {
      if (this.active) {
        this.poll();
      }
    }, interval);
  }

  private getAdaptiveInterval(stage: string): number {
    // Adaptive polling based on processing stage
    switch (stage.toLowerCase()) {
      case 'uploading': return 500;
      case 'queued': return 2000;
      case 'transcribing':
      case 'transcription': return 3000;
      case 'processing':
      case 'scoring': return 2000;
      case 'generating':
      case 'finalizing': return 1000;
      case 'completed': return 5000;
      case 'error': return 10000;
      default: return 1500;
    }
  }

  isActive(): boolean {
    return this.active;
  }

  getRetryCount(): number {
    return this.retryCount;
  }
}

// Helper function for easy usage
export function createProgressPoller(
  episodeId: string, 
  options: PollingOptions
): PollingService {
  return new PollingService(`/api/progress/${episodeId}`, options);
}
