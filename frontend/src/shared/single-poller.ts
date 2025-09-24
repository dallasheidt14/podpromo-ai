/**
 * Single poller manager to ensure only one poller per episode
 */

import { createProgressPoller, Poller, PollingOptions } from './polling';

// Global registry of active pollers
const pollers = new Map<string, { poller: Poller; refCount: number }>();

export function startProgressPoll(episodeId: string, url: string, options: PollingOptions = {}): () => void {
  // Check if poller already exists for this episode
  const existing = pollers.get(episodeId);
  if (existing) {
    // Increment reference count
    existing.refCount++;
    console.log(`[SINGLE_POLLER] Reusing existing poller for ${episodeId} (refCount: ${existing.refCount})`);
    return () => {
      existing.refCount--;
      if (existing.refCount <= 0) {
        console.log(`[SINGLE_POLLER] Stopping poller for ${episodeId}`);
        existing.poller.stop();
        pollers.delete(episodeId);
      } else {
        console.log(`[SINGLE_POLLER] Decremented refCount for ${episodeId} (refCount: ${existing.refCount})`);
      }
    };
  }

  // Create new poller with enhanced options
  const enhancedOptions: PollingOptions = {
    interval: 2000,  // Start with 2s to reduce load
    maxBackoff: 10000,  // Max 10s backoff
    enableETag: true,
    ...options,
    onSuccess: (data) => {
      console.log(`[SINGLE_POLLER] Progress update for ${episodeId}:`, data?.progress?.stage, data?.progress?.percent + '%');
      options.onSuccess?.(data);
    },
    onError: (error, retryCount) => {
      console.warn(`[SINGLE_POLLER] Error for ${episodeId} (attempt ${retryCount}):`, error);
      options.onError?.(error, retryCount);
    },
    onComplete: () => {
      console.log(`[SINGLE_POLLER] Processing completed for ${episodeId}`);
      options.onComplete?.();
    }
  };

  const poller = createProgressPoller(url, enhancedOptions);
  pollers.set(episodeId, { poller, refCount: 1 });
  
  console.log(`[SINGLE_POLLER] Started new poller for ${episodeId}`);
  poller.start();

  return () => {
    const entry = pollers.get(episodeId);
    if (entry) {
      entry.refCount--;
      if (entry.refCount <= 0) {
        console.log(`[SINGLE_POLLER] Stopping poller for ${episodeId}`);
        entry.poller.stop();
        pollers.delete(episodeId);
      } else {
        console.log(`[SINGLE_POLLER] Decremented refCount for ${episodeId} (refCount: ${entry.refCount})`);
      }
    }
  };
}

export function stopAllPollers(): void {
  console.log(`[SINGLE_POLLER] Stopping all ${pollers.size} active pollers`);
  for (const [episodeId, { poller }] of pollers) {
    console.log(`[SINGLE_POLLER] Stopping poller for ${episodeId}`);
    poller.stop();
  }
  pollers.clear();
}

export function getActivePollers(): string[] {
  return Array.from(pollers.keys());
}
