import { useEffect, useRef } from "react";
import { getProgressSimple, type ProgressResponse, apiUrl } from "../shared/api";
import { notifyClipsReady } from "../shared/events";
import { isTerminalProgress } from "../shared/progress";
import { startProgressPoll } from "../shared/single-poller";

type Options = {
  onUpdate?: (p: ProgressResponse) => void;
  onDone?: () => void;
  intervalMs?: number;
};

export function useProgressPoller(episodeId?: string, opts: Options = {}) {
  const { onUpdate, onDone, intervalMs = 1000 } = opts;
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!episodeId) return;

    const isTerminal = (p: ProgressResponse) => {
      return isTerminalProgress(p) || p.stage === "error";
    };

    // Use single poller per episode with ETag support
    const url = apiUrl(`/api/progress/${episodeId}`);
    stopRef.current = startProgressPoll(episodeId, url, {
      interval: intervalMs,
      onSuccess: (data) => {
        const progress = data?.progress || data;
        if (progress) {
          onUpdate?.(progress);
          if (isTerminal(progress)) {
            notifyClipsReady(episodeId);
            onDone?.();
          }
        }
      },
      onError: (error, retryCount) => {
        console.warn(`[useProgressPoller] Error for ${episodeId} (attempt ${retryCount}):`, error);
      },
      onComplete: () => {
        notifyClipsReady(episodeId);
        onDone?.();
      }
    });

    return () => {
      if (stopRef.current) {
        stopRef.current();
        stopRef.current = null;
      }
    };
  }, [episodeId, intervalMs, onDone, onUpdate]);
}
