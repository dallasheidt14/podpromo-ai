import { useEffect, useRef } from "react";
import { getProgressSimple, type ProgressResponse } from "../shared/api";
import { notifyClipsReady } from "../shared/events";
import { isTerminalProgress } from "../shared/progress";

type Options = {
  onUpdate?: (p: ProgressResponse) => void;
  onDone?: () => void;
  intervalMs?: number;
};

export function useProgressPoller(episodeId?: string, opts: Options = {}) {
  const { onUpdate, onDone, intervalMs = 4000 } = opts;
  const timer = useRef<number | null>(null);
  const stopRef = useRef(false);

  useEffect(() => {
    if (!episodeId) return;
    stopRef.current = false;

    const isTerminal = (p: ProgressResponse) => {
      return isTerminalProgress(p) || p.stage === "error";
    };

    const tick = async () => {
      try {
        const p = await getProgressSimple(episodeId);
        onUpdate?.(p);
        if (isTerminal(p)) {
          notifyClipsReady(episodeId);
          onDone?.();
          return;
        }
      } catch {
        // ignore transient errors
      }
      if (!stopRef.current) {
        timer.current = window.setTimeout(tick, intervalMs) as unknown as number;
      }
    };

    timer.current = window.setTimeout(tick, 1000) as unknown as number;
    return () => {
      stopRef.current = true;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [episodeId, intervalMs, onDone, onUpdate]);
}
