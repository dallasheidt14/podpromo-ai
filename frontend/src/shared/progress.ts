export function isTerminalProgress(p: { stage?: string; percent?: number } | null | undefined) {
  if (!p) return false;
  return p.stage === "completed" || p.stage === "error" || (p.stage === "scoring" && (p.percent ?? 0) >= 100);
}
