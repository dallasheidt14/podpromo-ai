// Formatting utilities for consistent display across components

export const toPct = (score0to1: number) =>
  `${Math.round(Math.max(0, Math.min(1, score0to1)) * 100)}%`;

export const toPctLabel = (x: number) => `${Math.round((x ?? 0) * 100)}/100`;

export const toHMMSS = (s: number) => {
  s = Math.max(0, Math.floor(Number(s ?? 0)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h > 0 ? `${h}:${String(m).padStart(2,"0")}:${String(ss).padStart(2,"0")}` : `${m}:${String(ss).padStart(2,"0")}`;
};

export const toSec = (sec: number, digits = 1) =>
  `${(Math.round(sec * Math.pow(10, digits)) / Math.pow(10, digits)).toFixed(digits)}s`;

export const formatDuration = (seconds: number): string => {
  if (seconds < 60) return toSec(seconds);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${toSec(remainingSeconds)}`;
};

export const formatTime = (seconds: number): string => {
  return toHMMSS(seconds);
};
