/**
 * Format seconds as timecode (M:SS or H:MM:SS)
 */
export function fmtTimecode(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  
  return h > 0 
    ? `${h}:${m.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`
    : `${m}:${ss.toString().padStart(2, '0')}`;
}
