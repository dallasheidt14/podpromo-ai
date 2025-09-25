// ui/ScorePill.tsx
import { getViralityPct, scoreToTone, toneClasses } from './score';
import type { Clip } from '../types/Clip';

export const ScorePill: React.FC<{ clip: Clip; className?: string }> = ({ clip, className }) => {
  const pct = getViralityPct(clip);
  const tone = scoreToTone(pct);
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${toneClasses(tone)} ${className ?? ''}`}>
      <span>Virality</span>
      <span className="tabular-nums">{pct}</span>
    </span>
  );
};
