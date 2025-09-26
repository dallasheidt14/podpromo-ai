// components/ClipRow.tsx
import { ScorePill } from '../ui/ScorePill';
import { MiniBar } from '../ui/MiniBar';
import { getPlatformFitPct } from '../ui/score';
import { buildScoreBundle } from '../app/lib/score';
import type { Clip } from '../types/Clip';

export const ClipRow: React.FC<{ clip: Clip; onClick?: () => void }> = ({ clip, onClick }) => {
  const ui = clip.uiScores ?? buildScoreBundle(clip);
  const platform = getPlatformFitPct(clip);

  return (
    <div className="group flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-3 hover:shadow-sm transition">
      {/* preview thumbnail */}
      <div className="aspect-video w-16 overflow-hidden rounded bg-black/5 flex-shrink-0">
        {/* your existing <video> / <img> preview goes here */}
      </div>

      {/* content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-sm font-semibold text-slate-900 truncate">
            {clip.title || 'Untitled clip'}
          </h3>
          <ScorePill clip={clip} />
        </div>
        
        {/* single virality bar */}
        <div className="w-32">
          <MiniBar label="Virality" pct={ui.viralityPct} title={`Platform fit: ${platform}%`} />
        </div>
      </div>

      {/* duration */}
      {clip.duration != null && (
        <div className="text-xs text-slate-500 tabular-nums flex-shrink-0">
          {formatDuration(clip.duration)}
        </div>
      )}

      {/* actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {/* your existing buttons */}
      </div>
    </div>
  );
};

function formatDuration(sec?: number) {
  if (!sec && sec !== 0) return '–';
  const s = Math.round(sec);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, '0')}`;
}
