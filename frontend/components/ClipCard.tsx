// components/ClipCard.tsx
import { ScorePill } from '../ui/ScorePill';
import { MiniBar } from '../ui/MiniBar';
import { getViralityPct, getPlatformFitPct } from '../ui/score';
import type { Clip } from '../types/Clip';

export const ClipCard: React.FC<{ clip: Clip; onClick?: () => void }> = ({ clip, onClick }) => {
  const virality = getViralityPct(clip);
  const platform = getPlatformFitPct(clip);

  return (
    <div className="group rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow-md transition p-3">
      {/* preview (existing) */}
      <div className="aspect-video w-full overflow-hidden rounded-xl bg-black/5 mb-3">
        {/* your existing <video> / <img> preview goes here */}
      </div>

      {/* header row */}
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="line-clamp-1 text-sm font-semibold text-slate-900">
          {clip.title || 'Untitled clip'}
        </h3>
        <div className="flex items-center gap-2">
          <ScorePill clip={clip} />
          {clip.duration != null && (
            <span className="text-[11px] text-slate-500 tabular-nums">
              {formatDuration(clip.duration)}
            </span>
          )}
        </div>
      </div>

      {/* bars */}
      <div className="space-y-2">
        <MiniBar label="Virality" pct={virality} />
        <MiniBar label="Platform fit" pct={platform} title="Length-fit for Shorts/TikTok/Reels" />
      </div>

      {/* actions (existing) */}
      <div className="mt-3 flex items-center justify-end gap-2">
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
