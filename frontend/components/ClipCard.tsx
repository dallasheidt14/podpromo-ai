// components/ClipCard.tsx
import { ScorePill } from '../ui/ScorePill';
import { MiniBar } from '../ui/MiniBar';
import { getPlatformFitPct } from '../ui/score';
import { buildScoreBundle } from '../app/lib/score';
import type { Clip } from '../types/Clip';
import { useEffect } from 'react';

export const ClipCard: React.FC<{ clip: Clip; onClick?: () => void }> = ({ clip, onClick }) => {
  const ui = clip.uiScores ?? buildScoreBundle(clip);
  const platform = getPlatformFitPct(clip);

  // Debug logs - remove after testing
  useEffect(() => {
    console.log('score fields', {
      display_score: clip.display_score,
      final_score: clip.final_score,
      virality: clip.virality,
      uiScores: clip.uiScores ?? buildScoreBundle(clip),
    });
  }, [clip]);

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
        <MiniBar label="Virality" pct={ui.viralityPct} />
        <MiniBar label="Platform fit" pct={platform} title="Length-fit for Shorts/TikTok/Reels" />
      </div>

      {/* Additional score breakdown */}
      <div className="grid grid-cols-3 gap-2 mt-2 text-xs">
        <div className="text-center">
          <div className="text-gray-500">Hook</div>
          <div className="font-semibold">{ui.hookPct}%</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Arousal</div>
          <div className="font-semibold">{ui.arousalPct}%</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Payoff</div>
          <div className="font-semibold">{ui.payoffPct}%</div>
        </div>
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
