// components/ClipCard.tsx
import { ScorePill } from '../ui/ScorePill';
import { MiniBar } from '../ui/MiniBar';
import { getPlatformFitPct } from '../ui/score';
import { buildScoreBundle } from '../app/lib/score';
import { ContextBadges } from './ContextBadges';
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

      {/* context badges */}
      <div className="mb-2">
        <ContextBadges clip={clip} />
      </div>

      {/* bars */}
      <div className="space-y-2">
        <MiniBar label="Virality" pct={ui.viralityPct} />
        <MiniBar label="Platform fit" pct={platform} title="Length-fit for Shorts/TikTok/Reels" />
      </div>

      {/* Platform Recommendations */}
      {clip.platform_recommendations && clip.platform_recommendations.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-500 mb-1">Will Perform Best On:</div>
          <div className="flex flex-wrap gap-1">
            {clip.platform_recommendations
              .filter(rec => rec.fit_score >= 0.7) // Only show high-fit platforms
              .slice(0, 3) // Limit to top 3
              .map((rec, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                  title={rec.reason}
                >
                  {rec.platform === 'shorts' ? 'YouTube Shorts' : 
                   rec.platform === 'tiktok' ? 'TikTok' :
                   rec.platform === 'reels' ? 'Instagram Reels' :
                   rec.platform.charAt(0).toUpperCase() + rec.platform.slice(1)}
                </span>
              ))}
          </div>
        </div>
      )}

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

      {/* debug section for development */}
      {process.env.NODE_ENV !== 'production' && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-slate-500">Debug</summary>
          <pre className="text-xs text-slate-600 mt-1 whitespace-pre-wrap">
            {JSON.stringify({
              seed_idx: clip.seed_idx,
              payoff_type: clip.features?.payoff_type,
              payoff_score: clip.features?.payoff_score,
              start: clip.startTime,
              end: clip.endTime,
              duration: clip.duration
            }, null, 2)}
          </pre>
        </details>
      )}
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
