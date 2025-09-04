"use client";

import { Dialog } from "@headlessui/react";
import { useEffect, useState } from "react";
import type { Clip } from "@shared/types";
import { toPct, toPctLabel, toHMMSS } from "@shared/format";

export default function ClipDetail({ clip, onClose }: { clip: Clip | null; onClose: () => void }) {
  const [open, setOpen] = useState(!!clip);
  
  useEffect(() => {
    console.log('[Detail] clip changed:', clip?.id);
    setOpen(!!clip);
  }, [clip]);
  
  if (!clip) return null;

  const bars: Array<[string, number | undefined, string | undefined]> = [
    ["Hook", clip.features?.hook_score, clip.grades?.hook],
    ["Flow", clip.features?.arousal_score, clip.grades?.flow],
    ["Value", clip.features?.payoff_score, clip.grades?.value],
    ["Trend", clip.features?.loopability,  clip.grades?.trend],
  ];

  return (
    <Dialog open={open} onClose={onClose} className="fixed inset-0 z-50">
      <div className="fixed inset-0 bg-black/70" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="card max-w-4xl w-full overflow-hidden">
          {/* Header */}
          <div className="p-5 border-b border-[#1e2636] flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">{clip.title}</h2>
            <div className="text-sm text-white/90">
              <span className="font-semibold">{toPctLabel(Number(clip.score ?? 0))}</span>
              <span className="ml-2 text-white/60">{clip.grades?.overall || ""}</span>
            </div>
          </div>

          {/* Body */}
          <div className="grid md:grid-cols-2 gap-0">
            {/* Media */}
            <div className="bg-black">
              {clip.previewUrl ? (
                (() => {
                  const src = clip.previewUrl;
                  const isAudio = !!src && /\.(mp3|m4a|aac|ogg|wav)$/i.test(src);
                  
                  return isAudio ? (
                    <audio src={src} controls className="w-full" />
                  ) : (
                    <video src={src} controls className="w-full max-h-[460px] object-cover" />
                  );
                })()
              ) : (
                <div className="h-full w-full grid place-items-center text-white/70 p-8">
                  No preview available
                </div>
              )}
            </div>

            {/* Transcript + Scores */}
            <div className="p-5 space-y-4">
              <div className="text-xs text-white/50">
                [{toHMMSS(Number(clip.startTime ?? 0))} – {toHMMSS(Number(clip.endTime ?? 0))}]
              </div>
              <div className="rounded card-2 p-3 text-sm leading-6 whitespace-pre-line text-white/90">
                {clip.text || "(no transcript snippet)"}
              </div>

              <div>
                <div className="text-sm font-semibold text-white/80 mb-2">Score breakdown</div>
                <div className="space-y-2">
                  {[
                    ["Hook", clip.features?.hook_score, clip.grades?.hook],
                    ["Flow", clip.features?.arousal_score, clip.grades?.flow],
                    ["Value", clip.features?.payoff_score, clip.grades?.value],
                    ["Trend", clip.features?.loopability,  clip.grades?.trend],
                  ].map(([label, v, g]) => (
                    <div key={label as string} className="flex items-center gap-2">
                      <div className="w-16 text-xs text-white/60">{label as string}</div>
                      <div className="flex-1 h-2 rounded bg-white/10">
                        <div className="h-2 rounded bg-white/80" style={{ width: `${Math.max(0, Math.min(1, Number(v ?? 0))) * 100}%` }} />
                      </div>
                      <div className="w-16 text-right text-xs text-white/80">
                        {v != null ? toPct(Number(v)) : "–"} {g ? `(${g})` : ""}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-[#1e2636] flex items-center justify-end gap-2">
            <button
              onClick={() => navigator.clipboard.writeText(clip.title)}
              className="px-3 py-1.5 text-sm rounded border border-[#2b3448] text-white hover:bg-white/5"
            >
              Copy title
            </button>
            {clip.downloadUrl && (
              <a
                href={clip.downloadUrl}
                download
                className="px-3 py-1.5 text-sm rounded border border-[#2b3448] text-white hover:bg-white/5"
              >
                Download
              </a>
            )}
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-sm rounded bg-white text-black hover:bg-white/90"
            >
              Close
            </button>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}
