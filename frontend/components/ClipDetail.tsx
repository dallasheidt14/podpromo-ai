// components/ClipDetail.tsx
"use client";
import React, { useRef } from "react";
import Modal from "./Modal";
import { Clip } from "@shared/types";
import WordSyncedTranscript from "./WordSyncedTranscript";
import { PreviewPlayer } from "../src/components/PreviewPlayer";
import { fmtTimecode } from "../src/utils/timecode";
import { SHOW_TRANSCRIPT } from "../src/config/features";
import { buildScoreBundle } from "../app/lib/score";

function fallbackFromText(t?: string, max = 80) {
  const s = (t || '').replace(/\s+/g, ' ').trim();
  if (!s) return 'Untitled Clip';
  const firstSent = s.split(/(?<=[.!?])\s/)[0] || s.slice(0, max);
  return firstSent.length > max ? firstSent.slice(0, max).replace(/[ ,;:.!-]+$/,'') : firstSent;
}

type Props = {
  clip: Clip | null;
  open: boolean;
  onClose: () => void;
};

export default function ClipDetail({ clip, open, onClose }: Props) {
  const mediaRef = useRef<HTMLVideoElement & HTMLAudioElement>(null as any);
  
  if (!clip) return null;

  const ui = clip.uiScores ?? buildScoreBundle(clip);

  const duration =
    clip.startTime != null && clip.endTime != null
      ? Math.max(0, (clip.endTime - clip.startTime) | 0)
      : undefined;

  const isVideo = clip.previewUrl?.match(/\.(mp4|mov|webm|avi)$/i);
  const isAudio = clip.previewUrl?.match(/\.(mp3|m4a|aac|ogg|wav)$/i);

  return (
    <Modal open={open} onClose={onClose} maxWidthClass="max-w-4xl">
      <div className="space-y-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <h2 className="text-xl sm:text-2xl font-semibold text-neutral-900">
            {clip.title || fallbackFromText(clip.display_text || clip.text) || "Clip Details"}
          </h2>

          <div className="flex items-center gap-2">
            {ui.viralityPct > 0 && (
              <span className="inline-flex items-center gap-1 rounded-full bg-primary-50 px-3 py-1 text-sm font-medium text-primary-700">
                <svg width="16" height="16" viewBox="0 0 24 24">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.85 1.19 6.88L12 17.77l-6.19 3.23L7 14.12 2 9.27l6.91-1.01L12 2z" fill="currentColor"/>
                </svg>
                {ui.viralityPct}%
              </span>
            )}
            {duration != null && (
              <span className="inline-flex items-center gap-1 rounded-full bg-neutral-100 px-3 py-1 text-sm text-neutral-700">
                ⏱ {fmtTimecode(clip.startTime!)}–{fmtTimecode(clip.endTime!)} ({duration}s)
              </span>
            )}
          </div>
        </div>

        <div className="aspect-video w-full overflow-hidden rounded-xl border border-neutral-200">
          <PreviewPlayer 
            clip={clip} 
            className="h-full w-full"
            mediaRef={mediaRef}
          />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          {SHOW_TRANSCRIPT && (
            <div className="rounded-xl border border-neutral-200 p-4">
              <h3 className="mb-2 text-sm font-medium text-neutral-600">Transcript (word-synced)</h3>
              <WordSyncedTranscript clip={clip} mediaRef={mediaRef} />
              {clip.transcript_source && (
                <div className="mt-1 text-xs text-neutral-500">
                  transcript source: {clip.transcript_source}
                </div>
              )}
              {clip.raw_text && clip.raw_text !== clip.text && (
                <div className="mt-2">
                  <h4 className="text-xs font-medium text-neutral-500 mb-1">Raw Text (for audio matching)</h4>
                  <div className="text-xs text-neutral-600 bg-neutral-100 p-2 rounded">
                    {clip.raw_text}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="rounded-xl border border-neutral-200 p-4 space-y-3">
            <h3 className="text-sm font-medium text-neutral-600">Actions</h3>
            <div className="flex flex-wrap gap-2">
              {clip.downloadUrl && (
                <a
                  href={clip.downloadUrl}
                  className="btn btn-primary"
                  download
                >
                  Download
                </a>
              )}
              {clip.downloadUrl && (
                <a href={clip.downloadUrl} className="btn">
                  Download Captions
                </a>
              )}
              <button
                className="btn"
                onClick={() => {
                  navigator.clipboard.writeText(clip.title || fallbackFromText(clip.display_text || clip.text) || "Great clip!");
                }}
              >
                Copy Title
              </button>
            </div>
          </div>
        </div>
      </div>
    </Modal>
  );
}