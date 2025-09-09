// components/ClipDetail.tsx
"use client";
import React from "react";
import Modal from "./Modal";
import { Clip } from "@shared/types";

type Props = {
  clip: Clip | null;
  open: boolean;
  onClose: () => void;
};

export default function ClipDetail({ clip, open, onClose }: Props) {
  if (!clip) return null;

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
            {clip.title || "Clip Details"}
          </h2>

          <div className="flex items-center gap-2">
            {typeof clip.score === "number" && (
              <span className="inline-flex items-center gap-1 rounded-full bg-primary-50 px-3 py-1 text-sm font-medium text-primary-700">
                <svg width="16" height="16" viewBox="0 0 24 24">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.85 1.19 6.88L12 17.77l-6.19 3.23L7 14.12 2 9.27l6.91-1.01L12 2z" fill="currentColor"/>
                </svg>
                {(clip.score * 100).toFixed(0)}%
              </span>
            )}
            {duration != null && (
              <span className="inline-flex items-center gap-1 rounded-full bg-neutral-100 px-3 py-1 text-sm text-neutral-700">
                ⏱ Duration: {duration}s ({Math.round(clip.startTime!)}s-{Math.round(clip.endTime!)}s)
              </span>
            )}
          </div>
        </div>

        <div className="aspect-video w-full overflow-hidden rounded-xl border border-neutral-200">
          {isVideo ? (
            <video src={clip.previewUrl} controls className="h-full w-full" />
          ) : isAudio ? (
            <div className="flex h-full w-full items-center justify-center bg-neutral-50">
              <audio src={clip.previewUrl} controls className="w-full max-w-2xl p-4" />
            </div>
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-neutral-50 text-neutral-500">
              No preview available
            </div>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border border-neutral-200 p-4">
            <h3 className="mb-2 text-sm font-medium text-neutral-600">Transcript</h3>
            <div className="max-h-44 overflow-auto whitespace-pre-wrap text-sm leading-relaxed text-neutral-800">
              {clip.text || "No transcript available."}
            </div>
          </div>

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
                  navigator.clipboard.writeText(clip.title || "Great clip!");
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