"use client";

import { useEffect, useRef, useState } from 'react';
import EpisodeUpload from '../components/EpisodeUpload';
import ClipGallery from '../components/ClipGallery';
import { normalizeClip } from '@shared/normalize';
import { Clip } from '@shared/types';
import { getClips, getProgress, handleApiResult } from '@shared/api';

export default function Page() {
  const [episodeId, setEpisodeId] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle'|'uploading'|'processing'|'completed'|'error'>('idle');
  const [clips, setClips] = useState<Clip[]>([]);
  const [clipsLoading, setClipsLoading] = useState(false);
  const didResumeRef = useRef(false);
  const [showResumePrompt, setShowResumePrompt] = useState(false);

  function startNewUpload() {
    try { localStorage.removeItem('lastEpisodeId'); } catch {}
    setEpisodeId(null);
    setUploadStatus('idle');
    setClips([]);
    setClipsLoading(false);
    // Clear ?episodeId
    try {
      const url = new URL(window.location.href);
      url.searchParams.delete('episodeId');
      window.history.replaceState({}, '', url.toString());
    } catch {}
    setShowResumePrompt(false);
  }

  function resumeLast() {
    try {
      const saved = localStorage.getItem('lastEpisodeId');
      if (saved) {
        setEpisodeId(saved);
        setUploadStatus('processing'); // EpisodeUpload will resume polling
        const url = new URL(window.location.href);
        url.searchParams.set('episodeId', saved);
        window.history.replaceState({}, '', url.toString());
      }
    } catch {}
    setShowResumePrompt(false);
  }

  async function fetchClips(id: string) {
    try {
      setClipsLoading(true);
      const result = await getClips(id);
      
      handleApiResult(
        result,
        (data) => {
          const raw = Array.isArray(data.clips) ? data.clips : [];
          const normalized = raw.map(normalizeClip);
          const isAd = (c: any) => Boolean(c?.is_advertisement || c?._ad_flag || c?.features?.is_advertisement);
          const ranked = normalized.filter(c => !isAd(c)).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 12);
          setClips(ranked);
          setUploadStatus('completed');
          localStorage.setItem('lastEpisodeId', id);
          console.log(`[resume] clips loaded: ${ranked.length} (raw ${normalized.length})`);
        },
        (error) => {
          console.warn("[resume] fetchClips failed:", error);
        }
      );
    } catch (e) {
      console.warn("[resume] unexpected error:", e);
    } finally {
      setClipsLoading(false);
    }
  }

  // 🔁 Resume flow on direct URL / refresh
  useEffect(() => {
    if (didResumeRef.current) return;
    const urlId = new URLSearchParams(window.location.search).get("episodeId");
    const saved = localStorage.getItem("lastEpisodeId");
    
    if (urlId) {
      didResumeRef.current = true;
      setEpisodeId(urlId);

      (async () => {
        try {
          console.log("[resume] checking progress for", urlId);
          const result = await getProgress(urlId);
          
          handleApiResult(
            result,
            (data) => {
              const stage = String(data.progress?.stage || data.status || "").toLowerCase();
              console.log("[resume] stage:", stage);
              if (stage === "completed") {
                fetchClips(urlId);
              } else {
                setUploadStatus('processing'); // tell EpisodeUpload to resume polling
              }
            },
            (error) => {
              console.warn("[resume] progress check failed:", error);
              setUploadStatus('processing'); // safe default: start polling
            }
          );
        } catch (e) {
          console.warn("[resume] failed, defaulting to processing", e);
          setUploadStatus('processing');
        }
      })();
    } else if (saved) {
      // Don't auto-resume; let the user choose.
      setShowResumePrompt(true);
    }
  }, []);

  // 🚀 Reset state when no episodeId (fresh page load)
  useEffect(() => {
    const urlId = new URLSearchParams(window.location.search).get("episodeId");
    const saved = localStorage.getItem("lastEpisodeId");
    const id = urlId || saved;
    
    if (!id) {
      // Fresh page load - reset everything
      setEpisodeId(null);
      setUploadStatus('idle');
      setClips([]);
      setClipsLoading(false);
    }
  }, []);

  function handleEpisodeUploaded(id: string) {
    setEpisodeId(id);
    setUploadStatus('processing'); // EpisodeUpload will poll; we'll load clips on completion
  }

  return (
    <main className="min-h-screen p-6 max-w-5xl mx-auto">
      {showResumePrompt && (
        <div className="rounded-xl border border-[#1e2636] bg-white/[0.04] p-4 mb-6 flex items-center justify-between gap-3">
          <div className="text-sm text-white/80">
            We found a previous session. Would you like to resume it, or start a new upload?
          </div>
          <div className="flex gap-2">
            <button 
              className="px-4 py-2 text-sm bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
              onClick={startNewUpload}
            >
              Start new upload
            </button>
            <button 
              className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              onClick={resumeLast}
            >
              Resume
            </button>
          </div>
        </div>
      )}

      <div className="mb-6">
        <h1 className="text-3xl font-extrabold tracking-tight text-white">
          PodPromo <span className="text-emerald-400">AI</span>
        </h1>
        <p className="mt-1 text-white/70">
          Upload your podcast episode to generate <span className="text-white">viral clips</span>
        </p>
      </div>

      <EpisodeUpload
        onEpisodeUploaded={handleEpisodeUploaded}
        initialEpisodeId={episodeId ?? undefined}
        initialUploadStatus={uploadStatus}
        onCompleted={() => episodeId && fetchClips(episodeId)}
      />

      {/* Session Controls */}
      {episodeId && (
        <div className="rounded-xl border border-[#1e2636] bg-white/[0.04] p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold text-white">Current Session</h2>
            <div className="flex gap-2">
              <button 
                className="px-4 py-2 text-sm bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
                onClick={startNewUpload}
              >
                Start another upload
              </button>
              <button 
                className="px-4 py-2 text-sm bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg transition-colors"
                onClick={() => {
                  try { localStorage.removeItem('lastEpisodeId'); } catch {}
                  setEpisodeId(null);
                  setUploadStatus('idle');
                  setClips([]);
                  setClipsLoading(false);
                  try {
                    const url = new URL(window.location.href);
                    url.searchParams.delete('episodeId');
                    window.history.replaceState({}, '', url.toString());
                  } catch {}
                }}
              >
                Clear session
              </button>
            </div>
          </div>
          <p className="text-sm text-white/60">Episode ID: {episodeId.slice(0, 8)}...</p>
        </div>
      )}

      {/* Top Clips */}
      <section id="clips" className="mt-8">
        <h2 className="text-2xl font-semibold text-white mb-3">Top Clips</h2>
        {clipsLoading && (
          <div className="rounded-xl border border-[#1e2636] bg-white/[0.04] p-4 text-center text-white/70">
            Loading clips…
          </div>
        )}
        {!clipsLoading && clips.length > 0 && <ClipGallery clips={clips} />}

        {!clipsLoading && clips.length === 0 && episodeId && (
          <div className="rounded-xl border border-[#1e2636] bg-white/[0.04] p-4 text-center text-white/70">
            No clips found for this episode yet. If processing just finished, give it a second or refresh.
          </div>
        )}
      </section>
    </main>
  );
}