"use client";

import { useEffect, useRef, useState } from 'react';
import EpisodeUpload from '../components/EpisodeUpload';
import ClipGallery from '../components/ClipGallery';
import { normalizeClip } from '@shared/normalize';
import { Clip } from '@shared/types';

export default function Page() {
  const [episodeId, setEpisodeId] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle'|'uploading'|'processing'|'completed'|'error'>('idle');
  const [clips, setClips] = useState<Clip[]>([]);
  const [clipsLoading, setClipsLoading] = useState(false);
  const didResumeRef = useRef(false);

  async function fetchClips(id: string) {
    try {
      setClipsLoading(true);
      const { fetchJson } = await import('../../shared/http');
      const j = await fetchJson(`/api/episodes/${id}/clips`);
      const raw = Array.isArray(j?.clips) ? j.clips : [];
      const normalized = raw.map(normalizeClip);
      const isAd = (c: any) => Boolean(c?.is_advertisement || c?._ad_flag || c?.features?.is_advertisement);
      const ranked = normalized.filter(c => !isAd(c)).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 12);
      setClips(ranked);
      setUploadStatus('completed');
      localStorage.setItem('lastEpisodeId', id);
      console.log(`[resume] clips loaded: ${ranked.length} (raw ${normalized.length}) source: ${j._source}`);
    } catch (e) {
      console.warn("[resume] fetchClips failed", e);
    } finally {
      setClipsLoading(false);
    }
  }

  // 🔁 Resume flow on direct URL / refresh
  useEffect(() => {
    if (didResumeRef.current) return;
    const urlId = new URLSearchParams(window.location.search).get("episodeId");
    const saved = localStorage.getItem("lastEpisodeId");
    const id = urlId || saved;
    if (!id) return;

    didResumeRef.current = true;
    setEpisodeId(id);

    (async () => {
      try {
        console.log("[resume] checking progress for", id);
        const r = await fetch(`/api/progress/${id}`);
        if (!r.ok) {
          console.warn("[resume] progress not ok", r.status);
          setUploadStatus('processing'); // safe default: start polling
          return;
        }
        const j = await r.json();
        const stage = String(j?.progress?.stage || j?.status || "").toLowerCase();
        console.log("[resume] stage:", stage);
        if (stage === "completed") {
          await fetchClips(id);
        } else {
          setUploadStatus('processing'); // tell EpisodeUpload to resume polling
        }
      } catch (e) {
        console.warn("[resume] failed, defaulting to processing", e);
        setUploadStatus('processing');
      }
    })();
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