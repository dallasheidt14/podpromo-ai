"use client";

import { useEffect, useRef, useState } from 'react';
import EpisodeUpload from '../components/EpisodeUpload';
import ClipGallery from '../components/ClipGallery';
import { normalizeClip } from '../src/shared/normalize';
import { Clip } from '../src/shared/types';
import { getClips, getProgress, handleApiResult } from '../src/shared/api';

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
    setShowResumePrompt(false);
  }

  function resumePreviousUpload() {
    setShowResumePrompt(false);
    // The EpisodeUpload component will handle the rest
  }

  function handleEpisodeUploaded(id: string) {
    setEpisodeId(id);
    setUploadStatus('processing');
  }

  function handleClipUpdate(clipId: string, updates: Partial<Clip>) {
    setClips(prevClips => 
      prevClips.map(clip => 
        clip.id === clipId ? { ...clip, ...updates } : clip
      )
    );
  }

  return (
    <div>
                {/* Header */}
                <div className="text-center mb-8">
                  <div className="flex items-center justify-center mb-4">
                    <img 
                      src="/logo.png" 
                      alt="Highlightly AI" 
                      className="h-96 w-auto"
                    />
                  </div>
                  <div className="bg-gradient-to-r from-blue-50 to-yellow-50 rounded-2xl p-6 border-2 border-blue-100 shadow-lg">
                    <h2 className="text-2xl font-bold text-gray-800 mb-3">
                      Find the clips that actually go viral
                    </h2>
                    <p className="text-lg text-gray-600 leading-relaxed">
                      We detect, score, and rank your best moments—then hand you social-ready cuts.
                    </p>
                  </div>
                </div>
      
      {/* Upload Section */}
      <div className="rounded-2xl border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-white shadow-card p-6 relative overflow-hidden">
        {/* Decorative accent */}
        <div className="absolute top-0 right-0 w-32 h-32 bg-yellow-200 rounded-full -translate-y-16 translate-x-16 opacity-20"></div>
        <EpisodeUpload
          onEpisodeUploaded={handleEpisodeUploaded}
          initialEpisodeId={episodeId ?? undefined}
          initialUploadStatus={uploadStatus}
          onClipsFetched={setClips}
          onCompleted={() => setUploadStatus('completed')}
        />
      </div>

      {/* Clips Section */}
      <div className="mt-6 rounded-2xl border-2 border-yellow-200 bg-gradient-to-br from-yellow-50 to-white shadow-card p-6 relative overflow-hidden">
        {/* Decorative accent */}
        <div className="absolute top-0 left-0 w-24 h-24 bg-blue-200 rounded-full -translate-y-12 -translate-x-12 opacity-20"></div>
        <h2 className="text-2xl font-bold mb-4 text-gray-800 relative z-10">Your AI-Generated Clips</h2>
        {clipsLoading && (
          <div className="text-center py-8">
            <div className="w-8 h-8 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-600">Analyzing your content...</p>
          </div>
        )}
        {!clipsLoading && clips.length > 0 && <ClipGallery clips={clips} onClipUpdate={handleClipUpdate} />}
        {!clipsLoading && clips.length === 0 && episodeId && (
          <div className="text-center py-8 text-gray-500">
            <p>No clips found yet. If processing just finished, give it a moment or refresh the page.</p>
          </div>
        )}
      </div>
    </div>
  );
}