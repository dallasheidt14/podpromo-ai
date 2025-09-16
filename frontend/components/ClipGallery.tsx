// components/ClipGallery.tsx
"use client";
import React, { useMemo, useState, useEffect, useCallback } from "react";
import Modal from "./Modal";
import { Clip } from "@shared/types";
import { getClipsSimple, apiUrl, type ClipSimple } from "../src/shared/api";
import { onClipsReady } from "../src/shared/events";

// Robust video preview component with error handling and remounting
function VideoPreview({ src, isVideo, isAudio, clipId }: { 
  src: string; 
  isVideo: boolean; 
  isAudio: boolean; 
  clipId: string; 
}) {
  const [error, setError] = useState(false);
  
  if (!src) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-2 bg-gray-400 rounded-full flex items-center justify-center">
            <span className="text-white text-2xl">📄</span>
          </div>
          <p className="text-sm text-gray-600 font-medium">No Preview</p>
        </div>
      </div>
    );
  }

  if (isVideo && !error) {
    return (
      <video
        key={src} // Force remount when src changes
        src={src}
        className="h-full w-full object-cover"
        data-clip-id={clipId}
        controls
        muted={false}
        playsInline
        preload="metadata"
        onError={() => setError(true)}
        style={{ background: "#000" }}
      />
    );
  }

  if (isAudio && !error) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-blue-100 to-purple-100">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-2 bg-blue-500 rounded-full flex items-center justify-center">
            <span className="text-white text-2xl">🎵</span>
          </div>
          <p className="text-sm text-gray-600 font-medium">Audio Preview</p>
        </div>
      </div>
    );
  }

  // Fallback to audio if video fails or if it's not clearly video/audio
  return (
    <audio
      key={src + ":audio"}
      src={src}
      controls
      className="h-full w-full"
      onError={() => setError(true)}
    />
  );
}

type Props = {
  clips?: Clip[]; // Make optional since we can fetch them ourselves
  emptyMessage?: string;
  onClipUpdate?: (clipId: string, updates: Partial<Clip>) => void;
  episodeId?: string;
  onClipsFetched?: (clips: Clip[]) => void; // Callback to notify parent of fetched clips
};

export default function ClipGallery({ clips, emptyMessage = "No clips yet.", onClipUpdate, episodeId, onClipsFetched }: Props) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Clip | null>(null);
  const [playingClipId, setPlayingClipId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [titleVariants, setTitleVariants] = useState<string[]>([]);
  const [currentTitle, setCurrentTitle] = useState<string>("");
  const [processedClips, setProcessedClips] = useState<Set<string>>(new Set());
  const [generatingClips, setGeneratingClips] = useState<Set<string>>(new Set());
  
  // Internal clips state for when we fetch them ourselves
  const [internalClips, setInternalClips] = useState<Clip[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  // Use clips from props if provided, otherwise use internal state
  const displayClips = clips || internalClips;

  // Normalize preview URL from various possible field names and ensure absolute
  const normalizeSrc = (clip: Clip | ClipSimple) => {
    const raw =
      (clip as any).previewUrl ||
      (clip as any).preview_url ||
      (clip as any).video_url ||
      (clip as any).audio_url || "";
    if (!raw) return "";
    if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
    return apiUrl(raw);
  };

  // Convert ClipSimple to Clip format
  const convertToClip = (simpleClip: ClipSimple): Clip => {
    return {
      id: simpleClip.id,
      startTime: simpleClip.start,
      endTime: simpleClip.end,
      start: simpleClip.start, // Alias for compatibility
      end: simpleClip.end,     // Alias for compatibility
      duration: simpleClip.end - simpleClip.start,
      score: 0, // Default score
      title: simpleClip.title || '',
      text: simpleClip.raw_text || simpleClip.full_transcript || '',
      raw_text: simpleClip.raw_text || '',
      full_transcript: simpleClip.full_transcript || '',
      status: 'completed' as const,
      features: {},
      downloadUrl: null,
      previewUrl: normalizeSrc(simpleClip),
      error: null,
    };
  };

  // Fetch clips function
  const fetchClips = useCallback(async (id: string) => {
    if (!id) return;
    
    setLoading(true);
    setError("");
    try {
      const simpleClips = await getClipsSimple(id);
      const convertedClips = simpleClips.map(convertToClip);
      setInternalClips(convertedClips);
      onClipsFetched?.(convertedClips);
    } catch (e: any) {
      setError(e?.message || "Failed to load clips");
    } finally {
      setLoading(false);
    }
  }, [onClipsFetched]);

  // Fetch clips when episodeId changes
  useEffect(() => {
    if (episodeId && !clips) {
      fetchClips(episodeId);
    }
  }, [episodeId, clips, fetchClips]);

  // Listen for clips-ready events
  useEffect(() => {
    if (!episodeId) return;
    
    return onClipsReady((completedEpisodeId) => {
      if (completedEpisodeId === episodeId) {
        fetchClips(completedEpisodeId);
      }
    });
  }, [episodeId, fetchClips]);

  // Reset UI state when episode changes to prevent title bleed
  useEffect(() => {
    setProcessedClips(new Set());
    setGeneratingClips(new Set());
    setTitleVariants([]);
    setCurrentTitle("");
    setSelected(null);
    setPlayingClipId(null);
  }, [episodeId]);

  useEffect(() => {
    if (displayClips && displayClips.length > 0) {
      // Auto-generate titles for clips that need them (only once per clip)
      displayClips.forEach((clip, index) => {
        if (processedClips.has(clip.id) || generatingClips.has(clip.id)) return; // Skip if already processed or generating
        
        const transcript = clip.text;
        const isTranscriptSnippet = clip.title && clip.title.length > 50 && transcript && clip.title === transcript.substring(0, clip.title.length);

        if (index === 0) {
          generateNewTitle(clip.id);
          setProcessedClips(prev => new Set([...Array.from(prev), clip.id]));
        } else if ((!clip.title || isTranscriptSnippet) && transcript) {
          generateNewTitle(clip.id);
          setProcessedClips(prev => new Set([...Array.from(prev), clip.id]));
        }
      });
    }
  }, [clips, processedClips, generatingClips]);

  // Initialize current title when selected clip changes
  useEffect(() => {
    if (selected) {
      setCurrentTitle(selected.title || "");
      setTitleVariants([]);
      
      // Auto-backfill title if missing or if it's just a transcript snippet
      const transcript = selected.text;
      const isTranscriptSnippet = selected.title && selected.title.length > 50 && transcript && selected.title === transcript.substring(0, selected.title.length);

      if ((!selected.title || isTranscriptSnippet) && transcript && !processedClips.has(selected.id) && !generatingClips.has(selected.id)) {
        generateNewTitle(selected.id);
        setProcessedClips(prev => new Set([...Array.from(prev), selected.id]));
      }
    }
  }, [selected]);

  const sorted = useMemo(
    () =>
      [...(clips || [])].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)),
    [clips]
  );

  const handlePlayPause = (clip: Clip) => {
    const selector = `video[data-clip-id="${clip.id}"]`;
    const thisVideo = document.querySelector(selector) as HTMLVideoElement | null;
    const allVideos = Array.from(document.querySelectorAll('video[data-clip-id]')) as HTMLVideoElement[];
    // Pause all others
    allVideos.forEach(v => { if (v !== thisVideo) { v.pause(); v.currentTime = 0; } });
    if (!thisVideo) return;
    if (!thisVideo.paused) {
      thisVideo.pause();
      setPlayingClipId(null);
      return;
    }
    thisVideo.muted = false;
    thisVideo.play().then(() => setPlayingClipId(clip.id)).catch(() => {});
  };

  const generateNewTitle = async (clipId: string) => {
    if (process.env.NODE_ENV === 'development') {
      console.log('Generating title for clip:', clipId);
    }
    setGeneratingClips(prev => new Set([...Array.from(prev), clipId]));
    setGenerating(true);
    try {
      const response = await fetch(`http://localhost:8000/api/clips/${clipId}/titles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          platform: 'shorts',
          n: 6,
          allow_emoji: true
        }),
        cache: 'no-store',
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Title generation failed:', response.status, errorText);
        
        // Retry once for 422 errors (clip not ready)
        if (response.status === 422) {
          console.log('Retrying title generation after 422...');
          await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
          
          const retryResponse = await fetch(`http://localhost:8000/api/clips/${clipId}/titles`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              platform: 'shorts',
              n: 6,
              allow_emoji: true
            }),
            cache: 'no-store',
          });
          
          if (!retryResponse.ok) {
            const retryErrorText = await retryResponse.text();
            console.error('Title generation retry failed:', retryResponse.status, retryErrorText);
            throw new Error(`Title generation failed after retry: ${retryResponse.status}`);
          }
          
          // Use retry response
          const retryData = await retryResponse.json();
          setClipTitles(prev => ({
            ...prev,
            [clipId]: retryData.variants
          }));
          setGeneratingClips(prev => {
            const newSet = new Set(prev);
            newSet.delete(clipId);
            return newSet;
          });
          setGenerating(false);
          return;
        }
        
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      if (process.env.NODE_ENV === 'development') {
        console.log('Generated title chosen:', data.chosen);
      }

      setCurrentTitle(data.chosen);
      setTitleVariants(data.variants);

      // Update the selected clip's title in the parent state
      if (selected) {
        selected.title = data.chosen;
      }

      // Notify parent component to update the clip
      if (onClipUpdate) {
        onClipUpdate(clipId, { title: data.chosen });
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('[titles] generate error', error);
      }
      // Graceful fallback: keep the old title
    } finally {
      setGenerating(false);
      setGeneratingClips(prev => {
        const newSet = new Set(prev);
        newSet.delete(clipId);
        return newSet;
      });
    }
  };

  const copyTitle = (title: string) => {
    navigator.clipboard.writeText(title);
  };

  const updateTitle = async (clipId: string, newTitle: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/clips/${clipId}/title`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ platform: 'shorts', title: newTitle })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      setCurrentTitle(newTitle);
      
      // Update the selected clip's title in the parent state
      if (selected) {
        selected.title = newTitle;
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('[titles] update error', error);
      }
    }
  };

  if (!clips?.length) {
    return (
      <div className="rounded-2xl border border-dashed border-neutral-300 p-8 text-center text-neutral-500">
        {emptyMessage}
      </div>
    );
  }

  return (
    <>
      {loading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading clips...</p>
        </div>
      )}
      {error && (
        <div className="text-center py-8">
          <div className="text-red-500 mb-2">⚠️ Error loading clips</div>
          <p className="text-gray-600">{error}</p>
        </div>
      )}
      {!loading && !error && displayClips.length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-600">{emptyMessage}</p>
        </div>
      )}
      {!loading && !error && displayClips.length > 0 && (
        <div className="grid gap-4 sm:gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {displayClips.map((clip) => {
          const src = normalizeSrc(clip);
          const isVideo = src && src.match(/\.(mp4|mov|webm|avi)$/i);
          const isAudio = src && src.match(/\.(mp3|m4a|aac|ogg|wav)$/i);
          return (
            <div
              key={clip.id}
              className="group rounded-2xl border-2 border-blue-100 bg-gradient-to-br from-white to-blue-50 shadow-card hover:shadow-card-hover transition-all duration-300 hover:border-blue-300 relative overflow-hidden"
            >
              {/* Decorative accent */}
              <div className="absolute top-0 right-0 w-16 h-16 bg-yellow-200 rounded-full -translate-y-8 translate-x-8 opacity-30 group-hover:opacity-50 transition-opacity"></div>
              <div className="aspect-video w-full overflow-hidden rounded-t-2xl bg-neutral-100">
                <VideoPreview 
                  src={normalizeSrc(clip)}
                  isVideo={isVideo}
                  isAudio={isAudio}
                  clipId={clip.id}
                />
              </div>

              <div className="p-4 space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="line-clamp-2 font-medium text-neutral-900">
                    {clip.title || clip.text?.slice(0, 60) + "..." || "Untitled clip"}
                  </h3>
                  {typeof clip.score === "number" && (
                    <span className="shrink-0 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 px-3 py-1.5 text-xs font-bold text-white shadow-lg">
                      ⭐ {Math.round(clip.score * 100)}/100
                    </span>
                  )}
                </div>

                {/* Timing Information */}
                {clip.startTime != null && clip.endTime != null && (
                  <div className="text-xs text-blue-800 bg-gradient-to-r from-blue-100 to-yellow-100 px-3 py-2 rounded-lg border border-blue-200 relative z-10">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">⏱️ Duration: {Math.max(0, Math.round(clip.endTime - clip.startTime))}s ({Math.round(clip.startTime)}s-{Math.round(clip.endTime)}s)</span>
                      {clip.previewDuration && clip.previewDuration < (clip.endTime - clip.startTime) && (
                        <span className="ml-2 px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full border border-orange-200">
                          trimmed to {Math.round(clip.previewDuration)}s
                        </span>
                      )}
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap gap-2 relative z-10">
                  <button
                    className="btn btn-primary bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-300"
                    onClick={() => {
                      setSelected(clip);
                      setOpen(true);
                    }}
                  >
                    View details
                  </button>

                  <button
                    className="btn bg-gradient-to-r from-yellow-400 to-yellow-500 hover:from-yellow-500 hover:to-yellow-600 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-300"
                    onClick={() => handlePlayPause(clip)}
                  >
                    {playingClipId === clip.id ? '⏸️ Pause Audio' : '🎵 Preview Audio'}
                  </button>

                  {/* Removed legacy external "Preview" link */}

                  {clip.downloadUrl && (
                    <a className="btn" href={clip.downloadUrl} download>
                      Download
                    </a>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        </div>
      )}

      {/* Removed legacy global <audio>; we control the clip's <video> directly */}

      <Modal open={open} onClose={() => setOpen(false)}>
        <div>
          <h4 className="text-xl font-semibold mb-4">{currentTitle || selected?.title || "Clip Details"}</h4>
          {selected?.score && (
            <div className="mb-6">
              <div className="text-4xl font-bold text-gray-900">
                {Math.round((selected.score * 100))}
                <span className="text-2xl text-gray-500">/100</span>
              </div>
            </div>
          )}

          {/* Timing Information */}
          {selected?.startTime != null && selected?.endTime != null && (
            <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center gap-2">
                <svg width="20" height="20" viewBox="0 0 24 24" className="text-blue-600">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
                </svg>
                <span className="text-lg font-semibold text-blue-900">
                  Duration: {Math.max(0, Math.round(selected.endTime - selected.startTime))}s ({Math.round(selected.startTime)}s-{Math.round(selected.endTime)}s)
                </span>
              </div>
            </div>
          )}

          {/* Score Breakdown Section */}
          {selected && (
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <h5 className="text-sm font-medium text-gray-700 mb-3">Score Breakdown</h5>
              <div className="grid grid-cols-4 gap-2">
                {[
                  { key: 'hook_score', name: 'Hook', weight: 0.35, description: 'Attention-grabbing opening that hooks viewers in the first 3 seconds' },
                  { key: 'arousal_score', name: 'Arousal', weight: 0.20, description: 'Energy and excitement level that keeps viewers engaged' },
                  { key: 'emotion_score', name: 'Emotion', weight: 0.15, description: 'Emotional engagement that creates connection and relatability' },
                  { key: 'question_score', name: 'Q/List', weight: 0.10, description: 'Questions or list format that encourages interaction and completion' },
                  { key: 'payoff_score', name: 'Payoff', weight: 0.10, description: 'Clear value or insight delivered that makes viewers feel they learned something' },
                  { key: 'info_density', name: 'Info', weight: 0.05, description: 'Information density - how much valuable content is packed in' },
                  { key: 'loopability', name: 'Loop', weight: 0.05, description: 'Replayability factor - how likely viewers are to watch again' },
                  { key: 'platform_len_match', name: 'Length', weight: 0.05, description: 'Optimal length for the target platform (Shorts/Reels/TikTok)' }
                ].map(({ key, name, weight, description }) => {
                  // Get score from features object (normalized in normalize.ts)
                  const score = selected.features?.[key] || 0;
                  const percentage = Math.round(score * 100);
                  
                  return (
                    <div
                      key={key}
                      className="relative group cursor-help"
                      title={description}
                    >
                      <div className="bg-white rounded-lg p-2 border border-gray-200 hover:border-gray-300 transition-colors">
                        <div className="text-xs font-medium text-gray-600 mb-1">{name}</div>
                        <div className="text-lg font-bold text-gray-900">{percentage}%</div>
                      </div>
                      
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-black text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 max-w-xs text-center">
                        {description}
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-black"></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          
          {selected?.text && (
            <div className="mb-4">
              <h5 className="font-medium mb-2">Transcript:</h5>
              <p className="text-sm text-gray-600">{selected.text}</p>
            </div>
          )}
          {selected && normalizeSrc(selected) && (
            <div className="mb-4">
              <h5 className="font-medium mb-2">Preview:</h5>
              <audio 
                src={normalizeSrc(selected)} 
                controls
                className="w-full" 
              />
            </div>
          )}
        </div>
      </Modal>
    </>
  );
}