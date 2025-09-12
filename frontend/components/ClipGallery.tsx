// components/ClipGallery.tsx
"use client";
import React, { useMemo, useState, useEffect } from "react";
import Modal from "./Modal";
import { Clip } from "@shared/types";

type Props = {
  clips: Clip[];
  emptyMessage?: string;
  onClipUpdate?: (clipId: string, updates: Partial<Clip>) => void;
  episodeId?: string;
};

export default function ClipGallery({ clips, emptyMessage = "No clips yet.", onClipUpdate, episodeId }: Props) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Clip | null>(null);
  const [playingClipId, setPlayingClipId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [titleVariants, setTitleVariants] = useState<string[]>([]);
  const [currentTitle, setCurrentTitle] = useState<string>("");
  const [processedClips, setProcessedClips] = useState<Set<string>>(new Set());
  const [generatingClips, setGeneratingClips] = useState<Set<string>>(new Set());

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
    if (clips && clips.length > 0) {
      // Auto-generate titles for clips that need them (only once per clip)
      clips.forEach((clip, index) => {
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
    if (playingClipId === clip.id) {
      // Stop current audio
      const audio = document.querySelector(`audio[data-clip-id="${clip.id}"]`) as HTMLAudioElement;
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }
      setPlayingClipId(null);
    } else {
      // Stop any currently playing audio
      const allAudio = document.querySelectorAll('audio');
      allAudio.forEach(audio => {
        (audio as HTMLAudioElement).pause();
        (audio as HTMLAudioElement).currentTime = 0;
      });
      // Start new audio
      setPlayingClipId(clip.id);
      // Trigger play after state update
      setTimeout(() => {
        const newAudio = document.querySelector(`audio[data-clip-id="${clip.id}"]`) as HTMLAudioElement;
        if (newAudio) {
          if (process.env.NODE_ENV === 'development') {
            console.log('Attempting to play audio:', newAudio.src);
          }
          newAudio.play().catch(e => {
            if (process.env.NODE_ENV === 'development') {
              console.log('Audio play failed:', e);
            }
          });
        } else {
          if (process.env.NODE_ENV === 'development') {
            console.log('Audio element not found');
          }
        }
      }, 100);
    }
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
        if (process.env.NODE_ENV === 'development') {
          console.error('Title generation failed:', response.status, errorText);
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
      <div className="grid gap-4 sm:gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {sorted.map((clip) => {
          const isVideo = clip.previewUrl?.match(/\.(mp4|mov|webm|avi)$/i);
          const isAudio = clip.previewUrl?.match(/\.(mp3|m4a|aac|ogg|wav)$/i);
          return (
            <div
              key={clip.id}
              className="group rounded-2xl border-2 border-blue-100 bg-gradient-to-br from-white to-blue-50 shadow-card hover:shadow-card-hover transition-all duration-300 hover:border-blue-300 relative overflow-hidden"
            >
              {/* Decorative accent */}
              <div className="absolute top-0 right-0 w-16 h-16 bg-yellow-200 rounded-full -translate-y-8 translate-x-8 opacity-30 group-hover:opacity-50 transition-opacity"></div>
              <div className="aspect-video w-full overflow-hidden rounded-t-2xl bg-neutral-100">
                {isVideo ? (
                  <video
                    src={clip.previewUrl}
                    className="h-full w-full object-cover"
                    muted
                    playsInline
                    preload="metadata"
                  />
                ) : isAudio ? (
                  <div 
                    className="flex h-full w-full items-center justify-center bg-gradient-to-br from-blue-50 to-purple-50 relative cursor-pointer hover:from-blue-100 hover:to-purple-100 transition-colors"
                    onClick={() => handlePlayPause(clip)}
                  >
                    {/* Waveform Visualization */}
                    <div className="absolute inset-0 flex items-center justify-center px-4">
                      <div className="flex items-end space-x-1 h-16">
                        {Array.from({ length: 20 }, (_, i) => {
                          const height = Math.random() * 0.8 + 0.2; // Random height between 20% and 100%
                          const isActive = playingClipId === clip.id;
                          return (
                            <div
                              key={i}
                              className={`w-1 rounded-full transition-all duration-150 ${
                                isActive 
                                  ? 'bg-blue-500 animate-pulse' 
                                  : 'bg-blue-300'
                              }`}
                              style={{ height: `${height * 100}%` }}
                            />
                          );
                        })}
                      </div>
                    </div>
                    
                    {/* Play/Pause Button Overlay */}
                    <div className="relative z-10">
                      <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-lg hover:shadow-xl transition-shadow">
                        {playingClipId === clip.id ? (
                          <svg width="24" height="24" viewBox="0 0 24 24" className="text-blue-600">
                            <rect x="6" y="4" width="4" height="16" fill="currentColor"/>
                            <rect x="14" y="4" width="4" height="16" fill="currentColor"/>
                          </svg>
                        ) : (
                          <svg width="24" height="24" viewBox="0 0 24 24" className="text-blue-600">
                            <path d="M8 5v14l11-7z" fill="currentColor"/>
                          </svg>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex h-full w-full items-center justify-center text-neutral-400">
                    No preview
                  </div>
                )}
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

                  {clip.previewUrl && isVideo && (
                    <a className="btn" href={clip.previewUrl} target="_blank" rel="noreferrer">
                      Preview
                    </a>
                  )}

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

      {/* Hidden audio element for playback */}
      {playingClipId && (() => {
        const clip = clips.find(c => c.id === playingClipId);
        const audioUrl = clip?.previewUrl ? 
          (clip.previewUrl.startsWith('http') ? clip.previewUrl : `http://localhost:8000${clip.previewUrl}`) : 
          null;
        return audioUrl ? (
          <audio
            key={playingClipId}
            data-clip-id={playingClipId}
            src={audioUrl}
            autoPlay
            onEnded={() => setPlayingClipId(null)}
            onError={(e) => {
              if (process.env.NODE_ENV === 'development') {
                console.error('Audio playback error:', e);
              }
              setPlayingClipId(null);
            }}
          />
        ) : null;
      })()}

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
          {selected?.previewUrl && (
            <div className="mb-4">
              <h5 className="font-medium mb-2">Preview:</h5>
              <audio 
                src={selected.previewUrl.startsWith('http') ? selected.previewUrl : `http://localhost:8000${selected.previewUrl}`} 
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