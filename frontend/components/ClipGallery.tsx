// components/ClipGallery.tsx
"use client";
import React, { useMemo, useState, useEffect, useCallback } from "react";
import Modal from "./Modal";
import { Clip } from "@shared/types";
import { getClipsSimple, apiUrl, type ClipSimple } from "../src/shared/api";
import { onClipsReady } from "../src/shared/events";
import { PreviewPlayer } from "../src/components/PreviewPlayer";
import { fmtTimecode } from "../src/utils/timecode";

// Utility functions for virality and platform fit
function getViralityPct(clip: Clip): number {
  // Check for new virality fields first
  if (typeof clip.features?.virality_pct === 'number') return clip.features.virality_pct;
  if (typeof clip.features?.virality_calibrated === 'number') return Math.round(clip.features.virality_calibrated * 100);
  
  // Fall back to legacy score conversion
  if (typeof clip.score === 'number') {
    const x = Math.max(0, Math.min(100, clip.score * 100));
    return Math.round(x); // Already in 0-100 range
  }
  
  return 0;
}

function getPlatformFitPct(clip: Clip): number {
  if (typeof clip.features?.platform_fit_pct === 'number') return clip.features.platform_fit_pct;
  if (typeof clip.features?.platform_fit === 'number') return Math.round(clip.features.platform_fit * 100);
  return 0;
}

function getViralityTone(pct: number): 'ok' | 'good' | 'great' | 'fire' {
  if (pct >= 85) return 'fire';
  if (pct >= 70) return 'great';
  if (pct >= 55) return 'good';
  return 'ok';
}

function getViralityToneClasses(clip: Clip): string {
  const pct = getViralityPct(clip);
  const tone = getViralityTone(pct);
  
  switch (tone) {
    case 'fire':  return 'bg-red-600/10 text-red-600 ring-1 ring-red-600/20';
    case 'great': return 'bg-emerald-600/10 text-emerald-600 ring-1 ring-emerald-600/20';
    case 'good':  return 'bg-amber-600/10 text-amber-700 ring-1 ring-amber-600/20';
    default:      return 'bg-slate-600/10 text-slate-600 ring-1 ring-slate-600/20';
  }
}

function MiniBar({ label, pct, title }: { label: string; pct: number; title?: string }) {
  const safe = Math.max(0, Math.min(100, Math.round(pct)));
  return (
    <div className="space-y-1" title={title}>
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span className="tabular-nums">{safe}%</span>
      </div>
      <div className="h-1.5 w-full rounded bg-slate-200">
        <div className="h-1.5 rounded bg-slate-800" style={{ width: `${safe}%` }} />
      </div>
    </div>
  );
}

function fallbackFromText(t?: string, max = 80) {
  const s = (t || '').replace(/\s+/g, ' ').trim();
  if (!s) return 'Untitled Clip';
  const firstSent = s.split(/(?<=[.!?])\s/)[0] || s.slice(0, max);
  return firstSent.length > max ? firstSent.slice(0, max).replace(/[ ,;:.!-]+$/,'') : firstSent;
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


  // Convert ClipSimple to Clip format
  const convertToClip = (simpleClip: ClipSimple): Clip => {
    return {
      id: simpleClip.id,
      startTime: simpleClip.start,
      endTime: simpleClip.end,
      start: simpleClip.start, // Alias for compatibility
      end: simpleClip.end,     // Alias for compatibility
      duration: simpleClip.end - simpleClip.start,
      score: simpleClip.score || 0, // Use actual score if available
      title: simpleClip.title || '',
      text: simpleClip.raw_text || simpleClip.full_transcript || '',
      raw_text: simpleClip.raw_text || '',
      full_transcript: simpleClip.full_transcript || '',
      status: 'completed' as const,
      features: {
        // Map new virality and platform fit fields
        virality_calibrated: simpleClip.virality_calibrated,
        virality_pct: simpleClip.virality_pct,
        platform_fit: simpleClip.platform_fit,
        platform_fit_pct: simpleClip.platform_fit_pct,
        // Map other existing fields
        display_score: simpleClip.display_score,
        clip_score_100: simpleClip.clip_score_100,
        confidence: simpleClip.confidence,
        confidence_color: simpleClip.confidence_color,
        // Map scoring features
        hook_score: simpleClip.hook_score,
        arousal_score: simpleClip.arousal_score,
        emotion_score: simpleClip.emotion_score,
        payoff_score: simpleClip.payoff_score,
        question_score: simpleClip.question_score,
        loopability: simpleClip.loopability,
        info_density: simpleClip.info_density,
        platform_len_match: simpleClip.platform_len_match,
      },
      downloadUrl: null,
      previewUrl: null, // Will be handled by PreviewPlayer
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
  }, [displayClips, processedClips, generatingClips, generateNewTitle]);

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
  }, [selected, generateNewTitle, processedClips, generatingClips]);

  const sorted = useMemo(
    () =>
      [...(clips || [])].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)),
    [clips]
  );


  const generateNewTitle = useCallback(async (clipId: string) => {
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
  }, [selected, onClipUpdate]);

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
          return (
            <div
              key={clip.id}
              className="group rounded-2xl border-2 border-blue-100 bg-gradient-to-br from-white to-blue-50 shadow-card hover:shadow-card-hover transition-all duration-300 hover:border-blue-300 relative overflow-hidden"
            >
              {/* Decorative accent */}
              <div className="absolute top-0 right-0 w-16 h-16 bg-yellow-200 rounded-full -translate-y-8 translate-x-8 opacity-30 group-hover:opacity-50 transition-opacity"></div>
              <div className="aspect-video w-full overflow-hidden rounded-t-2xl bg-neutral-100">
                <PreviewPlayer 
                  clip={clip}
                  className="h-full w-full"
                />
              </div>

              <div className="p-4 space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="line-clamp-2 font-medium text-neutral-900">
                    {clip.title || fallbackFromText(clip.display_text || clip.text) || "Untitled clip"}
                  </h3>
                  <div className="flex items-center gap-2">
                    {/* New virality pill */}
                    {(() => {
                      const virality = getViralityPct(clip);
                      return virality > 0 && (
                        <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${getViralityToneClasses(clip)}`}>
                          <span>Virality</span>
                          <span className="tabular-nums">{virality}</span>
                        </span>
                      );
                    })()}
                    {/* Legacy score fallback */}
                    {(() => {
                      const virality = getViralityPct(clip);
                      return typeof clip.score === "number" && virality === 0 && (
                        <span className="shrink-0 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 px-3 py-1.5 text-xs font-bold text-white shadow-lg">
                          ⭐ {Math.round(clip.score * 100)}/100
                        </span>
                      );
                    })()}
                  </div>
                </div>

                {/* Timing Information */}
                {clip.startTime != null && clip.endTime != null && (
                  <div className="text-xs text-blue-800 bg-gradient-to-r from-blue-100 to-yellow-100 px-3 py-2 rounded-lg border border-blue-200 relative z-10">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">⏱️ {clip.display_range || `${fmtTimecode(clip.startTime)}–${fmtTimecode(clip.endTime)}`} ({Math.max(0, Math.round(clip.endTime - clip.startTime))}s)</span>
                      {clip.previewDuration && clip.previewDuration < (clip.endTime - clip.startTime) && (
                        <span className="ml-2 px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full border border-orange-200">
                          trimmed to {Math.round(clip.previewDuration)}s
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* New virality and platform fit bars */}
                <div className="space-y-2">
                  {(() => {
                    const virality = getViralityPct(clip);
                    const platformFit = getPlatformFitPct(clip);
                    return (
                      <>
                        <MiniBar label="Virality" pct={virality} />
                        {platformFit > 0 && (
                          <MiniBar 
                            label="Platform fit" 
                            pct={platformFit} 
                            title="Length-fit for Shorts/TikTok/Reels" 
                          />
                        )}
                      </>
                    );
                  })()}
                </div>

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
                  {selected.display_range || `${fmtTimecode(selected.startTime)}–${fmtTimecode(selected.endTime)}`} ({Math.max(0, Math.round(selected.endTime - selected.startTime))}s)
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
          {selected && (
            <div className="mb-4">
              <h5 className="font-medium mb-2">Preview:</h5>
              <PreviewPlayer 
                clip={selected}
                className="w-full"
              />
            </div>
          )}
        </div>
      </Modal>
    </>
  );
}