'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Video, Download, Play, Clock, Star, CheckCircle, AlertCircle, Eye } from 'lucide-react';
import { Clip } from '@shared/types';
import { toPct, toPctLabel, toHMMSS, toSec } from '@shared/format';
import { playPreview, stopPreview, isPlaying, getCurrentSrc } from '@shared/previewAudio';
import ClipDetail from './ClipDetail';

interface ClipGalleryProps {
  clips: Clip[];
}

export default function ClipGallery({ clips }: ClipGalleryProps) {
  const [selectedClip, setSelectedClip] = useState<Clip | null>(null);

  const getScoreColor = (score: number): string => {
    if (score >= 0.8) return 'text-green-400 bg-green-900/30';
    if (score >= 0.6) return 'text-blue-400 bg-blue-900/30';
    if (score >= 0.4) return 'text-yellow-400 bg-yellow-900/30';
    return 'text-gray-400 bg-gray-900/30';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'generating':
        return <Clock className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };



  return (
    <>
      <div className="rounded-2xl border border-[#1e2636] bg-white/[0.04] shadow-[0_10px_30px_rgba(0,0,0,0.35)] p-6 text-white">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center">
              <Video className="w-4 h-4 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-white">Clips</h3>
          </div>
          <span className="text-sm text-white/70">{clips.length} clips</span>
        </div>

      <div className="space-y-4">
        {clips.map((clip, index) => (
          <motion.div
            key={clip.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            className="relative rounded-xl overflow-hidden border border-[#1e2636] bg-white/[0.04] hover:border-white/35 transition cursor-pointer"
            onClick={() => { 
              console.log('[open]', clip.id); 
              setSelectedClip(clip); 
            }}
          >
            <div className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className={`text-[10px] font-semibold text-white px-1.5 py-0.5 rounded ${getScoreColor(clip.score)}`}>
                  {toPctLabel(Number(clip.score ?? 0))}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-white/80 capitalize">
                  {clip.status || "completed"}
                </span>
              </div>

              <div className="mt-2 text-sm font-semibold line-clamp-2 text-white">
                {clip.title || clip.text?.slice(0, 80) || "Untitled clip"}
              </div>

              {clip.text && (
                <div className="mt-1 text-xs text-white/60 line-clamp-2">
                  {clip.text}
                </div>
              )}

              <div className="mt-3 flex items-center justify-between text-xs text-white/50">
                <span>{toHMMSS(Number(clip.startTime ?? 0))} - {toHMMSS(Number(clip.endTime ?? 0))}</span>
                <span>{toSec(Number(clip.duration ?? 0))}s</span>
              </div>

              {clip.status === 'error' && clip.error && (
                <div className="mt-3 p-3 bg-red-900/30 border border-red-500/30 rounded-lg">
                  <p className="text-sm text-red-400">
                    <AlertCircle className="w-4 h-4 inline mr-2" />
                    {clip.error}
                  </p>
                </div>
              )}

              {clip.status === 'completed' && (
                <div className="mt-3 flex space-x-2">
                  <button
                    onClick={(e) => { 
                      e.stopPropagation(); 
                      console.log('[open-btn]', clip.id); 
                      setSelectedClip(clip); 
                    }}
                    className="px-3 py-1.5 text-xs rounded border border-[#2b3448] text-white hover:bg-white/5"
                    title="View detailed clip information"
                    aria-label="View detailed clip information"
                  >
                    <Eye className="w-3 h-3" />
                    <span>View Details</span>
                  </button>
                  {clip.previewUrl && (
                    <button
                      onClick={(e) => { 
                        e.stopPropagation(); 
                        playPreview(clip.previewUrl!); 
                      }}
                      className="px-3 py-1.5 text-xs rounded border border-[#2b3448] text-white hover:bg-white/5 flex items-center space-x-1"
                      title={`Preview clip starting at ${Math.round(clip.start_time || 0)}s`}
                      aria-label={`Preview clip starting at ${Math.round(clip.start_time || 0)}s`}
                    >
                      <Play className="w-3 h-3" />
                      <span>Preview</span>
                    </button>
                  )}
                  {clip.downloadUrl && (
                    <a
                      href={clip.downloadUrl}
                      download
                      onClick={(e) => e.stopPropagation()}
                      className="px-3 py-1.5 text-xs rounded border border-[#2b3448] text-white hover:bg-white/5 flex items-center space-x-1"
                      title="Download clip file"
                      aria-label="Download clip file"
                    >
                      <Download className="w-3 h-3" />
                      <span>Download</span>
                    </a>
                  )}
                </div>
              )}
            </div>

            {/* Clip Score Breakdown */}
            <div className="mt-4 pt-4 border-t border-[#1e2636]">
              <h5 className="text-sm font-medium text-white/80 mb-2">ClipScore Breakdown</h5>
              {clip.features ? (
                <div className="mt-3 grid grid-cols-2 gap-1 text-xs">
                  {[
                    ['Hook', clip.features.hook_score ?? clip.features.hook],
                    ['Emotion', clip.features.emotion_score ?? clip.features.emotion],
                    ['Prosody', clip.features.arousal_score ?? clip.features.prosody ?? clip.features.arousal],
                    ['Payoff', clip.features.payoff_score ?? clip.features.payoff],
                    ['Loop', clip.features.loopability ?? clip.features.loop],
                  ].map(([label, val]) =>
                    val == null ? null : (
                      <div key={label} className="flex items-center justify-between">
                        <span className="text-white/70">{label}</span>
                        <span className="font-medium text-white">{toPctLabel(Number(val ?? 0))}</span>
                      </div>
                    )
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                  <div className="text-center">
                    <div className="w-full bg-white/10 rounded-full h-2 mb-1">
                      <div 
                        className="bg-white/80 h-2 rounded-full" 
                        style={{ width: `${(clip.score * 0.25) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-white/70">Hook</span>
                  </div>
                  <div className="text-center">
                    <div className="w-full bg-white/10 rounded-full h-2 mb-1">
                      <div 
                        className="bg-white/80 h-2 rounded-full" 
                        style={{ width: `${(clip.score * 0.20) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-white/70">Emotion</span>
                  </div>
                  <div className="text-center">
                    <div className="w-full bg-white/10 rounded-full h-2 mb-1">
                      <div 
                        className="bg-white/80 h-2 rounded-full" 
                        style={{ width: `${(clip.score * 0.20) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-white/70">Prosody</span>
                  </div>
                  <div className="text-center">
                    <div className="w-full bg-white/10 rounded-full h-2 mb-1">
                      <div 
                        className="bg-white/80 h-2 rounded-full" 
                        style={{ width: `${(clip.score * 0.20) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-white/70">Payoff</span>
                  </div>
                  <div className="text-center">
                    <div className="w-full bg-white/10 rounded-full h-2 mb-1">
                      <div 
                        className="bg-white/80 h-2 rounded-full" 
                        style={{ width: `${(clip.score * 0.15) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-white/70">Loop</span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Download All Button */}
      {clips.some(clip => clip.status === 'completed' && clip.downloadUrl) && (
        <div className="mt-6 pt-6 border-t border-[#1e2636]">
          <div className="text-center">
            <p className="text-sm text-white/60 mb-2">
              {clips.filter(clip => clip.status === 'completed' && clip.downloadUrl).length} clips ready for download
            </p>
            <p className="text-xs text-white/50">
              Click individual download buttons above to get specific clips
            </p>
          </div>
        </div>
      )}
      </div>

      {/* Clip Detail Modal */}
      <ClipDetail 
        clip={selectedClip} 
        onClose={() => setSelectedClip(null)} 
      />
    </>
  );
}
