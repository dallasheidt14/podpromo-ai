'use client';

import { motion } from 'framer-motion';
import { Video, Download, Play, Clock, Star, CheckCircle, AlertCircle } from 'lucide-react';
import { Clip } from '../../shared/types';

interface ClipGalleryProps {
  clips: Clip[];
}

export default function ClipGallery({ clips }: ClipGalleryProps) {
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getScoreColor = (score: number): string => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-blue-600 bg-blue-100';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-gray-600 bg-gray-100';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'generating':
        return <Clock className="w-4 h-4 text-blue-600 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  const handleDownload = async (clip: Clip) => {
    if (clip.status !== 'completed' || !clip.download_url) {
      return;
    }

    try {
      const response = await fetch(`/api/clips/${clip.id}/download`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `clip_${clip.id}.mp4`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const handlePreview = (clip: Clip) => {
    if (clip.status !== 'completed' || !clip.download_url) {
      return;
    }
    
    // In a real app, you'd open a video preview modal
    window.open(clip.download_url, '_blank');
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-secondary-100 rounded-lg flex items-center justify-center">
            <Video className="w-4 h-4 text-secondary-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900">Generated Clips</h3>
        </div>
        <span className="text-sm text-gray-500">{clips.length} clips</span>
      </div>

      <div className="space-y-4">
        {clips.map((clip, index) => (
          <motion.div
            key={clip.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <Video className="w-5 h-5 text-gray-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">{clip.title}</h4>
                    <p className="text-sm text-gray-600">{clip.description}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Duration:</span>
                    <span className="ml-2 font-medium">{formatDuration(clip.duration)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Start Time:</span>
                    <span className="ml-2 font-medium">{formatTime(clip.start_time)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">End Time:</span>
                    <span className="ml-2 font-medium">{formatTime(clip.end_time)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Score:</span>
                    <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(clip.score)}`}>
                      {(clip.score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {clip.status === 'failed' && clip.error && (
                  <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-red-700">
                      <AlertCircle className="w-4 h-4 inline mr-2" />
                      {clip.error}
                    </p>
                  </div>
                )}
              </div>

              <div className="flex flex-col items-end space-y-2 ml-4">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(clip.status)}
                  <span className="text-xs text-gray-500 capitalize">{clip.status}</span>
                </div>

                {clip.status === 'completed' && (
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handlePreview(clip)}
                      className="btn-outline py-1 px-3 text-sm flex items-center space-x-1"
                    >
                      <Play className="w-3 h-3" />
                      <span>Preview</span>
                    </button>
                    <button
                      onClick={() => handleDownload(clip)}
                      className="btn-primary py-1 px-3 text-sm flex items-center space-x-1"
                    >
                      <Download className="w-3 h-3" />
                      <span>Download</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Clip Score Breakdown */}
            <div className="mt-4 pt-4 border-t border-gray-100">
              <h5 className="text-sm font-medium text-gray-700 mb-2">ClipScore Breakdown</h5>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                <div className="text-center">
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${(clip.score * 0.25) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-600">Hook</span>
                </div>
                <div className="text-center">
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                    <div 
                      className="bg-purple-600 h-2 rounded-full" 
                      style={{ width: `${(clip.score * 0.20) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-600">Emotion</span>
                </div>
                <div className="text-center">
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${(clip.score * 0.20) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-600">Prosody</span>
                </div>
                <div className="text-center">
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                    <div 
                      className="bg-orange-600 h-2 rounded-full" 
                      style={{ width: `${(clip.score * 0.20) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-600">Payoff</span>
                </div>
                <div className="text-center">
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                    <div 
                      className="bg-pink-600 h-2 rounded-full" 
                      style={{ width: `${(clip.score * 0.15) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-600">Loop</span>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Download All Button */}
      {clips.some(clip => clip.status === 'completed') && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          <button
            onClick={() => {
              clips.forEach(clip => {
                if (clip.status === 'completed') {
                  handleDownload(clip);
                }
              });
            }}
            className="w-full btn-secondary py-3 flex items-center justify-center space-x-2"
          >
            <Download className="w-5 h-5" />
            <span>Download All Completed Clips</span>
          </button>
        </div>
      )}
    </div>
  );
}
