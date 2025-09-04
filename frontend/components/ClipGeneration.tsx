'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Video, Settings, Play, Sparkles, Clock, Target } from 'lucide-react';
import { Episode } from '@shared/types';
import { AnimatePresence } from 'framer-motion';

interface ClipGenerationProps {
  episode: Episode;
  onGenerateClips: () => void;
  isGenerating: boolean;
}

export default function ClipGeneration({ episode, onGenerateClips, isGenerating }: ClipGenerationProps) {
  const [clipCount, setClipCount] = useState(3);
  const [minDuration, setMinDuration] = useState(12);
  const [maxDuration, setMaxDuration] = useState(30);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleGenerate = async () => {
    try {
      const response = await fetch(`/api/episodes/${episode.id}/clips?regenerate=true`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to start clip generation');
      }

      const result = await response.json();
      console.log('Clip generation started:', result);
      
      // Trigger parent callback
      onGenerateClips();
      
    } catch (error) {
      console.error('Error starting clip generation:', error);
      // In a real app, you'd show an error message to the user
    }
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-secondary-100 rounded-lg flex items-center justify-center">
            <Video className="w-5 h-5 text-secondary-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Generate Clips</h3>
            <p className="text-sm text-gray-600">
              Create engaging social media clips from your episode
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <Clock className="w-4 h-4" />
          <span>~5 minutes</span>
        </div>
      </div>

      {/* Basic Settings */}
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Clips
            </label>
            <select
              value={clipCount}
              onChange={(e) => setClipCount(Number(e.target.value))}
              className="input-field"
              disabled={isGenerating}
            >
              <option value={3}>3 clips</option>
              <option value={4}>4 clips</option>
              <option value={5}>5 clips</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Min Duration
            </label>
            <select
              value={minDuration}
              onChange={(e) => setMinDuration(Number(e.target.value))}
              className="input-field"
              disabled={isGenerating}
            >
              <option value={8}>8 seconds</option>
              <option value={10}>10 seconds</option>
              <option value={12}>12 seconds</option>
              <option value={15}>15 seconds</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Duration
            </label>
            <select
              value={maxDuration}
              onChange={(e) => setMaxDuration(Number(e.target.value))}
              className="input-field"
              disabled={isGenerating}
            >
              <option value={20}>20 seconds</option>
              <option value={25}>25 seconds</option>
              <option value={30}>30 seconds</option>
              <option value={35}>35 seconds</option>
            </select>
          </div>
        </div>

        {/* Advanced Settings Toggle */}
        <div className="border-t border-gray-200 pt-4">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            disabled={isGenerating}
          >
            <Settings className="w-4 h-4" />
            <span>Advanced Settings</span>
            <motion.div
              animate={{ rotate: showAdvanced ? 90 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <Play className="w-3 h-3" />
            </motion.div>
          </button>
        </div>

        {/* Advanced Settings */}
        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Caption Style
                  </label>
                  <select className="input-field" defaultValue="modern">
                    <option value="modern">Modern</option>
                    <option value="classic">Classic</option>
                    <option value="minimal">Minimal</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Video Quality
                  </label>
                  <select className="input-field" defaultValue="1080p">
                    <option value="720p">720p</option>
                    <option value="1080p">1080p</option>
                  </select>
                </div>
              </div>
              
              <div className="space-y-3">
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    disabled={isGenerating}
                  />
                  <span className="text-sm text-gray-700">Enable punch-ins and zoom effects</span>
                </label>
                
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    disabled={isGenerating}
                  />
                  <span className="text-sm text-gray-700">Normalize audio levels</span>
                </label>
                
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    disabled={isGenerating}
                  />
                  <span className="text-sm text-gray-700">Create loopable endings</span>
                </label>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Generation Button */}
        <div className="pt-4">
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full btn-primary py-3 text-lg font-semibold flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Generating Clips...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                <span>Generate {clipCount} Clips</span>
              </>
            )}
          </button>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-900">AI-Powered Selection</span>
            </div>
            <p className="text-xs text-blue-700">
              Our ClipScore algorithm finds the most engaging moments using hook cues, 
              emotion detection, and audio analysis.
            </p>
          </div>
          
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Video className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium text-green-900">Vertical Format</span>
            </div>
            <p className="text-xs text-green-700">
              Optimized 1080x1920 MP4 clips perfect for Instagram, TikTok, and YouTube Shorts.
            </p>
          </div>
        </div>

        {/* Estimated Output */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Estimated Output</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Total Duration:</span>
              <span className="ml-2 font-medium text-gray-900">
                {formatDuration(clipCount * ((minDuration + maxDuration) / 2))}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Format:</span>
              <span className="ml-2 font-medium text-gray-900">1080x1920 MP4</span>
            </div>
            <div>
              <span className="text-gray-600">Processing Time:</span>
              <span className="ml-2 font-medium text-gray-900">3-5 minutes</span>
            </div>
            <div>
              <span className="text-gray-600">File Size:</span>
              <span className="ml-2 font-medium text-gray-900">~15-30 MB</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
