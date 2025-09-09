"use client";

import { useState, useEffect } from 'react';
import { Episode } from '../src/shared/types';

interface EpisodeSelectorProps {
  onEpisodeSelected: (episodeId: string) => void;
}

export default function EpisodeSelector({ onEpisodeSelected }: EpisodeSelectorProps) {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [loading, setLoading] = useState(false);

  const loadEpisodes = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/episodes');
      if (response.ok) {
        const data = await response.json();

        // Handle the correct response format: {ok: true, episodes: [...]} 
        if (data.ok && Array.isArray(data.episodes)) {
          // Convert backend format to frontend format
          const convertedEpisodes = data.episodes.map((ep: any) => ({
            id: ep.episode_id,
            filename: ep.title || `Episode ${ep.episode_id}`,
            originalName: ep.title || `Episode ${ep.episode_id}`,
            size: 0,
            duration: ep.duration_s || 0,
            status: ep.status,
            uploadedAt: ep.created_at || new Date().toISOString()
          }));
          setEpisodes(convertedEpisodes);
        } else {
          if (process.env.NODE_ENV === 'development') {
            console.warn('[EpisodeSelector] Unexpected response format:', data);
          }
          setEpisodes([]);
        }
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Failed to load episodes:', error);
      }
      setEpisodes([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEpisodes();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Loading episodes...</p>
        </div>
      </div>
    );
  }

  if (episodes.length === 0) {
    return (
      <div className="card">
        <div className="text-center py-4">
          <p className="text-sm text-gray-600">No episodes found. Upload a new episode to get started.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Test Different Episodes</h3>
        <button
          onClick={loadEpisodes}
          className="text-sm text-blue-600 hover:text-blue-700"
        >
          Refresh
        </button>
      </div>
      
      <div className="space-y-2">
        {episodes.map((episode) => (
          <div
            key={episode.id}
            className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
          >
            <div className="flex-1">
              <h4 className="font-medium text-gray-900 truncate">
                {episode.filename || episode.originalName || `Episode ${episode.id}`}
              </h4>
              <div className="flex items-center space-x-4 text-sm text-gray-600">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  episode.status === 'completed' ? 'bg-green-100 text-green-700' :
                  episode.status === 'processing' ? 'bg-blue-100 text-blue-700' :
                  episode.status === 'failed' ? 'bg-red-100 text-red-700' :
                  'bg-gray-100 text-gray-700'
                }`}>
                  {episode.status}
                </span>
                <span>ID: {episode.id.slice(0, 8)}...</span>
              </div>
            </div>
            <button
              onClick={() => onEpisodeSelected(episode.id)}
              className="ml-4 px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              View Clips
            </button>
          </div>
        ))}
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          💡 Tip: You can also use URL parameters like <code>?episodeId=your-episode-id</code>
        </p>
      </div>
    </div>
  );
}
