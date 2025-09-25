// examples/ClipGalleryExample.tsx
import { useState, useEffect } from 'react';
import { ClipCard } from '../components/ClipCard';
import { ClipCardWithBadge } from '../components/ClipCardWithBadge';
import { ClipRow } from '../components/ClipRow';
import { RunSummary } from '../components/RunSummary';
import { mapClip } from '../data/mapClip';
import type { Clip } from '../types/Clip';

interface RunData {
  clips: any[];
  summary: {
    seeds: number;
    cand: number;
    kept: number;
    finals: number[];
    eos: number;
    dens: number;
    mode: 'strict' | 'balanced';
    avgVirality?: number;
    avgPlatformFit?: number;
    platformProtect?: boolean;
    plV2Weight?: number;
  };
}

export const ClipGalleryExample: React.FC = () => {
  const [clips, setClips] = useState<Clip[]>([]);
  const [runData, setRunData] = useState<RunData | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  useEffect(() => {
    // Example: fetch from your API
    fetchClips().then(data => {
      const mappedClips = data.clips.map(mapClip);
      setClips(mappedClips);
      setRunData(data.summary);
    });
  }, []);

  const fetchClips = async (): Promise<RunData> => {
    // Replace with your actual API call
    const response = await fetch('/api/clips');
    return response.json();
  };

  if (!runData) return <div>Loading...</div>;

  return (
    <div className="p-6">
      {/* Run Summary */}
      <RunSummary {...runData} />

      {/* View Toggle */}
      <div className="mb-4 flex gap-2">
        <button
          onClick={() => setViewMode('grid')}
          className={`px-3 py-1 rounded text-sm ${
            viewMode === 'grid' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
          }`}
        >
          Grid
        </button>
        <button
          onClick={() => setViewMode('list')}
          className={`px-3 py-1 rounded text-sm ${
            viewMode === 'list' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
          }`}
        >
          List
        </button>
      </div>

      {/* Clips */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {clips.map((clip, index) => (
            <ClipCardWithBadge
              key={clip.id}
              clip={clip}
              clips={clips}
              onClick={() => console.log('Selected clip:', clip.id)}
            />
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {clips.map((clip, index) => (
            <ClipRow
              key={clip.id}
              clip={clip}
              onClick={() => console.log('Selected clip:', clip.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
};
