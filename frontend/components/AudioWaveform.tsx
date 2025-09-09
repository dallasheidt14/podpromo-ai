"use client";
import React, { useMemo } from "react";

interface Props {
  isActive: boolean;
  bars?: number;
}

/**
 * Renders a simple audio waveform visualization.
 * Heights are generated once on mount to avoid re-renders
 * using new random values.
 */
const AudioWaveform: React.FC<Props> = ({ isActive, bars = 20 }) => {
  // Generate bar heights once
  const heights = useMemo(
    () => Array.from({ length: bars }, () => Math.random() * 0.8 + 0.2),
    [bars]
  );

  return (
    <div className="absolute inset-0 flex items-center justify-center px-4">
      <div className="flex items-end space-x-1 h-16">
        {heights.map((height, i) => (
          <div
            key={i}
            className={`w-1 rounded-full ${
              isActive ? "bg-blue-500 equalizer-bar" : "bg-blue-300"
            }`}
            style={{
              height: `${height * 100}%`,
              animationDelay: `${i * 0.1}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default AudioWaveform;