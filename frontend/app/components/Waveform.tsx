"use client";

interface WaveformProps {
  src: string;
  start: number;
  end: number;
  onChange: (range: { start: number; end: number }) => void;
}

export default function Waveform({ src, start, end, onChange }: WaveformProps) {
  const handleRangeChange = (newStart: number, newEnd: number) => {
    onChange({ start: newStart, end: newEnd });
  };

  const jumpToSelection = () => {
    const audio = document.querySelector('audio') as HTMLAudioElement;
    if (audio) {
      audio.currentTime = start;
      audio.play();
    }
  };

  return (
    <div style={{border: "1px solid #e5e7eb", padding: 16, borderRadius: 8}}>
      <div style={{marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center"}}>
        <div>
          <strong>Selected Range:</strong> {start.toFixed(1)}s → {end.toFixed(1)}s 
          <span style={{color: "#666", marginLeft: 8}}>(Duration: {(end - start).toFixed(1)}s)</span>
        </div>
        <button 
          onClick={jumpToSelection}
          style={{
            padding: "6px 12px",
            backgroundColor: "#17a2b8",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: "14px"
          }}
        >
          🎯 Jump to Selection
        </button>
      </div>
      
      <div style={{display: "flex", gap: 8, alignItems: "center"}}>
        <input
          type="range"
          min="0"
          max="300"
          step="0.1"
          value={start}
          onChange={(e) => handleRangeChange(parseFloat(e.target.value), end)}
          style={{flex: 1}}
        />
        <span style={{minWidth: "60px"}}>Start: {start.toFixed(1)}s</span>
      </div>
      <div style={{display: "flex", gap: 8, alignItems: "center", marginTop: 8}}>
        <input
          type="range"
          min="0"
          max="300"
          step="0.1"
          value={end}
          onChange={(e) => handleRangeChange(start, parseFloat(e.target.value))}
          style={{flex: 1}}
        />
        <span style={{minWidth: "60px"}}>End: {end.toFixed(1)}s</span>
      </div>
      
      <div style={{marginTop: 16}}>
        <audio controls src={src} style={{width: "100%"}} />
      </div>
    </div>
  );
}
