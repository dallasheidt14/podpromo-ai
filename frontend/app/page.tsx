"use client";

import * as React from "react";
import { useState } from "react";
import Controls from "./components/Controls";
import Waveform from "./components/Waveform";
import { API_URL } from "./lib/config";

type Candidate = { start: number; end: number; text?: string; score: number; features?: any };

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const [cands, setCands] = useState<Candidate[]>([]);
  const [mediaURL, setMediaURL] = useState("");
  const [range, setRange] = useState({ start: 0, end: 15 });
  const [selected, setSelected] = useState<number>(0);
  const [outputs, setOutputs] = useState<string[]>([]);
  const [pollCount, setPollCount] = useState(0);
  const [progress, setProgress] = useState(0);

  const upload = async () => {
    if (!file) return;
    setStatus("Uploading…");
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_URL}/api/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (data.ok) {
      setStatus("Processing...");
      setMediaURL(URL.createObjectURL(file));
      // Start polling for completion
      pollForCompletion();
    } else setStatus(`Upload error: ${data.error || "unknown"}`);
  };

  const pollForCompletion = async () => {
    setPollCount(0);
    setProgress(0);
    const interval = setInterval(async () => {
      try {
        setPollCount(prev => prev + 1);
        
        // First check progress
        const progressRes = await fetch(`${API_URL}/api/progress`);
        const progressData = await progressRes.json();
        if (progressData.ok) {
          setProgress(progressData.progress);
          if (progressData.progress >= 100) {
            setStatus("Candidates ready.");
            clearInterval(interval);
            return;
          }
        }
        
        // Then check if candidates are ready
        const res = await fetch(`${API_URL}/api/candidates`);
        const data = await res.json();
        if (data.ok && data.candidates && data.candidates.length > 0) {
          setStatus("Candidates ready.");
          setProgress(100);
          clearInterval(interval);
        } else if (data.error && data.error.includes("still processing")) {
          // Still processing, continue polling
          setStatus("Processing...");
        } else if (data.error && data.error.includes("No episodes found")) {
          // Server restarted, no episodes found
          setStatus("Server restarted. Please upload again.");
          clearInterval(interval);
        } else {
          // Something went wrong
          setStatus("Processing failed. Please try again.");
          clearInterval(interval);
        }
      } catch (error) {
        console.error("Polling error:", error);
        // If we can't reach the server, it might have restarted
        if (error.message.includes("fetch")) {
          setStatus("Server unreachable. Please wait for restart...");
        }
      }
    }, 15000); // Check every 15 seconds (much more reasonable)

    // Stop polling after 10 minutes (300 seconds)
    setTimeout(() => {
      clearInterval(interval);
      if (status === "Processing...") {
        setStatus("Processing timeout. Please try again.");
      }
    }, 300000);
  };

  const findCandidates = async () => {
    setStatus("Finding candidates…");
    const res = await fetch(`${API_URL}/api/candidates`);
    const data = await res.json();
    if (data.ok) {
      setCands(data.candidates || []);
      if (data.candidates?.length) {
        setSelected(0);
        setRange({ start: data.candidates[0].start, end: data.candidates[0].end });
      }
      setStatus("Candidates ready.");
    } else setStatus(`Error: ${data.error || "unknown"}`);
  };

  const pick = (i: number) => {
    setSelected(i);
    const c = cands[i];
    setRange({ start: c.start, end: c.end });
    
    // Auto-scroll to waveform section
    const waveformSection = document.querySelector('h3');
    if (waveformSection) {
      waveformSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const nudge = (d: number) => {
    const dur = Math.max(0.5, range.end - range.start);
    const s = Math.max(0, range.start + d);
    setRange({ start: s, end: s + dur });
  };
  
  const tighten = (d: number) => setRange(r => ({ start: r.start, end: Math.max(r.start + 0.5, r.end + d) }));

  const renderOne = async (accepted = true) => {
    setStatus("Rendering…");
    const form = new FormData();
    form.append("start", String(range.start));
    form.append("end", String(range.end));
    const res = await fetch(`${API_URL}/api/render-one`, { method: "POST", body: form });
    const data = await res.json();
    if (data.ok) {
      setOutputs(prev => [...prev, data.output]);
      setStatus("Rendered.");
      // log metrics for learning later
      const cand = cands[selected] || { score: 0, features: {} };
      await fetch(`${API_URL}/api/metrics/log-choice`, {
        method: "POST", 
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          start: range.start, 
          end: range.end, 
          accepted,
          score: cand.score || 0, 
          features: cand.features || {}
        })
      });
    } else setStatus(`Render error: ${data.error || "unknown"}`);
  };

  return (
    <div style={{display: "grid", gap: 12, maxWidth: 960, margin: "0 auto", padding: "20px"}}>
      <h2>PodPromo AI</h2>

      <input type="file" accept="audio/*,video/*" onChange={e => setFile(e.target.files?.[0] || null)} />
      <div style={{display: "flex", gap: 8}}>
        <button onClick={upload} disabled={!file}>Upload</button>
        {status === "Processing..." && (
          <div style={{display: "flex", flexDirection: "column", gap: 8, alignItems: "center"}}>
            <button disabled>Processing... {progress > 0 ? `${progress}%` : ''} (Checked {pollCount}x)</button>
            {progress > 0 && (
              <div style={{width: "100%", maxWidth: 300, height: 8, backgroundColor: "#eee", borderRadius: 4, overflow: "hidden"}}>
                <div style={{
                  width: `${progress}%`, 
                  height: "100%", 
                  backgroundColor: "#007bff", 
                  transition: "width 0.3s ease"
                }} />
              </div>
            )}
          </div>
        )}
        {status === "Candidates ready." && (
          <button onClick={findCandidates}>Show Me Highlights</button>
        )}
      </div>

      <Controls />

      <p>{status}</p>

      {!!mediaURL && (
        <>
          <h3>Waveform</h3>
          <Waveform src={mediaURL} start={range.start} end={range.end} onChange={setRange} />
          <div style={{display: "flex", gap: 8}}>
            <button onClick={() => nudge(-0.5)}>-0.5s</button>
            <button onClick={() => nudge(0.5)}>+0.5s</button>
            <button onClick={() => tighten(-0.5)}>End -0.5s</button>
            <button onClick={() => tighten(0.5)}>End +0.5s</button>
            <button onClick={() => renderOne(true)}>Render Selected</button>
          </div>
          <div>Selection: {range.start.toFixed(2)}s → {range.end.toFixed(2)}s</div>
        </>
      )}

      {cands.length > 0 && (
        <>
          <h3>Top Candidates</h3>
          <ol>
            {cands.map((c, i) => (
              <li key={i} style={{marginBottom: 16, padding: 12, border: "1px solid #e5e7eb", borderRadius: 8}}>
                <div style={{display: "flex", alignItems: "center", gap: 8, marginBottom: 8}}>
                  <label style={{cursor: "pointer", display: "flex", alignItems: "center", gap: 8}}>
                    <input type="radio" name="cand" checked={selected === i} onChange={() => pick(i)} />
                    <b>ClipScore: {c.score?.toFixed(2) ?? "—"}</b>
                  </label>
                  <span style={{color: "#666"}}>{c.start.toFixed(1)}s → {c.end.toFixed(1)}s</span>
                  <span style={{color: "#666"}}>({(c.end - c.start).toFixed(1)}s)</span>
                </div>
                
                <div style={{marginBottom: 12, opacity: 0.8, lineHeight: 1.4}}>
                  {(c.text || "").slice(0, 140)}…
                </div>
                
                <div style={{display: "flex", gap: 8, alignItems: "center"}}>
                  <button 
                    onClick={() => {
                      setRange({ start: c.start, end: c.end });
                      // Auto-play the audio at this timestamp
                      const audio = document.querySelector('audio') as HTMLAudioElement;
                      if (audio) {
                        audio.currentTime = c.start;
                        audio.play();
                      }
                    }}
                    style={{
                      padding: "6px 12px",
                      backgroundColor: "#007bff",
                      color: "white",
                      border: "none",
                      borderRadius: 4,
                      cursor: "pointer",
                      fontSize: "14px"
                    }}
                  >
                    🔊 Preview Audio
                  </button>
                  
                  <button 
                    onClick={() => setRange({ start: c.start, end: c.end })}
                    style={{
                      padding: "6px 12px",
                      backgroundColor: "#28a745",
                      color: "white",
                      border: "none",
                      borderRadius: 4,
                      cursor: "pointer",
                      fontSize: "14px"
                    }}
                  >
                    📍 Set Range
                  </button>
                </div>
              </li>
            ))}
          </ol>
        </>
      )}

      {outputs.length > 0 && (
        <>
          <h3>Rendered Clips</h3>
          <ul>{outputs.map((o, i) => <li key={i}>
            <video controls width={300} src={`${API_URL}/files/${o}`} />
          </li>)}</ul>
        </>
      )}
    </div>
  );
}
