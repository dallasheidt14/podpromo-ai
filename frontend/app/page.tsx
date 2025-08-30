"use client";

import * as React from "react";
import { useState } from "react";
import Controls from "./components/Controls";
import Waveform from "./components/Waveform";
import CandidateDetailsModal from "./components/CandidateDetailsModal";
import { API_URL } from "./lib/config";

// Helper function for viral potential bands with stricter thresholds
function viralBand(score100: number) {
  if (score100 >= 75) return { label: "🔥 HIGH", color: "#10b981" };      // green
  if (score100 >= 50) return { label: "⚡ MED", color: "#f59e0b" };       // amber
  return { label: "📉 LOW", color: "#ef4444" };                           // red
}

type Candidate = { 
  start: number; 
  end: number; 
  text?: string; 
  score: number; 
  features?: any;
  raw_score?: number;
  clip_score_100?: number;
  synergy_mult?: number;
  flagged?: boolean;
  prosody?: any;
  display_score?: number;
  confidence?: string;
  confidence_color?: string;
};

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
  const [renderStyle, setRenderStyle] = useState("bold");
  const [enableCaptions, setEnableCaptions] = useState(true);
  const [enablePunchIns, setEnablePunchIns] = useState(true);
  const [enableLoopSeam, setEnableLoopSeam] = useState(true);
  const [abTests, setAbTests] = useState<any>(null);
  const [showABTests, setShowABTests] = useState(false);
  const [tooltip, setTooltip] = useState<{text: string, x: number, y: number} | null>(null);
  const [platform, setPlatform] = useState<"tiktok_reels"|"shorts"|"linkedin_sq">("tiktok_reels");
  const [genre, setGenre] = useState<string | undefined>(undefined);
  const [modalOpen, setModalOpen] = useState(false);
  const [activeCandidate, setActiveCandidate] = useState<Candidate | null>(null);
  const [activeTab, setActiveTab] = useState<"candidates">("candidates");
  const [debugMode, setDebugMode] = useState(false);

  // Helper function to convert seconds to readable time format
  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
  };

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
    const url = `${API_URL}/api/candidates?platform=${platform}${genre ? `&genre=${genre}` : ""}&debug=1`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.ok) {
      setCands(data.candidates || []);
      if (data.candidates?.length) {
        setSelected(0);
        setRange({ start: data.candidates[0].start, end: data.candidates[0].end });
      }
      setStatus("Candidates ready.");
      // Dispatch event to reset weights applied flag in Controls component
      window.dispatchEvent(new CustomEvent('candidatesUpdated'));
    } else setStatus(`Error: ${data.error || "unknown"}`);
  };

  const rescoreCandidates = async () => {
    if (!cands.length) {
      setStatus("No candidates to re-score. Please find candidates first.");
      return;
    }
    
    setStatus("Re-scoring candidates with current weights...");
    const url = `${API_URL}/api/candidates?platform=${platform}${genre ? `&genre=${genre}` : ""}&debug=1`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.ok) {
      setCands(data.candidates || []);
      setStatus("Candidates re-scored with new weights!");
    } else setStatus(`Re-score error: ${data.error || "unknown"}`);
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
    form.append("style", renderStyle);
    form.append("captions", String(enableCaptions));
    form.append("punch_ins", String(enablePunchIns));
    form.append("loop_seam", String(enableLoopSeam));
    
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
          features: cand.features || {},
          render_options: {
            style: renderStyle,
            captions: enableCaptions,
            punch_ins: enablePunchIns,
            loop_seam: enableLoopSeam
          }
        })
      });
    } else setStatus(`Render error: ${data.error || "unknown"}`);
  };

  const exportSquare = async () => {
    if (!cands[selected]) return;
    
    setStatus("Exporting square format...");
    
    // Create a unique clip ID for this export
    const clip_id = `clip_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const body = { 
      clip_id: clip_id,
      variant: "square", 
      style: renderStyle, 
      captions: enableCaptions 
    };
    
    try {
      console.log("Exporting square with body:", body);
      
      const res = await fetch(`${API_URL}/api/render-variant`, {
        method: "POST", 
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      
      const data = await res.json();
      console.log("Export response:", data);
      
      if (data.ok || data.success) {
        const outputName = data.output || data.path || `square_export_${clip_id}.mp4`;
        setOutputs(prev => [...prev, outputName]);
        setStatus(`Square export complete! Output: ${outputName}`);
        console.log("Export successful, added to outputs:", outputName);
      } else {
        const errorMsg = data.error || data.message || "Unknown error";
        setStatus(`Square export failed: ${errorMsg}`);
        console.error("Export failed:", errorMsg);
      }
    } catch (error) {
      console.error("Export error:", error);
      setStatus(`Export error: ${error.message || "Network error"}`);
    }
  };

  const createABTests = async () => {
    if (!cands[selected]) return;
    
    setStatus("Creating A/B tests...");
    const form = new FormData();
    form.append("start", String(cands[selected].start));
    form.append("end", String(cands[selected].end));
    
    const res = await fetch(`${API_URL}/api/create-ab-tests`, { method: "POST", body: form });
    const data = await res.json();
    
    if (data.ok) {
      setAbTests(data.ab_tests);
      setShowABTests(true);
      setStatus("A/B tests created!");
    } else {
      setStatus(`A/B test creation failed: ${data.error || "unknown"}`);
    }
  };

  const logABChoice = async (choice: string) => {
    if (!abTests) return;
    
    try {
      await fetch(`${API_URL}/api/log-ab-choice`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          test_id: abTests.test_id,
          choice: choice,
          clip_id: cands[selected]?.start + "_" + cands[selected]?.end
        })
      });
      
      setStatus(`A/B choice logged: ${choice}`);
    } catch (error) {
      console.error("Failed to log A/B choice:", error);
    }
  };

  const flagAsWrong = async (candidate: Candidate, index: number) => {
    try {
      const feedback = {
        timestamp: new Date().toISOString(),
        episode_id: "current", // You might want to get this from context
        clip_start: candidate.start,
        clip_end: candidate.end,
        clip_score: candidate.clip_score_100 || candidate.score,
        features: candidate.features,
        synergy_mult: candidate.synergy_mult,
        reason: "Flagged as wrong by user",
        user_rating: "wrong"
      };
      
      // Send feedback to backend
      const res = await fetch(`${API_URL}/api/feedback/flag-wrong`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(feedback)
      });
      
      if (res.ok) {
        setStatus(`Clip flagged as wrong - feedback logged for improvement`);
        // Optionally remove from candidates list or mark as flagged
        setCands(prev => prev.map((c, i) => 
          i === index ? { ...c, flagged: true } : c
        ));
      } else {
        setStatus("Failed to log feedback");
      }
    } catch (error) {
      console.error("Failed to flag clip:", error);
      setStatus("Error logging feedback");
    }
  };

  return (
    <>
      {/* CSS to prevent browser tooltips */}
      <style jsx>{`
        * {
          -webkit-tooltip: none !important;
          tooltip: none !important;
        }
      `}</style>
      
      {/* Modern Tooltip */}
      {tooltip && (
        <div style={{
          position: "fixed",
          left: tooltip.x + 10,
          top: tooltip.y - 10,
          backgroundColor: "#1a1a1a",
          color: "white",
          padding: "12px 16px",
          borderRadius: "8px",
          fontSize: "14px",
          maxWidth: "300px",
          zIndex: 1000,
          boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
          border: "1px solid #333",
          pointerEvents: "none"
        }}>
          {tooltip.text}
          <div style={{
            position: "absolute",
            left: "-6px",
            top: "15px",
            width: 0,
            height: 0,
            borderTop: "6px solid transparent",
            borderBottom: "6px solid transparent",
            borderRight: "6px solid #1a1a1a"
          }} />
        </div>
      )}
      
      <div style={{display: "grid", gap: 12, maxWidth: 960, margin: "0 auto", padding: "20px"}}>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "center"}}>
        <h2>PodPromo AI</h2>
        <a 
          href="/history"
          style={{
            padding: "8px 16px",
            backgroundColor: "#6b7280",
            color: "white",
            textDecoration: "none",
            borderRadius: 6,
            fontSize: "14px"
          }}
        >
          📹 Clip History
        </a>
      </div>

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
            <div style={{fontSize: "12px", color: "#666", textAlign: "center"}}>
              {progress > 0 && progress < 100 && (
                <>
                  <div>Transcribing audio... This may take 5-10 minutes for long episodes</div>
                  <div>Current speed: ~{Math.round(progress / (Date.now() - (window as any).startTime || 1) * 60000)}% per minute</div>
                </>
              )}
            </div>
          </div>
        )}
        {status === "Candidates ready." && (
          <>
            <div style={{
              display: "flex", 
              gap: 12, 
              alignItems: "center", 
              flexWrap: "wrap", 
              margin: "8px 0 16px",
              padding: "16px",
              backgroundColor: "#f8f9fa",
              border: "1px solid #e5e7eb",
              borderRadius: "8px"
            }}>
              <label style={{display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", fontWeight: "500"}}>
                Platform:&nbsp;
                <select 
                  value={platform} 
                  onChange={e=>setPlatform(e.target.value as any)}
                  style={{
                    padding: "6px 8px",
                    borderRadius: "4px",
                    border: "1px solid #ddd",
                    fontSize: "14px"
                  }}
                >
                  <option value="tiktok_reels">TikTok / Reels</option>
                  <option value="shorts">YouTube Shorts</option>
                  <option value="linkedin_sq">LinkedIn (Square)</option>
                </select>
              </label>

              <label style={{display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", fontWeight: "500"}}>
                Genre:&nbsp;
                <select value={genre} onChange={(e) => setGenre(e.target.value || undefined)} className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option value="">Auto-detect</option>
                  <option value="comedy">Comedy / Entertainment</option>
                  <option value="fantasy_sports">Fantasy Sports</option>
                  <option value="sports">Sports</option>
                  <option value="true_crime">True Crime</option>
                  <option value="business">Business / Entrepreneurship</option>
                  <option value="news_politics">News / Politics</option>
                  <option value="education">Educational / Science</option>
                  <option value="health_wellness">Health / Wellness</option>
                </select>
              </label>
              <div style={{
                fontSize: "12px",
                color: "#666",
                backgroundColor: "#e9ecef",
                padding: "4px 8px",
                borderRadius: "4px",
                border: "1px solid #dee2e6"
              }}>
                🎯 Optimizing for {platform.replace("_", " ")} {genre ? `+ ${genre.replace("_", " ")}` : ""}
              </div>
              <button 
                onClick={() => setDebugMode(!debugMode)}
                style={{
                  padding: "4px 8px",
                  fontSize: "12px",
                  backgroundColor: debugMode ? "#dc3545" : "#6c757d",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer"
                }}
                title={debugMode ? "Disable debug mode" : "Enable debug mode"}
              >
                {debugMode ? "🔴 Debug ON" : "⚫ Debug OFF"}
              </button>
            </div>
            <button onClick={findCandidates}>Show Me Highlights</button>
          </>
        )}
        {cands.length > 0 && (
          <button 
            onClick={rescoreCandidates}
            style={{
              backgroundColor: "#17a2b8",
              color: "white",
              border: "none",
              padding: "8px 16px",
              borderRadius: 4,
              cursor: "pointer"
            }}
          >
            🔄 Re-score with New Weights
          </button>
        )}
      </div>

             {/* Controls hidden for cleaner UI - weights are still applied automatically */}

      <p>{status}</p>

      {!!mediaURL && (
        <>
          <h3>Waveform</h3>
          <Waveform src={mediaURL} start={range.start} end={range.end} onChange={setRange} />
          
          {/* Enhanced Rendering Options */}
          <div style={{border: "1px solid #e5e7eb", padding: 16, borderRadius: 8, backgroundColor: "#f8f9fa"}}>
            <h4 style={{margin: "0 0 12px 0", fontSize: "16px"}}>🎬 Rendering Options</h4>
            
            <div style={{display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16}}>
              {/* Caption Style */}
              <div>
                <label style={{display: "block", marginBottom: 4, fontSize: "14px", fontWeight: "bold"}}>
                  Caption Style:
                </label>
                <select 
                  value={renderStyle} 
                  onChange={(e) => setRenderStyle(e.target.value)}
                  style={{width: "100%", padding: "8px", borderRadius: 4, border: "1px solid #ddd"}}
                >
                  <option value="bold">Bold (High Impact)</option>
                  <option value="clean">Clean (Minimal)</option>
                  <option value="caption-heavy">Caption Heavy (Text Focus)</option>
                </select>
              </div>
              
              {/* Feature Toggles */}
              <div>
                <label style={{display: "block", marginBottom: 8, fontSize: "14px", fontWeight: "bold"}}>
                  Features:
                </label>
                <div style={{display: "flex", flexDirection: "column", gap: 8}}>
                  <label style={{display: "flex", alignItems: "center", gap: 8}}>
                    <input 
                      type="checkbox" 
                      checked={enableCaptions} 
                      onChange={(e) => setEnableCaptions(e.target.checked)}
                    />
                    <span style={{fontSize: "14px"}}>Captions</span>
                  </label>
                  <label style={{display: "flex", alignItems: "center", gap: 8}}>
                    <input 
                      type="checkbox" 
                      checked={enablePunchIns} 
                      onChange={(e) => setEnablePunchIns(e.target.checked)}
                    />
                    <span style={{fontSize: "14px"}}>Punch-ins</span>
                  </label>
                  <label style={{display: "flex", alignItems: "center", gap: 8}}>
                    <input 
                      type="checkbox" 
                      checked={enableLoopSeam} 
                      onChange={(e) => setEnableLoopSeam(e.target.checked)}
                    />
                    <span style={{fontSize: "14px"}}>Loop Seam</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
          
          <div style={{display: "flex", gap: 8, flexWrap: "wrap"}}>
            <button onClick={() => nudge(-0.5)}>-0.5s</button>
            <button onClick={() => nudge(0.5)}>+0.5s</button>
            <button onClick={() => tighten(-0.5)}>End -0.5s</button>
            <button onClick={() => tighten(0.5)}>End +0.5s</button>
            <button onClick={() => renderOne(true)}>Render Selected</button>
            <button 
              onClick={exportSquare}
              style={{
                backgroundColor: "#ff6b35",
                color: "white"
              }}
            >
              🎬 Export Square 1:1
            </button>
          </div>
                     <div style={{
             backgroundColor: "#e3f2fd",
             padding: "12px",
             borderRadius: "6px",
             border: "1px solid #2196f3",
             marginTop: "8px"
           }}>
             <div style={{fontWeight: "bold", color: "#1976d2", marginBottom: "4px"}}>
               🎯 Current Selection
             </div>
             <div style={{fontSize: "14px", color: "#424242"}}>
               Start: {formatTime(range.start)} | End: {formatTime(range.end)} | Duration: {formatTime(range.end - range.start)}
             </div>
           </div>
        </>
      )}

             {cands.length > 0 && (
         <>
           <h3>🎯 AI-Powered Clip Candidates</h3>
           
           {/* Tab Navigation */}
           <div style={{
             display: "flex",
             borderBottom: "2px solid #e9ecef",
             marginBottom: "20px"
           }}>
             <button
               onClick={() => setActiveTab("candidates")}
               style={{
                 padding: "12px 24px",
                 border: "none",
                 backgroundColor: activeTab === "candidates" ? "#007bff" : "transparent",
                 color: activeTab === "candidates" ? "white" : "#6c757d",
                 borderRadius: "8px 8px 0 0",
                 cursor: "pointer",
                 fontWeight: activeTab === "candidates" ? "600" : "400"
               }}
             >
               📊 All Candidates
             </button>
           </div>
           
           {/* Viral Insights Panel */}
           <div style={{
             backgroundColor: "#f8f9fa",
             border: "1px solid #e9ecef",
             borderRadius: 8,
             padding: "16px",
             marginBottom: "20px"
           }}>
            <h4 style={{margin: "0 0 12px 0", color: "#495057", fontSize: "16px"}}>
              🚀 Viral Insights & AI Analysis
            </h4>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "16px"
            }}>
              <div style={{textAlign: "center"}}>
                <div style={{fontSize: "24px", fontWeight: "bold", color: "#28a745"}}>
                  {cands.length}
                </div>
                <div style={{fontSize: "12px", color: "#6c757d"}}>Viral Moments Found</div>
              </div>
                             <div style={{textAlign: "center"}}>
                 <div style={{fontSize: "24px", fontWeight: "bold", color: "#007bff"}}>
                   {Math.max(...cands.map(c => c.display_score ?? c.clip_score_100 ?? Math.round((c.score || 0) * 100)))}%
                 </div>
                 <div style={{fontSize: "12px", color: "#6c757d"}}>Top ClipScore</div>
               </div>
               <div style={{textAlign: "center"}}>
                 <div style={{fontSize: "24px", fontWeight: "bold", color: "#ffc107"}}>
                   {cands.filter(c => (c.display_score ?? c.clip_score_100 ?? Math.round((c.score || 0) * 100)) >= 75).length}
                 </div>
                 <div style={{fontSize: "12px", color: "#6c757d"}}>High Viral Potential</div>
               </div>
              <div style={{textAlign: "center"}}>
                <div style={{fontSize: "24px", fontWeight: "bold", color: "#e83e8c"}}>
                  {cands.filter(c => c.features?.arousal_score > 0.6).length}
                </div>
                <div style={{fontSize: "12px", color: "#6c757d"}}>High Energy Clips</div>
              </div>
            </div>
          </div>
          {/* Tab Content */}
          {activeTab === "candidates" && (
            <ol>
              {cands.map((c, i) => (
              <li key={i} style={{marginBottom: 16, padding: 12, border: "1px solid #e5e7eb", borderRadius: 8}}>
                <div style={{display: "flex", alignItems: "center", gap: 8, marginBottom: 8}}>
                  <label style={{cursor: "pointer", display: "flex", alignItems: "center", gap: 8}}>
                    <input type="radio" name="cand" checked={selected === i} onChange={() => pick(i)} />
                    <b>ClipScore: {c.display_score ?? c.clip_score_100 ?? Math.round((c.score || 0) * 100)}/100</b>
                  </label>
                  {c.confidence && (
                    <span style={{
                      padding: "2px 8px",
                      backgroundColor: "#f8f9fa",
                      borderRadius: "12px",
                      fontSize: "12px",
                      fontWeight: "500",
                      color: "#495057"
                    }}>
                      {c.confidence}
                    </span>
                  )}
                  <span style={{color: "#666"}}>{formatTime(c.start)} → {formatTime(c.end)}</span>
                  <span style={{color: "#666"}}>({formatTime(c.end - c.start)})</span>
                </div>
                
                {/* Carried By Badge */}
                {c.synergy_mult && (
                  <div style={{marginBottom: 8}}>
                    {c.synergy_mult < 0.8 ? (
                      <div style={{
                        display: "inline-block",
                        padding: "4px 8px",
                        backgroundColor: "#fff3cd",
                        color: "#856404",
                        border: "1px solid #ffeaa7",
                        borderRadius: "4px",
                        fontSize: "12px",
                        fontWeight: "500"
                      }}>
                        ⚠️ High score but low energy (may not engage viewers)
                      </div>
                    ) : c.synergy_mult > 0.95 ? (
                      <div style={{
                        display: "inline-block",
                        padding: "4px 8px",
                        backgroundColor: "#d4edda",
                        color: "#155724",
                        border: "1px solid #c3e6cb",
                        borderRadius: "4px",
                        fontSize: "12px",
                        fontWeight: "500"
                      }}>
                        ✅ Strong balance (Hook+Energy+Payoff)
                      </div>
                    ) : (
                      <div style={{
                        display: "inline-block",
                        padding: "4px 8px",
                        backgroundColor: "#e2e3e5",
                        color: "#383d41",
                        border: "1px solid #d6d8db",
                        borderRadius: "4px",
                        fontSize: "12px",
                        fontWeight: "500"
                      }}>
                        🔄 Mixed signals (some factors strong, others weak)
                      </div>
                    )}
                  </div>
                )}

                                  {/* Debug Prosody Readout (Admin Only) */}
                  {debugMode && c.prosody && (
                    <div style={{fontSize:12, opacity:.7, marginTop:6, marginBottom:8}}>
                      prosody: rv {c.prosody.rms_var?.toFixed(2)} • rd {c.prosody.rms_delta?.toFixed(2)}
                      • lift {c.prosody.rms_lift_0_3s?.toFixed(2)} • pauses {c.prosody.pause_frac?.toFixed(2)}
                      • cent {c.prosody.centroid_var?.toFixed(2)} • laugh {String(c.prosody.laugh_flag)}
                      • f0var {c.prosody.f0_var?.toFixed(2)} • f0rng {c.prosody.f0_range?.toFixed(2)} • voiced {c.prosody.voiced_frac?.toFixed(2)}
                    </div>
                  )}
                
                {/* AI Cards - Viral Meter & Features */}
                <div style={{
                  display: "grid", 
                  gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", 
                  gap: 8, 
                  marginBottom: 12,
                  padding: "8px",
                  backgroundColor: "#f8f9fa",
                  borderRadius: 6
                }}>
                                     {/* Viral Meter */}
                   <div 
                     style={{
                       textAlign: "center",
                       padding: "8px",
                       backgroundColor: "#fff",
                       borderRadius: 4,
                       border: "1px solid #e9ecef",
                       cursor: "help",
                       position: "relative"
                     }}
                     onMouseEnter={(e) => setTooltip({
                       text: "Viral Potential: How likely this clip is to go viral on social media. Based on AI analysis of engagement factors like hook strength, emotional impact, and shareability.",
                       x: e.clientX,
                       y: e.clientY
                     })}
                     onMouseLeave={() => setTooltip(null)}
                   >
                     <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Viral Potential</div>
                     <div style={{
                       fontSize: "16px", 
                       fontWeight: "bold",
                       color: viralBand(c.clip_score_100 ?? Math.round((c.score || 0) * 100)).color
                     }}>
                       {viralBand(c.clip_score_100 ?? Math.round((c.score || 0) * 100)).label}
                     </div>
                   </div>
                  
                                                         {/* Hook Score */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Hook Power: How effectively this clip grabs attention in the first few seconds. Higher scores mean stronger opening hooks that keep viewers engaged.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Hook Power</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#007bff"}}>
                        {Math.round((c.features?.hook_score || 0) * 100)}%
                      </div>
                    </div>
                  
                                                         {/* Arousal Score (V2) */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Arousal/Energy: How energetic and engaging this clip is. Combines vocal energy, exclamation density, and laughter indicators for maximum viewer engagement.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Arousal/Energy</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#e83e8c"}}>
                        {Math.round((c.features?.arousal_score || 0) * 100)}%
                      </div>
                    </div>

                                       {/* Question/List Score (V2) */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Question/List: How well this clip uses questions or list structures to maintain viewer engagement. Higher scores mean better content structure.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Q/List</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#fd7e14"}}>
                        {Math.round((c.features?.question_score || 0) * 100)}%
                      </div>
                    </div>
                  
                                                         {/* Loopability */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Loopable: How well this clip works when played on repeat. Higher scores mean the clip flows naturally from end to beginning, perfect for TikTok loops.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Loopable</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#17a2b8"}}>
                        {Math.round((c.features?.loopability || 0) * 100)}%
                      </div>
                    </div>
                  
                                                                             {/* Platform Length Match (V2) */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Platform Length Match: How well this clip's duration fits the optimal length range for your selected platform. Higher scores mean better length optimization.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Length Match</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#6f42c1"}}>
                        {Math.round((c.features?.platform_len_match || 0) * 100)}%
                      </div>
                    </div>

                    {/* Payoff Score */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Payoff Score: How quickly this clip delivers value to viewers. Higher scores mean viewers get rewarded for watching, increasing engagement and sharing.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Payoff</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#20c997"}}>
                        {Math.round((c.features?.payoff_score || 0) * 100)}%
                      </div>
                    </div>

                    {/* Info Density */}
                    <div 
                      style={{
                        textAlign: "center",
                        padding: "8px",
                        backgroundColor: "#fff",
                        borderRadius: 4,
                        border: "1px solid #e9ecef",
                        cursor: "help"
                      }}
                      onMouseEnter={(e) => setTooltip({
                        text: "Info Density: How much valuable content is packed into this clip. Higher scores mean minimal filler words and maximum substance per second.",
                        x: e.clientX,
                        y: e.clientY
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    >
                      <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Info Density</div>
                      <div style={{fontSize: "14px", fontWeight: "bold", color: "#fd7e14"}}>
                        {Math.round((c.features?.info_density || 0) * 100)}%
                      </div>
                    </div>

                    {/* Platform Recommendation */}
                   <div 
                     style={{
                       textAlign: "center",
                       padding: "8px",
                       backgroundColor: "#fff",
                       borderRadius: 4,
                       border: "1px solid #e9ecef",
                       cursor: "help"
                     }}
                     onMouseEnter={(e) => setTooltip({
                       text: "Best For: Our AI's recommendation for which social media platform this clip will perform best on, based on length, style, and engagement potential.",
                       x: e.clientX,
                       y: e.clientY
                     })}
                     onMouseLeave={() => setTooltip(null)}
                   >
                     <div style={{fontSize: "10px", color: "#6c757d", marginBottom: 2}}>Best For</div>
                     <div style={{fontSize: "12px", fontWeight: "bold", color: "#6f42c1"}}>
                       {(c.clip_score_100 ?? Math.round((c.score || 0) * 100)) >= 75 ? "🎯 TikTok/Reels" : (c.clip_score_100 ?? Math.round((c.score || 0) * 100)) >= 50 ? "📱 Instagram" : "💬 Twitter"}
                     </div>
                   </div>
                </div>
                
                <div style={{marginBottom: 12, opacity: 0.8, lineHeight: 1.4}}>
                  {(c.text || "").slice(0, 140)}…
                </div>
                
                <div style={{display: "flex", gap: 8, alignItems: "center"}}>
                                     <button 
                     onClick={() => {
                       setRange({ start: c.start, end: c.end });
                       setStatus(`Previewing audio from ${formatTime(c.start)} → ${formatTime(c.end)}`);
                       
                       // Auto-play the audio at this timestamp
                       const audio = document.querySelector('audio') as HTMLAudioElement;
                       if (audio) {
                         audio.currentTime = c.start;
                         audio.play().catch(e => {
                           console.log("Audio preview failed:", e);
                           setStatus("Audio preview failed - check browser audio settings");
                         });
                       } else {
                         setStatus("No audio element found for preview");
                       }
                       
                       // Auto-scroll to waveform section
                       const waveformSection = document.querySelector('h3');
                       if (waveformSection) {
                         waveformSection.scrollIntoView({ behavior: 'smooth' });
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
                     onClick={() => {
                       setRange({ start: c.start, end: c.end });
                       setStatus(`Range set to ${formatTime(c.start)} → ${formatTime(c.end)} (${formatTime(c.end - c.start)})`);
                       
                       // Auto-scroll to waveform section
                       const waveformSection = document.querySelector('h3');
                       if (waveformSection) {
                         waveformSection.scrollIntoView({ behavior: 'smooth' });
                       }
                     }}
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
                   
                   <button 
                     onClick={() => {
                       setActiveCandidate(c);
                       setModalOpen(true);
                     }}
                     style={{
                       padding: "6px 12px",
                       backgroundColor: "#6c757d",
                       color: "white",
                       border: "none",
                       borderRadius: 4,
                       cursor: "pointer",
                       fontSize: "14px"
                     }}
                   >
                     🔍 View Details
                   </button>
                   
                   <button 
                     onClick={() => flagAsWrong(c, i)}
                     style={{
                       padding: "6px 12px",
                       backgroundColor: "#dc3545",
                       color: "white",
                       border: "none",
                       borderRadius: 4,
                       cursor: "pointer",
                       fontSize: "14px"
                     }}
                   >
                     🚩 Flag as Wrong
                   </button>
                </div>
              </li>
            ))}
            </ol>
          )}
        </>
      )}

      {outputs.length > 0 && (
        <>
          <h3>🎬 Rendered Clips ({outputs.length})</h3>
          <div style={{backgroundColor: "#f8f9fa", padding: "16px", borderRadius: "8px", marginBottom: "20px"}}>
            <p style={{margin: "0 0 12px 0", fontSize: "14px", color: "#666"}}>
              Latest exports will appear below. If videos don't load, check the browser console for details.
            </p>
          </div>
          <ul style={{listStyle: "none", padding: 0}}>
            {outputs.map((o, i) => (
              <li key={i} style={{
                marginBottom: "16px", 
                padding: "16px", 
                border: "1px solid #e5e7eb", 
                borderRadius: "8px",
                backgroundColor: "#fff"
              }}>
                <div style={{marginBottom: "8px", fontSize: "14px", color: "#666"}}>
                  Export #{i + 1}: {o}
                </div>
                                 <video 
                   controls 
                   width={300} 
                   preload="metadata"
                   crossOrigin="anonymous"
                   src={`${API_URL}/clips/${o}`}
                   onError={(e) => {
                     console.error(`Video load error for ${o}:`, e);
                     const video = e.target as HTMLVideoElement;
                     console.error(`Video target:`, video);
                     console.error(`Video error details:`, video.error);
                     console.error(`Video readyState:`, video.readyState);
                     console.error(`Video networkState:`, video.networkState);
                   }}
                   onLoadStart={() => console.log(`Loading video: ${o}`)}
                   onLoadedData={() => console.log(`Video loaded successfully: ${o}`)}
                   onCanPlay={() => console.log(`Video can play: ${o}`)}
                   onLoad={() => console.log(`Video load event: ${o}`)}
                   onLoadedMetadata={() => console.log(`Video metadata loaded: ${o}`)}
                   onCanPlayThrough={() => console.log(`Video can play through: ${o}`)}
                 />
                                 <div style={{marginTop: "8px", fontSize: "12px", color: "#999"}}>
                   Video URL: {`${API_URL}/clips/${o}`}
                 </div>
                 
                 {/* Fallback Video Player */}
                 <div style={{marginTop: "12px", padding: "12px", backgroundColor: "#f8f9fa", borderRadius: "4px"}}>
                   <div style={{fontSize: "12px", color: "#666", marginBottom: "8px"}}>
                     🔧 Alternative Video Player (if main player fails):
                   </div>
                   <video 
                     controls 
                     width={300} 
                     preload="none"
                     style={{border: "1px solid #ddd"}}
                   >
                     <source src={`${API_URL}/clips/${o}`} type="video/mp4" />
                     <source src={`${API_URL}/clips/${o}`} type="video/webm" />
                     Your browser does not support the video tag.
                   </video>
                 </div>
                                 <div style={{marginTop: "8px", display: "flex", gap: "8px"}}>
                   <a 
                     href={`${API_URL}/clips/${o}`} 
                     download={o}
                     style={{
                       padding: "6px 12px",
                       backgroundColor: "#28a745",
                       color: "white",
                       textDecoration: "none",
                       borderRadius: "4px",
                       fontSize: "12px",
                       display: "inline-block"
                     }}
                   >
                     📥 Download Video
                   </a>
                   <button
                     onClick={() => {
                       const videoUrl = `${API_URL}/clips/${o}`;
                       console.log(`Testing video URL: ${videoUrl}`);
                       window.open(videoUrl, '_blank');
                     }}
                     style={{
                       padding: "6px 12px",
                       backgroundColor: "#007bff",
                       color: "white",
                       border: "none",
                       borderRadius: "4px",
                       fontSize: "12px",
                       cursor: "pointer"
                     }}
                   >
                     🔗 Test URL
                   </button>
                 </div>
              </li>
            ))}
          </ul>
        </>
      )}
      
      {/* Candidate Details Modal */}
      <CandidateDetailsModal 
        open={modalOpen} 
        onClose={() => setModalOpen(false)} 
        cand={activeCandidate} 
      />
    </div>
    </>
  );
}
