"use client";
import { useEffect, useState } from "react";
import { API_URL } from "../lib/config";

type Weights = {
  hook: number; arousal: number; payoff: number; info: number; 
  loop: number; q_or_list: number; platform_len: number;
};

export default function Controls() {
  const [weights, setWeights] = useState<Weights>({
    hook: 0.35, arousal: 0.20, payoff: 0.15, info: 0.10,
    loop: 0.08, q_or_list: 0.07, platform_len: 0.05
  });
  const [status, setStatus] = useState("");
  const [weightsApplied, setWeightsApplied] = useState(false);

  useEffect(() => {
    (async () => {
      const res = await fetch(`${API_URL}/config/get`);
      const data = await res.json();
      if (data?.ok) setWeights(data.config.weights);
    })();
    
    // Listen for candidates update event to reset weights applied flag
    const handleCandidatesUpdated = () => {
      setWeightsApplied(false);
    };
    
    window.addEventListener('candidatesUpdated', handleCandidatesUpdated);
    
    return () => {
      window.removeEventListener('candidatesUpdated', handleCandidatesUpdated);
    };
  }, []);

  const setPreset = async (name: string) => {
    setStatus(`Loading preset ${name}…`);
    const res = await fetch(`${API_URL}/config/load-preset`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ name })
    });
    const data = await res.json();
    if (data.ok) {
      setWeights(data.weights);
      setStatus(`Preset ${name} active.`);
      setWeightsApplied(false);
    } else {
      setStatus(`Error: ${data.error || "unknown"}`);
    }
  };

  const updateWeight = (key: keyof Weights, val: number) => {
    const next = {...weights, [key]: val};
    setWeights(next);
  };

  const applyWeights = async () => {
    setStatus("Applying weights…");
    const res = await fetch(`${API_URL}/config/set-weights`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(weights)
    });
    const data = await res.json();
    if (data.ok) {
      setWeights(data.weights);
      setStatus("Weights applied. Click '🔄 Re-score with New Weights' to see changes!");
      setWeightsApplied(true);
    } else {
      setStatus(`Error: ${data.error || "unknown"}`);
    }
  };

  const reload = async () => {
    setStatus("Reloading config…");
    const res = await fetch(`${API_URL}/config/reload`, { method: "POST" });
    const data = await res.json();
    if (data.ok) {
      setStatus("Reloaded.");
      setWeightsApplied(false);
    } else {
      setStatus("Reload failed.");
    }
  };

  const slider = (key: keyof Weights, label: string) => (
    <div key={key} style={{display:"grid", gridTemplateColumns:"140px 1fr 60px", gap:8, alignItems:"center"}}>
      <label>{label}</label>
      <input type="range" min={0.02} max={0.6} step={0.01}
        value={weights[key]}
        onChange={e=>updateWeight(key, parseFloat(e.target.value))} />
      <span>{weights[key].toFixed(2)}</span>
    </div>
  );

  return (
    <div style={{border:"1px solid #e5e7eb", padding:12, borderRadius:8, marginTop:12}}>
      <h3>Scoring Controls</h3>
      <div style={{display:"flex", gap:8, flexWrap:"wrap", marginBottom:8}}>
        <button 
          onClick={()=>setPreset("business")}
          title="Business: Higher weights on hook, payoff, and structured content. Perfect for professional/business podcasts."
        >
          🏢 Business Preset
        </button>
        <button 
          onClick={()=>setPreset("comedy")}
          title="Comedy: Higher weights on arousal/energy, hook, and payoff. Perfect for entertainment/funny content."
        >
          😄 Comedy Preset
        </button>
        <button 
          onClick={reload}
          title="Reload: Reset to default weights and reload configuration from files."
        >
          🔄 Reload
        </button>
      </div>
      <div style={{display:"grid", gap:6}}>
        {slider("hook","Hook")}
        {slider("arousal","Arousal/Energy")}
        {slider("payoff","Payoff")}
        {slider("info","Info Density")}
        {slider("loop","Loopability")}
        {slider("q_or_list","Question/List")}
        {slider("platform_len","Platform Length")}
      </div>
      <div style={{marginTop:8}}>
        <button onClick={applyWeights}>Apply Weights</button>
      </div>
      <div style={{opacity:0.7, marginTop:6}}>{status}</div>
      {weightsApplied && (
        <div style={{
          backgroundColor: "#d4edda",
          border: "1px solid #c3e6cb",
          color: "#155724",
          padding: "8px 12px",
          borderRadius: 4,
          marginTop: 8,
          fontSize: "14px"
        }}>
          ✅ Weights updated! Use &quot;🔄 Re-score with New Weights&quot; button above to see your clips re-ranked.
        </div>
      )}
      <small style={{opacity:0.6}}>💡 Tip: After applying weights, click &quot;🔄 Re-score with New Weights&quot; to see clips re-ranked with your new settings!</small>
    </div>
  );
}
