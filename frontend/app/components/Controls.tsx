"use client";
import { useEffect, useState } from "react";
import { API_URL } from "../lib/config";

type Weights = {
  hook: number; prosody: number; emotion: number; q_or_list: number;
  payoff: number; info: number; loop: number;
};

export default function Controls() {
  const [weights, setWeights] = useState<Weights>({
    hook: 0.35, prosody: 0.20, emotion: 0.15, q_or_list: 0.10,
    payoff: 0.10, info: 0.05, loop: 0.05
  });
  const [status, setStatus] = useState("");

  useEffect(() => {
    (async () => {
      const res = await fetch(`${API_URL}/config/get`);
      const data = await res.json();
      if (data?.ok) setWeights(data.config.weights);
    })();
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
      setStatus("Weights applied.");
    } else {
      setStatus(`Error: ${data.error || "unknown"}`);
    }
  };

  const reload = async () => {
    setStatus("Reloading config…");
    const res = await fetch(`${API_URL}/config/reload`, { method: "POST" });
    const data = await res.json();
    if (data.ok) setStatus("Reloaded.");
    else setStatus("Reload failed.");
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
        <button onClick={()=>setPreset("business")}>Preset: Business</button>
        <button onClick={()=>setPreset("comedy")}>Preset: Comedy</button>
        <button onClick={reload}>Reload</button>
      </div>
      <div style={{display:"grid", gap:6}}>
        {slider("hook","Hook")}
        {slider("prosody","Prosody")}
        {slider("emotion","Emotion")}
        {slider("q_or_list","Question/List")}
        {slider("payoff","Payoff")}
        {slider("info","Info Density")}
        {slider("loop","Loopability")}
      </div>
      <div style={{marginTop:8}}>
        <button onClick={applyWeights}>Apply Weights</button>
      </div>
      <div style={{opacity:0.7, marginTop:6}}>{status}</div>
      <small style={{opacity:0.6}}>Tip: after applying weights, click "Find Candidates" again to see new rankings.</small>
    </div>
  );
}
