"use client";

export default function ViralMeter({score}:{score:number}) {
  const s = Math.max(0, Math.min(100, score || 0));
  const color = s >= 75 ? "#10b981" : s >= 50 ? "#f59e0b" : "#ef4444";
  const label = s >= 75 ? "High" : s >= 50 ? "Medium" : "Low";
  return (
    <div style={{marginTop:8}}>
      <div style={{display:"flex", justifyContent:"space-between", fontSize:12, opacity:0.8}}>
        <span>Viral Potential</span><span>{label} ({s})</span>
      </div>
      <div style={{height:10, background:"#e5e7eb", borderRadius:6, marginTop:4}}>
        <div style={{height:"100%", width:`${s}%`, background:color, borderRadius:6}}/>
      </div>
    </div>
  );
}
