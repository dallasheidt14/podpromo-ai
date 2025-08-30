"use client";
import React from "react";

const LABELS: Record<string,string> = {
  hook_score:"Hook", arousal_score:"Arousal", payoff_score:"Payoff",
  info_density:"Info", loopability:"Loop", question_score:"Q/List",
  platform_len_match:"Length"
};

function Bar({label, value}:{label:string; value:number}) {
  const pct = Math.round((value||0)*100);
  return (
    <div style={{margin:"6px 0"}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:12,opacity:.8}}>
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style={{height:8, background:"#eee", borderRadius:6}}>
        <div style={{height:8, width:`${pct}%`, background:"#0ea5e9", borderRadius:6}}/>
      </div>
    </div>
  );
}

export default function CandidateDetailsModal({
  open, onClose, cand
}:{ open:boolean; onClose:()=>void; cand:any }) {
  if (!open || !cand) return null;
  const f = cand.features || {};
  return (
    <div style={{position:"fixed",inset:0, background:"rgba(0,0,0,.45)", display:"flex",alignItems:"center",justifyContent:"center", padding:16, zIndex:50}}>
      <div style={{background:"#fff", borderRadius:12, width:"min(820px, 96vw)", maxHeight:"90vh", overflow:"auto", padding:16}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <h3 style={{margin:0}}>Clip details {cand.start?.toFixed(1)}–{cand.end?.toFixed(1)}s</h3>
          <button onClick={onClose}>✕</button>
        </div>

        {cand.why_summary && <p style={{marginTop:8, fontWeight:500}}>{cand.why_summary}</p>}

        <h4 style={{margin:"12px 0 6px"}}>Transcript</h4>
        <pre style={{whiteSpace:"pre-wrap", fontSize:14, background:"#fafafa", padding:12, borderRadius:8, border:"1px solid #eee"}}>
          {cand.full_text || cand.text}
        </pre>

        <h4 style={{margin:"12px 0 6px"}}>Signals</h4>
        {Object.entries(LABELS).map(([k,label]) => (
          k in f ? <Bar key={k} label={label} value={f[k]} /> : null
        ))}

        {cand.explain?.top?.length ? (
          <>
            <h4 style={{margin:"12px 0 6px"}}>Top contributors</h4>
            <ul style={{marginTop:0}}>
              {cand.explain.top.map(([k,v]:[string,number])=>(
                <li key={k}>{LABELS[k]||k}: {(v||0).toFixed(2)}</li>
              ))}
            </ul>
          </>
        ) : null}

        {cand.caps?.length ? (
          <>
            <h4 style={{margin:"12px 0 6px"}}>Applied guards (debug)</h4>
            <code>{cand.caps.join(", ")}</code>
          </>
        ):null}
      </div>
    </div>
  );
}

