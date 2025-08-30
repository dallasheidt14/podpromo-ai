"use client";

export default function PlatformBadges({platforms}:{platforms:string[]}) {
  if (!platforms || platforms.length===0) return null;
  const colors: Record<string,string> = {
    "TikTok":"#000000",
    "Instagram Reels":"#d946ef",
    "YouTube Shorts":"#ef4444",
    "LinkedIn":"#0a66c2",
  };
  return (
    <div style={{display:"flex", gap:8, flexWrap:"wrap", marginTop:8}}>
      {platforms.map((p)=>(
        <span key={p} style={{
          padding:"2px 8px", borderRadius:999, fontSize:12,
          border:`1px solid ${colors[p]||"#334155"}`,
          color: colors[p]||"#334155"
        }}>{p}</span>
      ))}
    </div>
  );
}
