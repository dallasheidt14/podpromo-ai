import * as React from "react";

type Clip = { 
  id: string; 
  start: number; 
  end: number; 
  transcript?: string;
};

export function useClipTranscript(clip: Clip) {
  // Priority order: exact transcript -> full_text -> hook text
  const transcript = 
    clip?.transcript?.trim() ??
    clip?.full_text?.trim() ??
    clip?.text?.trim() ??
    "";
    
  const [text, setText] = React.useState<string>(transcript);
  const [loading, setLoading] = React.useState<boolean>(!transcript);
  const [error, setError] = React.useState<string>("");

  React.useEffect(() => {
    if (transcript) return;
    
    let alive = true;
    setLoading(true);
    
    fetch(`/api/clips/${clip.id}/transcript`)
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))
      .then(d => { 
        if (alive) setText(d.transcript || ""); 
      })
      .catch(e => { 
        if (alive) setError(e.message || "Failed to load transcript"); 
      })
      .finally(() => { 
        if (alive) setLoading(false); 
      });
      
    return () => { alive = false; };
  }, [clip.id, clip.transcript]);

  return { text, loading, error };
}
