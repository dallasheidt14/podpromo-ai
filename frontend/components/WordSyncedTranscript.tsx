"use client";
import React, { useEffect, useMemo, useState } from "react";
import { getClipWords } from "@shared/api";
import { Clip } from "@shared/types";

type Props = {
  clip: Clip;
  mediaRef: React.RefObject<HTMLVideoElement | HTMLAudioElement>;
};

export default function WordSyncedTranscript({ clip, mediaRef }: Props) {
  const [words, setWords] = useState<{ t: number; d: number; w: string }[]>([]);
  const [now, setNow] = useState(0);

  useEffect(() => {
    let alive = true;
    getClipWords(clip.id).then((data) => {
      if (!alive) return;
      setWords(data.words || []);
    }).catch(() => {});
    const el = mediaRef.current;
    if (!el) return;
    const onTime = () => setNow(el.currentTime + (clip.start ?? 0));
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("seeked", onTime);
    return () => { alive = false; el?.removeEventListener("timeupdate", onTime); el?.removeEventListener("seeked", onTime); };
  }, [clip.id]);

  const content = useMemo(() => {
    return words.map((w, i) => {
      const active = now >= w.t && now <= w.t + w.d;
      return (
        <span key={i} className={active ? "bg-yellow-200 rounded px-0.5" : ""}>
          {w.w}{" "}
        </span>
      );
    });
  }, [words, now]);

  if (!words.length) {
    return (
      <div className="max-h-64 overflow-auto whitespace-pre-wrap leading-relaxed text-neutral-800 bg-neutral-50 p-3 rounded-lg">
        {clip.full_transcript || clip.text || "No transcript available."}
      </div>
    );
  }

  return (
    <div className="max-h-64 overflow-auto whitespace-pre-wrap leading-relaxed text-neutral-800 bg-neutral-50 p-3 rounded-lg">
      {content}
    </div>
  );
}