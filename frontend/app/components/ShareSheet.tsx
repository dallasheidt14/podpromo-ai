"use client";
import { useState } from "react";

export default function ShareSheet({ url, filename }: { url: string; filename?: string }) {
  const [copied, setCopied] = useState(false);

  async function share() {
    try {
      if (navigator.share) {
        await navigator.share({ title: "PodPromo AI Clip", url });
        return;
      }
    } catch {}
    await copy();
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  }

  return (
    <div style={{display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap"}}>
      <button 
        onClick={share}
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
        Share
      </button>
      <a 
        href={url} 
        download={filename || "clip.mp4"}
        style={{textDecoration: "none"}}
      >
        <button
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
          Download
        </button>
      </a>
      <button 
        onClick={copy}
        style={{
          padding: "6px 12px",
          backgroundColor: copied ? "#6c757d" : "#ffc107",
          color: copied ? "white" : "black",
          border: "none",
          borderRadius: 4,
          cursor: "pointer",
          fontSize: "14px"
        }}
      >
        {copied ? "Copied!" : "Copy Link"}
      </button>
    </div>
  );
}
