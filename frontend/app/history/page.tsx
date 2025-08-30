"use client";
import { useEffect, useState } from "react";
import { API_URL } from "../lib/config";
import ShareSheet from "../components/ShareSheet";

type Item = {
  ts: string;
  clip_id: string;
  format: "vertical" | "square" | string;
  output: string;  // filename under /clips
  duration?: number;
  style?: string;
  captions?: boolean;
  punch_ins?: boolean;
  loop_seam?: boolean;
};

export default function HistoryPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/history?limit=100`)
      .then(r => r.json())
      .then(d => {
        if (d?.ok) setItems(d.items || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div style={{padding: 16, textAlign: "center"}}>
        <h1>Loading Clip History...</h1>
      </div>
    );
  }

  return (
    <div style={{padding: 16, maxWidth: 1200, margin: "0 auto"}}>
      <div style={{marginBottom: 24}}>
        <h1 style={{margin: "0 0 8px 0", fontSize: "32px", fontWeight: "bold"}}>
          📹 Rendered Clips
        </h1>
        <p style={{margin: 0, color: "#666", fontSize: "16px"}}>
          Your complete clip creation history with {items.length} clips
        </p>
      </div>

      {items.length === 0 ? (
        <div style={{
          textAlign: "center", 
          padding: "48px 24px", 
          backgroundColor: "#f8f9fa", 
          borderRadius: 8,
          border: "1px solid #e9ecef"
        }}>
          <h3 style={{margin: "0 0 16px 0", color: "#6c757d"}}>No clips yet</h3>
          <p style={{margin: 0, color: "#6c757d"}}>
            Upload a podcast episode and render some clips to see them here!
          </p>
        </div>
      ) : (
        <div style={{
          display: "grid", 
          gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))", 
          gap: 20
        }}>
          {items.map((it, idx) => (
            <div key={idx} style={{
              border: "1px solid #e5e7eb", 
              borderRadius: 12, 
              padding: 16,
              backgroundColor: "white",
              boxShadow: "0 1px 3px rgba(0,0,0,0.1)"
            }}>
              {/* Header with timestamp and format */}
              <div style={{
                display: "flex", 
                justifyContent: "space-between", 
                alignItems: "center",
                marginBottom: 12
              }}>
                <div style={{fontSize: "12px", color: "#6b7280"}}>
                  {new Date(it.ts).toLocaleString()}
                </div>
                <div style={{
                  padding: "4px 8px",
                  backgroundColor: it.format === "vertical" ? "#dbeafe" : "#fef3c7",
                  color: it.format === "vertical" ? "#1e40af" : "#92400e",
                  borderRadius: 6,
                  fontSize: "11px",
                  fontWeight: "bold",
                  textTransform: "uppercase"
                }}>
                  {it.format}
                </div>
              </div>

              {/* Duration and style info */}
              <div style={{
                display: "flex", 
                justifyContent: "space-between", 
                alignItems: "center",
                marginBottom: 16
              }}>
                <span style={{
                  fontSize: "14px", 
                  color: "#374151",
                  fontWeight: "500"
                }}>
                  {it.duration ? `${it.duration.toFixed(1)}s` : "Unknown duration"}
                </span>
                {it.style && (
                  <span style={{
                    fontSize: "12px",
                    color: "#6b7280",
                    fontStyle: "italic"
                  }}>
                    Style: {it.style}
                  </span>
                )}
              </div>

              {/* Feature badges */}
              <div style={{
                display: "flex", 
                gap: 6, 
                marginBottom: 16, 
                flexWrap: "wrap"
              }}>
                {it.captions && (
                  <span style={{
                    backgroundColor: "#d1fae5",
                    color: "#065f46",
                    padding: "2px 6px",
                    borderRadius: 4,
                    fontSize: "10px",
                    fontWeight: "bold"
                  }}>Captions</span>
                )}
                {it.punch_ins && (
                  <span style={{
                    backgroundColor: "#dbeafe",
                    color: "#1e40af",
                    padding: "2px 6px",
                    borderRadius: 4,
                    fontSize: "10px",
                    fontWeight: "bold"
                  }}>Punch-ins</span>
                )}
                {it.loop_seam && (
                  <span style={{
                    backgroundColor: "#fef3c7",
                    color: "#92400e",
                    padding: "2px 6px",
                    borderRadius: 4,
                    fontSize: "10px",
                    fontWeight: "bold"
                  }}>Loop Seam</span>
                )}
              </div>

              {/* Video preview */}
              <video 
                controls 
                width="100%" 
                style={{
                  borderRadius: 8, 
                  marginBottom: 16,
                  backgroundColor: "#f3f4f6"
                }}
                src={`${API_URL}/clips/${it.output}`}
              />

              {/* Share actions */}
              <div style={{marginBottom: 12}}>
                <ShareSheet 
                  url={`${API_URL}/clips/${it.output}`} 
                  filename={it.output}
                />
              </div>

              {/* Clip ID for reference */}
              <div style={{
                fontSize: "10px", 
                color: "#9ca3af", 
                textAlign: "center",
                fontFamily: "monospace"
              }}>
                ID: {it.clip_id}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Navigation back to main app */}
      <div style={{
        marginTop: 32,
        textAlign: "center",
        padding: "16px",
        borderTop: "1px solid #e5e7eb"
      }}>
        <a 
          href="/"
          style={{
            padding: "8px 16px",
            backgroundColor: "#6b7280",
            color: "white",
            textDecoration: "none",
            borderRadius: 6,
            fontSize: "14px"
          }}
        >
          ← Back to PodPromo AI
        </a>
      </div>
    </div>
  );
}
