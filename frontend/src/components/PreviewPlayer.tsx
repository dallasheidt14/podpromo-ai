"use client";
import { useEffect, useState } from "react";
import { getSignedPreviewUrl } from "../shared/api";

interface PreviewPlayerProps {
  clip: {
    preview_name?: string;
    previewUrl?: string; // fallback for backwards compatibility
    preview_url?: string; // fallback for backwards compatibility
  };
  className?: string;
}

export function PreviewPlayer({ clip, className = "" }: PreviewPlayerProps) {
  const [src, setSrc] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    
    const loadPreview = async () => {
      // Try preview_name first (new secure approach)
      if (clip.preview_name) {
        try {
          const signedUrl = await getSignedPreviewUrl(clip.preview_name);
          if (alive) {
            setSrc(signedUrl);
            setError(null);
          }
        } catch (e) {
          console.error("Failed to get signed preview URL:", e);
          if (alive) {
            setError("Failed to load preview");
          }
        }
        return;
      }

      // Fallback to old preview URLs for backwards compatibility
      const fallbackUrl = clip.previewUrl || clip.preview_url;
      if (fallbackUrl) {
        if (alive) {
          setSrc(fallbackUrl);
          setError(null);
        }
      } else if (alive) {
        setError("No preview available");
      }
    };

    loadPreview();
    
    return () => {
      alive = false;
    };
  }, [clip.preview_name, clip.previewUrl, clip.preview_url]);

  if (error) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`}>
        <div className="text-center text-gray-500">
          <div className="text-sm">{error}</div>
        </div>
      </div>
    );
  }

  if (!src) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`}>
        <div className="text-center text-gray-500">
          <div className="text-sm">Loading preview...</div>
        </div>
      </div>
    );
  }

  return (
    <video
      controls
      playsInline
      preload="metadata"
      src={src}
      className={`w-full rounded-lg ${className}`}
      onError={(e) => {
        console.warn("Video preview failed to load:", e);
        setError("Preview failed to load");
      }}
      style={{ maxHeight: "400px" }}
    />
  );
}
