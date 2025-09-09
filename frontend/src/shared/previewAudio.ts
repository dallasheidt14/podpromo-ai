// src/shared/previewAudio.ts
// Audio preview manager to prevent overlapping previews

let audio: HTMLAudioElement | null = null;

export function playPreview(url: string): Promise<void> | undefined {
  try {
    if (!audio) audio = new Audio();
    if (!url) {
      stopPreview();
      return undefined;
    }
    
    // Stop any currently playing audio
    audio.pause();
    audio.src = url;
    audio.currentTime = 0;
    
    // Return the play promise for error handling
    return audio.play();
  } catch (e) {
    console.warn("[preview] failed to play:", e);
    return undefined;
  }
}

export function stopPreview(): void {
  if (audio) { 
    audio.pause(); 
    audio.src = ""; 
  }
}

export function isPlaying(): boolean {
  return audio ? !audio.paused : false;
}

export function getCurrentSrc(): string | null {
  return audio ? audio.src : null;
}
