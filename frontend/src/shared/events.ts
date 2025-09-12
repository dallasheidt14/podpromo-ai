// Global event for "clips for episodeId are now ready"
export type ClipsReadyDetail = { episodeId: string };
const EVT = "clips-ready";

export function notifyClipsReady(episodeId: string) {
  window.dispatchEvent(new CustomEvent<ClipsReadyDetail>(EVT, { detail: { episodeId } }));
}

export function onClipsReady(handler: (episodeId: string) => void) {
  const listener = (e: Event) => {
    const ce = e as CustomEvent<ClipsReadyDetail>;
    if (ce?.detail?.episodeId) handler(ce.detail.episodeId);
  };
  window.addEventListener(EVT, listener);
  return () => window.removeEventListener(EVT, listener);
}
