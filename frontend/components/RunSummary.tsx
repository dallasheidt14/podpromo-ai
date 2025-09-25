// components/RunSummary.tsx
interface RunSummaryProps {
  seeds: number;
  cand: number;
  kept: number;
  finals: number[];
  eos: number;
  dens: number; // e.g., 0.079
  mode: 'strict' | 'balanced';
  avgVirality?: number;      // 0–100
  avgPlatformFit?: number;   // 0–100
  platformProtect?: boolean;
  plV2Weight?: number;       // 0–1
}

export const RunSummary: React.FC<RunSummaryProps> = (p) => {
  const finalsStr = p.finals.map(x => `${Math.round(x * 10) / 10}s`).join(',');
  return (
    <div className="mb-3 rounded-lg bg-slate-50 px-3 py-2 text-[13px] text-slate-700">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-semibold">RUN_SUMMARY</span>
        <span>seeds={p.seeds}</span>
        <span>cand={p.cand}</span>
        <span>kept={p.kept}</span>
        <span>finals=[{finalsStr}]</span>
        <span>eos={p.eos}</span>
        <span>dens={p.dens.toFixed(3)}</span>
        <span>relax={p.mode}</span>
        {p.avgVirality != null && <span>avg_virality={Math.round(p.avgVirality)}</span>}
        {p.avgPlatformFit != null && <span>avg_platform_fit={Math.round(p.avgPlatformFit)}</span>}
        {p.plV2Weight != null && <span>pl_w={p.plV2Weight}</span>}
        {p.platformProtect != null && <span>protect={p.platformProtect ? 'on' : 'off'}</span>}
      </div>
    </div>
  );
};
