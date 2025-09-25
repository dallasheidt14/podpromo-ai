// ui/MiniBar.tsx
export const MiniBar: React.FC<{ label: string; pct: number; title?: string }> = ({ label, pct, title }) => {
  const safe = Math.max(0, Math.min(100, Math.round(pct)));
  return (
    <div className="space-y-1" title={title}>
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span className="tabular-nums">{safe}%</span>
      </div>
      <div className="h-1.5 w-full rounded bg-slate-200">
        <div className="h-1.5 rounded bg-slate-800" style={{ width: `${safe}%` }} />
      </div>
    </div>
  );
};
