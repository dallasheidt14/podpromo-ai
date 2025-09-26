// app/lib/metrics.ts
export const toPct = (v?: number) => Math.round(((v ?? 0) as number) * 100);
