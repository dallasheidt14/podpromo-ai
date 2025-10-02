// components/ContextBadges.tsx
import type { Clip } from '../types/Clip';

interface PayoffBadgeProps {
  type?: string;
}

export function PayoffBadge({ type }: PayoffBadgeProps) {
  if (!type) return null;
  
  const getLabel = (t: string) => {
    switch (t) {
      case 'conclusion': return 'Conclusion';
      case 'no_clear_payoff': return 'No Payoff';
      case 'unknown': return 'Unknown';
      default: return t;
    }
  };
  
  const getColor = (t: string) => {
    switch (t) {
      case 'conclusion': return 'bg-green-100 text-green-800';
      case 'no_clear_payoff': return 'bg-red-100 text-red-800';
      case 'unknown': return 'bg-gray-100 text-gray-800';
      default: return 'bg-neutral-200 text-neutral-800';
    }
  };
  
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs ${getColor(type)}`}>
      {getLabel(type)}
    </span>
  );
}

interface SeedBadgeProps {
  seedIdx?: number | null;
}

export function SeedBadge({ seedIdx }: SeedBadgeProps) {
  if (seedIdx == null) return null;
  
  return (
    <span className="px-2 py-0.5 rounded-full text-xs bg-emerald-100 text-emerald-800">
      Seeded
    </span>
  );
}

interface PunctuationBadgeProps {
  text?: string;
}

export function PunctuationBadge({ text }: PunctuationBadgeProps) {
  if (!text || !/\.|!|\?$/.test(text.trim())) return null;
  
  return (
    <span 
      className="px-1.5 py-0.5 rounded text-[10px] bg-blue-100 text-blue-800 ml-2" 
      title="Ends on punctuation"
    >
      •
    </span>
  );
}

interface ContextBadgesProps {
  clip: Clip;
}

export function ContextBadges({ clip }: ContextBadgesProps) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <PayoffBadge type={clip.features?.payoff_type} />
      <SeedBadge seedIdx={clip.seed_idx} />
      <PunctuationBadge text={clip.text} />
    </div>
  );
}
