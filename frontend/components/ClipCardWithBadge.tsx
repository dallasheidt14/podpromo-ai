// components/ClipCardWithBadge.tsx
import { ClipCard } from './ClipCard';
import { BestBadge } from '../ui/BestBadge';
import { getViralityPct } from '../ui/score';
import type { Clip } from '../types/Clip';

interface ClipCardWithBadgeProps {
  clip: Clip;
  clips: Clip[]; // all clips to determine if this is the best
  onClick?: () => void;
}

export const ClipCardWithBadge: React.FC<ClipCardWithBadgeProps> = ({ clip, clips, onClick }) => {
  const isBest = clips.length > 0 && getViralityPct(clip) === Math.max(...clips.map(c => getViralityPct(c)));
  
  return (
    <div className="relative">
      {isBest && <BestBadge />}
      <ClipCard clip={clip} onClick={onClick} />
    </div>
  );
};
