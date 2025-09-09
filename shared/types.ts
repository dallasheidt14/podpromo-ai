export interface Episode {
  id: string;
  filename?: string;
  originalName?: string;
  size?: number;
  duration?: number;
  status?: string;
  uploadedAt?: string;
}

export interface Clip {
  id: string;
  start: number;
  end: number;
  duration: number;
  score?: number;
  url?: string;
  transcript?: string;
  is_advertisement?: boolean;
  features?: Record<string, any>;
  [key: string]: any;
}

export interface ProgressInfo {
  stage: string;
  percentage: number;
  eta?: number | null;
  [key: string]: any;
}