import useSWR from 'swr';
import { getJson, apiUrl } from '@shared/api';
import type { Progress, Episode, Clip } from '@shared/types';

// Fetcher function for SWR
const fetcher = async (url: string) => {
  const result = await getJson(url);
  if (result.success) {
    return result.data;
  }
  throw new Error(result.error || 'Failed to fetch data');
};

// Auth-aware fetcher for protected endpoints
const authFetcher = async (url: string) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : '';
  const result = await getJson(url, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Cache-Control': 'no-store'
    }
  });
  if (result.success) {
    return result.data;
  }
  throw new Error(result.error || 'Failed to fetch data');
};

// Hook for episode progress polling
export function useEpisodeProgress(episodeId: string | undefined) {
  const { data, error, isLoading, mutate } = useSWR(
    episodeId ? apiUrl(`/api/progress/${episodeId}`) : null,
    fetcher,
    {
      refreshInterval: 2000, // Poll every 2 seconds
      revalidateOnFocus: false,
      revalidateOnReconnect: true,
      dedupingInterval: 1000, // Dedupe requests within 1 second
    }
  );

  return {
    progress: data as Progress | undefined,
    error,
    isLoading,
    refresh: mutate,
  };
}

// Hook for episode details
export function useEpisode(episodeId: string | undefined) {
  const { data, error, isLoading, mutate } = useSWR(
    episodeId ? apiUrl(`/api/episodes/${episodeId}`) : null,
    fetcher,
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
    }
  );

  return {
    episode: data as Episode | undefined,
    error,
    isLoading,
    refresh: mutate,
  };
}

// Hook for episode clips
export function useEpisodeClips(episodeId: string | undefined) {
  const { data, error, isLoading, mutate } = useSWR(
    episodeId ? apiUrl(`/api/episodes/${episodeId}/clips`) : null,
    fetcher,
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
    }
  );

  return {
    clips: data as Clip[] | undefined,
    error,
    isLoading,
    refresh: mutate,
  };
}

// Hook for signed download URLs
export function useSignedDownload(area: "uploads" | "clips", name: string | undefined, ttl: number = 300) {
  const { data, error, isLoading, mutate } = useSWR(
    name ? apiUrl(`/api/signed-download?area=${area}&name=${encodeURIComponent(name)}&ttl=${ttl}`) : null,
    authFetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      dedupingInterval: 60000, // Cache for 1 minute
    }
  );

  return {
    signedUrl: data?.url as string | undefined,
    expiresAt: data?.expires_at as number | undefined,
    error,
    isLoading,
    refresh: mutate,
  };
}

// Hook for app configuration
export function useAppConfig() {
  const { data, error, isLoading, mutate } = useSWR(
    apiUrl('/api/config'),
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: true,
      dedupingInterval: 300000, // Cache for 5 minutes
    }
  );

  return {
    config: data,
    error,
    isLoading,
    refresh: mutate,
  };
}
