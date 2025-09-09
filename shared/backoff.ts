// Reusable backoff utility for retry logic with exponential backoff and jitter

export interface BackoffOptions {
  tries?: number;
  baseMs?: number;
  maxMs?: number;
  jitterMs?: number;
}

export async function withBackoff<T>(
  fn: () => Promise<T>, 
  opts: BackoffOptions = {}
): Promise<T> {
  const {
    tries = 3,
    baseMs = 600,
    maxMs = 10000,
    jitterMs = 150
  } = opts;

  let lastError: unknown;
  
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      // Don't wait after the last attempt
      if (i === tries - 1) {
        break;
      }
      
      // Calculate delay with exponential backoff and jitter
      const exponentialDelay = baseMs * Math.pow(2, i);
      const cappedDelay = Math.min(exponentialDelay, maxMs);
      const jitter = Math.random() * jitterMs;
      const totalDelay = cappedDelay + jitter;
      
      await new Promise(resolve => setTimeout(resolve, totalDelay));
    }
  }
  
  throw lastError;
}

// Convenience function for fetch requests
export async function fetchWithBackoff(
  url: string, 
  options: RequestInit = {}, 
  backoffOpts: BackoffOptions = {}
): Promise<Response> {
  return withBackoff(async () => {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response;
  }, backoffOpts);
}

// Convenience function for JSON fetch requests
export async function fetchJsonWithBackoff<T = any>(
  url: string, 
  options: RequestInit = {}, 
  backoffOpts: BackoffOptions = {}
): Promise<T> {
  const response = await fetchWithBackoff(url, options, backoffOpts);
  return response.json();
}
