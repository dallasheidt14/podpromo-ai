/**
 * Debug logging utilities that only work in development
 */

const isDevelopment = process.env.NODE_ENV === 'development';

export const debugLog = (...args: any[]) => {
  if (isDevelopment) {
    console.log(...args);
  }
};

export const debugError = (...args: any[]) => {
  if (isDevelopment) {
    console.error(...args);
  }
};

export const debugWarn = (...args: any[]) => {
  if (isDevelopment) {
    console.warn(...args);
  }
};

export const debugInfo = (...args: any[]) => {
  if (isDevelopment) {
    console.info(...args);
  }
};

// For more specific debug categories
export const createDebugLogger = (category: string) => ({
  log: (...args: any[]) => debugLog(`[${category}]`, ...args),
  error: (...args: any[]) => debugError(`[${category}]`, ...args),
  warn: (...args: any[]) => debugWarn(`[${category}]`, ...args),
  info: (...args: any[]) => debugInfo(`[${category}]`, ...args),
});
