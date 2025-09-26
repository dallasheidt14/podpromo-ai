// frontend/app/lib/config.ts
export const API_URL =
  process.env.NEXT_PUBLIC_API_BASE ||
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const config = { API_URL };
export default config;