/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const target = process.env.BACKEND_ORIGIN || 'http://localhost:8000';
    return [
      { source: '/api/:path*', destination: `${target}/api/:path*` },
      { source: '/health', destination: `${target}/health` },
    ];
  },
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'backend', port: '8000' },
      { protocol: 'http', hostname: 'localhost', port: '8000' },
      // add prod hosts when ready:
      // { protocol: 'https', hostname: 'app.yourdomain.com' },
      // { protocol: 'https', hostname: 'cdn.yourdomain.com' },
    ],
  },
};

module.exports = nextConfig;
