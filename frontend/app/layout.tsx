import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Highlightly AI - Generate Engaging Podcast Clips',
  description: 'AI-powered podcast clip generation service. Upload your episode and get 3-5 engaging social media clips in under 5 minutes.',
  keywords: 'podcast, clips, AI, social media, video generation, content creation',
  authors: [{ name: 'Highlightly AI Team' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-neutral-900 antialiased">
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
