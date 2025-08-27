import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'PodPromo AI - Generate Engaging Podcast Clips',
  description: 'AI-powered podcast clip generation service. Upload your episode and get 3-5 engaging social media clips in under 5 minutes.',
  keywords: 'podcast, clips, AI, social media, video generation, content creation',
  authors: [{ name: 'PodPromo AI Team' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 text-gray-900 antialiased`}>
        <div className="min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
