import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000'),
  title: 'KuaiLab — Agent Research Replay',
  description: 'Follow the evidence behind every recommendation experiment: recorded research, validation gates, and inspectable decisions.',
  openGraph: {
    title: 'KuaiLab — Agent Research Replay',
    description: 'Evidence behind every decision. Replay recorded experiments or inspect the live research control room.',
    type: 'website',
    images: [{ url: '/og.png', width: 1731, height: 909, alt: 'KuaiLab. Agent Research Replay. Evidence behind every decision.' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'KuaiLab — Agent Research Replay',
    description: 'Evidence behind every decision.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
