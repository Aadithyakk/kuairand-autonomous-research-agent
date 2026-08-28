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
  title: 'KuaiRand Research Pilot',
  description: 'Autonomous recommender research, under evidence and budget control.',
  openGraph: {
    title: 'KuaiRand Research Pilot',
    description: 'Autonomous recommender research, under evidence and budget control.',
    url: 'https://kuairand-research-cockpit.hanrakkgungi.chatgpt.site',
    images: [{ url: 'https://kuairand-research-cockpit.hanrakkgungi.chatgpt.site/og.png', width: 1677, height: 941 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'KuaiRand Research Pilot',
    description: 'Autonomous recommender research, under evidence and budget control.',
    images: ['https://kuairand-research-cockpit.hanrakkgungi.chatgpt.site/og.png'],
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
