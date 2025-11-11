import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Load fonts
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

// --- Metadata (for SEO & PWA) ---
export const metadata: Metadata = {
  title: "PDF Q&A Chatbot",
  description:
    "Upload PDFs and ask AI-powered questions. Powered by OpenAI GPT and Whisper.",
  keywords: [
    "AI chatbots",
    "PDF Q&A",
    "OpenAI",
    "Whisper",
    "FastAPI",
    "Next.js",
  ],
  authors: [{ name: "Your Name" }],
  openGraph: {
    title: "PDF Q&A Chatbot",
    description:
      "Upload your PDFs and chat with AI to get instant answers from your documents.",
    url: "https://your-domain.com",
    siteName: "PDF Chatbot",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "PDF Chatbot preview",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  icons: {
    icon: "/favicon.ico",
  },
  themeColor: "#0f172a", // slate-900
};

// --- Root Layout ---
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-slate-50 text-slate-900`}
      >
        {children}
      </body>
    </html>
  );
}
