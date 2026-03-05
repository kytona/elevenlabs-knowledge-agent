import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ElevenLabs Knowledge Agent",
  description: "Starter kit for connecting ElevenLabs Conversational AI to a custom RAG backend.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

