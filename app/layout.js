import "./globals.css";

export const metadata = {
  title: "Signal — Live Lyric Transcriber",
  description:
    "Turns your mic or system audio into a live karaoke-style transcript, fully local — no cloud API, no subscription. Powered by Whisper via transformers.js.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
