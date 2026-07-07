"use client";

import dynamic from "next/dynamic";

const LyricTranscriber = dynamic(
  () => import("../components/LyricTranscriber"),
  { ssr: false }
);

export default function Page() {
  return <LyricTranscriber />;
}
