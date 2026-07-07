"use client";

import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { groupWordsIntoLines } from "@/lib/lineGrouping";

// One full line at a time, large and centered, crossfading to the next
// line on a sentence break or word-count cap. There's no lookahead — only
// what's actually been transcribed renders, nothing pre-fills ahead of it.
export default function MusicVideoLyrics({ words }) {
  const lines = useMemo(() => groupWordsIntoLines(words, { maxWordsPerLine: 9 }), [words]);
  const currentLine = lines.length ? lines[lines.length - 1] : null;

  return (
    <div className="flex h-full w-full items-center justify-center px-[6vw] text-center">
      <AnimatePresence>
        {currentLine && (
          <motion.div
            key={currentLine.id}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -24 }}
            transition={{ duration: 0.45, ease: [0.2, 0.8, 0.3, 1] }}
            className="flex flex-wrap justify-center gap-x-[0.4em] gap-y-2 font-bold leading-snug"
            style={{ fontSize: "clamp(28px, 6vw, 64px)" }}
          >
            {currentLine.words.map((w) => (
              <span key={w.id} className="text-cyan" style={{ textShadow: "0 0 26px rgba(76,224,210,0.4)" }}>
                {w.text}
              </span>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
