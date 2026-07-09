"use client";

import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { groupWordsIntoLines } from "@/lib/lineGrouping";

const BACKDROP_VARIANTS = [
  "radial-gradient(circle at 20% 20%, rgba(34,211,238,0.25), transparent 40%)",
  "radial-gradient(circle at 80% 20%, rgba(168,85,247,0.24), transparent 38%)",
  "radial-gradient(circle at 50% 80%, rgba(236,72,153,0.24), transparent 42%)",
];

function getLineStyle(lineId) {
  const seed = (lineId % 6) + 1;
  const positions = [
    { top: "8%", left: "8%", rotate: "-6deg" },
    { top: "18%", left: "58%", rotate: "5deg" },
    { top: "34%", left: "12%", rotate: "-4deg" },
    { top: "56%", left: "68%", rotate: "7deg" },
    { top: "72%", left: "20%", rotate: "-8deg" },
    { top: "78%", left: "54%", rotate: "4deg" },
  ];
  const pos = positions[(seed - 1) % positions.length];

  return {
    top: pos.top,
    left: pos.left,
    transform: `rotate(${pos.rotate})`,
  };
}

export default function MusicVideoLyrics({ words }) {
  const lines = useMemo(() => groupWordsIntoLines(words, { maxWordsPerLine: 9 }), [words]);
  const currentLine = lines.length ? lines[lines.length - 1] : null;

  return (
    <div className="relative flex h-full w-full overflow-hidden rounded-[28px] border border-white/10 bg-black/70 px-[4vw] py-[4vh]">
      <motion.div
        key={currentLine?.id ?? "idle"}
        className="absolute inset-0"
        animate={{
          backgroundImage: BACKDROP_VARIANTS[(currentLine?.id ?? 0) % BACKDROP_VARIANTS.length],
          opacity: [0.35, 0.7, 0.35],
        }}
        transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
      />

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.08),transparent_60%)]" />

      <AnimatePresence mode="wait">
        {currentLine && (
          <motion.div
            key={currentLine.id}
            initial={{ opacity: 0, scale: 0.94, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 1.04, y: -16 }}
            transition={{ duration: 0.5, ease: [0.2, 0.8, 0.3, 1] }}
            className="absolute inset-0"
          >
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.35 }}
              className="absolute flex max-w-[78%] flex-wrap items-center gap-x-[0.35em] gap-y-[0.25em] font-black uppercase leading-[0.95] tracking-[0.12em] text-cyan"
              style={{
                ...getLineStyle(currentLine.id),
                textShadow: "0 0 24px rgba(76,224,210,0.45)",
              }}
            >
              {currentLine.words.map((word, index) => {
                const size = 0.9 + (index % 4) * 0.18 + (currentLine.words.length > 6 ? 0.08 : 0);
                return (
                  <span
                    key={word.id}
                    style={{ fontSize: `clamp(${20 + size * 4}px, ${1.8 + size * 1.5}vw, ${34 + size * 14}px)` }}
                  >
                    {word.text}
                  </span>
                );
              })}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
