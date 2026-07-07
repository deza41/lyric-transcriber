"use client";

import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { groupWordsIntoLines } from "@/lib/lineGrouping";

const VISIBLE_LINES = 8;

// Several lines visible at once; Framer's layout/FLIP animation on each
// line block handles the "push up" automatically as new lines enter the
// bottom and old ones fall off the top of the visible slice.
export default function Teleprompter({ words }) {
  const lines = useMemo(() => groupWordsIntoLines(words, { maxWordsPerLine: 9 }), [words]);
  const visibleLines = useMemo(() => lines.slice(-VISIBLE_LINES), [lines]);
  const currentId = visibleLines.length ? visibleLines[visibleLines.length - 1].id : null;

  return (
    <div className="mx-auto flex h-full w-full max-w-[900px] flex-col justify-end gap-2 overflow-hidden px-[4vw]">
      <AnimatePresence mode="popLayout" initial={false}>
        {visibleLines.map((line) => {
          const isCurrent = line.id === currentId;
          return (
            <motion.div
              key={line.id}
              layout
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: isCurrent ? 1 : 0.55, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{
                layout: { duration: 0.4, ease: "easeOut" },
                opacity: { duration: 0.3 },
                y: { duration: 0.3 },
              }}
              className="flex flex-wrap gap-x-2 font-semibold"
              style={{ fontSize: "clamp(20px, 3.4vw, 34px)" }}
            >
              {line.words.map((w) => (
                <span key={w.id} className={isCurrent ? "text-bright" : "text-mid"}>
                  {w.text}
                </span>
              ))}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
