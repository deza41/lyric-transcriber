"use client";

import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

const VISIBLE_COUNT = 7;

// Current word large and bright in the center; recent words shrink and
// fade outward to its left as a trail. Size is expressed purely through
// `scale` on a shared base font size (not a conditional fontSize) so it
// animates smoothly instead of snapping when a word stops being current.
export default function CenteredSpotlight({ words }) {
  const recent = useMemo(() => words.slice(-VISIBLE_COUNT), [words]);

  return (
    <div className="flex h-full w-full items-center justify-center gap-3 px-[6vw]">
      <AnimatePresence mode="popLayout">
        {recent.map((w, i) => {
          const distanceFromCurrent = recent.length - 1 - i;
          const isCurrent = distanceFromCurrent === 0;
          const scale = isCurrent ? 1.3 : Math.max(0.35, 0.9 - distanceFromCurrent * 0.15);
          const opacity = Math.max(0.15, 1 - distanceFromCurrent * 0.16);
          return (
            <motion.span
              key={w.id}
              layout
              initial={{ opacity: 0, scale: 0.6, y: 10 }}
              animate={{
                opacity,
                scale,
                y: 0,
                color: isCurrent ? "var(--color-cyan)" : "var(--color-dim)",
                textShadow: isCurrent ? "0 0 28px rgba(76,224,210,0.45)" : "0 0 0px rgba(76,224,210,0)",
              }}
              exit={{ opacity: 0, scale: 0.3 }}
              transition={{
                scale: { type: "spring", stiffness: 260, damping: 24 },
                y: { type: "spring", stiffness: 260, damping: 24 },
                opacity: { duration: 0.3 },
                color: { duration: isCurrent ? 0.05 : 1.1 },
                textShadow: { duration: isCurrent ? 0.05 : 1.1 },
              }}
              className="whitespace-pre font-bold"
              style={{ fontSize: "clamp(26px, 4.5vw, 48px)" }}
            >
              {w.text}
            </motion.span>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
