"use client";

import { useEffect, useMemo, useRef } from "react";
import { motion } from "framer-motion";

const MAX_VISIBLE = 220;

// Same wrapped layout as Classic Karaoke, but each word pops in with an
// underdamped spring instead of a scripted tween — a visibly bouncier,
// more playful entrance.
export default function WaveBounce({ words }) {
  const scrollRef = useRef(null);
  const visible = useMemo(() => words.slice(-MAX_VISIBLE), [words]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visible.length]);

  return (
    <div
      ref={scrollRef}
      className="mx-auto flex h-full w-full max-w-[900px] flex-col justify-end overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
    >
      <div className="flex flex-wrap content-end gap-x-[0.55em] gap-y-[0.35em] leading-[1.5]">
        {visible.map((w, i) => {
          const isCurrent = i === visible.length - 1;
          return (
            <motion.span
              key={w.id}
              layout
              initial={{ opacity: 0, y: 30, scale: 0.5, rotate: -6 }}
              animate={{
                opacity: 1,
                y: 0,
                scale: 1,
                rotate: 0,
                color: isCurrent ? "var(--color-cyan)" : "var(--color-dim)",
                textShadow: isCurrent ? "0 0 22px rgba(76,224,210,0.45)" : "0 0 0px rgba(76,224,210,0)",
              }}
              transition={{
                y: { type: "spring", stiffness: 500, damping: 15, mass: 0.6 },
                scale: { type: "spring", stiffness: 500, damping: 15, mass: 0.6 },
                rotate: { type: "spring", stiffness: 500, damping: 15, mass: 0.6 },
                opacity: { duration: 0.2 },
                color: { duration: isCurrent ? 0.05 : 1.1 },
                textShadow: { duration: isCurrent ? 0.05 : 1.1 },
              }}
              className="whitespace-pre font-semibold"
              style={{ fontSize: "clamp(22px, 4.6vw, 44px)" }}
            >
              {w.text}
            </motion.span>
          );
        })}
      </div>
    </div>
  );
}
