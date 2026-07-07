"use client";

import { useEffect, useRef } from "react";

const VU_BARS = 20;

// Reads raw analyser bytes and writes bar heights/colors directly to the
// DOM via refs at ~60fps — deliberately not React state. This runs far
// faster than the ~2s cadence everything else in the app updates at, and
// re-rendering 20 elements every frame would be pure waste for a effect
// that's purely decorative.
export default function VuMeter({ analyserNodeRef, isListening }) {
  const vuBarsRef = useRef([]);
  const rafIdRef = useRef(null);

  useEffect(() => {
    if (!isListening || !analyserNodeRef.current) return;

    const analyserNode = analyserNodeRef.current;
    const data = new Uint8Array(analyserNode.fftSize);

    const step = () => {
      analyserNode.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const level = Math.sqrt(sum / data.length);
      const activeBars = Math.round(Math.min(1, level * 6) * VU_BARS);
      vuBarsRef.current.forEach((bar, i) => {
        if (!bar) return;
        const on = i < activeBars;
        bar.style.height = on ? `${6 + (i / VU_BARS) * 16}px` : "3px";
        bar.style.background = on ? (i > VU_BARS * 0.75 ? "#FF5C5C" : "#FFB238") : "var(--amber-dim)";
      });
      rafIdRef.current = requestAnimationFrame(step);
    };
    step();

    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
      vuBarsRef.current.forEach((bar) => {
        if (!bar) return;
        bar.style.height = "3px";
        bar.style.background = "var(--amber-dim)";
      });
    };
  }, [isListening, analyserNodeRef]);

  return (
    <div className="flex h-[22px] items-end gap-[3px]">
      {Array.from({ length: VU_BARS }).map((_, i) => (
        <span
          key={i}
          ref={(el) => (vuBarsRef.current[i] = el)}
          className="block h-[3px] w-[3px] rounded-[1px] bg-amber-dim transition-[height,background] duration-[60ms] ease-linear"
        />
      ))}
    </div>
  );
}
