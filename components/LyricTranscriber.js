"use client";

import { useState } from "react";
import { useLiveTranscription } from "@/hooks/useLiveTranscription";
import VuMeter from "./VuMeter";
import IdleHint from "./IdleHint";
import { DISPLAY_MODES } from "./display-modes";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

const MODEL_OPTIONS = [
  { value: "onnx-community/whisper-tiny.en_timestamped", label: "Whisper Tiny (fastest)" },
  { value: "onnx-community/whisper-base.en_timestamped", label: "Whisper Base (balanced)" },
  { value: "onnx-community/whisper-small.en_timestamped", label: "Whisper Small (most accurate)" },
];

export default function LyricTranscriber() {
  const { status, isLive, isBusy, isListening, words, modelProgress, analyserNodeRef, start, stop } =
    useLiveTranscription();

  const [sourceType, setSourceType] = useState("mic");
  const [modelId, setModelId] = useState(MODEL_OPTIONS[1].value);
  const [modeId, setModeId] = useState(DISPLAY_MODES[0].id);

  const ActiveMode = DISPLAY_MODES.find((m) => m.id === modeId)?.Component ?? DISPLAY_MODES[0].Component;
  const showIdleHint = words.length === 0 && !isListening;

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-panel-line px-[22px] py-3.5">
        <div className="flex items-center gap-2.5 text-[15px] font-bold tracking-wide">
          <span
            className={cn(
              "h-[9px] w-[9px] rounded-full transition-all duration-200",
              isLive ? "animate-rec-pulse bg-rec shadow-[0_0_10px_2px_rgba(255,92,92,0.6)]" : "bg-dim"
            )}
          />
          <span>SIGNAL</span>
        </div>
        <div className="text-center font-mono text-[11px] tracking-wide text-mid">{status}</div>
        <VuMeter analyserNodeRef={analyserNodeRef} isListening={isListening} />
      </header>

      <main className="relative flex-1 overflow-hidden px-[8vw] py-[4vh] [mask-image:linear-gradient(to_bottom,transparent_0%,black_14%,black_100%)]">
        {showIdleHint ? (
          <div className="flex h-full items-center justify-center">
            <IdleHint />
          </div>
        ) : (
          <ActiveMode words={words} isListening={isListening} />
        )}
      </main>

      <footer className="flex shrink-0 flex-wrap items-center gap-3.5 border-t border-panel-line bg-panel px-[22px] py-3.5">
        <label className="flex flex-col gap-1 text-[10.5px] tracking-wide text-dim">
          SOURCE
          <Select value={sourceType} onValueChange={setSourceType} disabled={isBusy}>
            <SelectTrigger className="w-[160px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mic">Microphone</SelectItem>
              <SelectItem value="system">System / tab audio</SelectItem>
            </SelectContent>
          </Select>
        </label>

        <label className="flex flex-col gap-1 text-[10.5px] tracking-wide text-dim">
          MODEL
          <Select value={modelId} onValueChange={setModelId} disabled={isBusy}>
            <SelectTrigger className="w-[200px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MODEL_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        <label className="flex flex-col gap-1 text-[10.5px] tracking-wide text-dim">
          DISPLAY
          <Tabs value={modeId} onValueChange={setModeId}>
            <TabsList>
              {DISPLAY_MODES.map((mode) => (
                <TabsTrigger key={mode.id} value={mode.id}>
                  {mode.label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </label>

        <Button onClick={() => start(sourceType, modelId)} disabled={isBusy}>
          ▶ Start
        </Button>
        <Button variant="destructive" onClick={stop} disabled={!isListening}>
          ■ Stop
        </Button>

        <div className="flex-1" />

        {modelProgress && (
          <div className="flex items-center gap-2 font-mono text-[11.5px] text-mid">
            <span>{modelProgress.label}</span>
            <Progress value={modelProgress.percent} className="w-[110px]" />
          </div>
        )}
      </footer>
    </div>
  );
}
