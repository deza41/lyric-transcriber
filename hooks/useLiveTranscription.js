"use client";

import { useEffect, useRef, useState } from "react";

const TARGET_SR = 16000;
// Sliding window instead of disjoint chunks: each inference call sees
// OVERLAP_SECONDS of audio it has already transcribed plus NEW_AUDIO_SECONDS
// of fresh audio. The overlap gives Whisper real context for words that
// start right at the window boundary (a hard cut mid-word/mid-sentence is a
// major accuracy killer for chunked streaming), and re-transcribed
// overlapping words get deduplicated against what's already on screen. This
// also means an update lands every NEW_AUDIO_SECONDS instead of waiting for
// a much longer chunk to fill.
const NEW_AUDIO_SECONDS = 2;
const OVERLAP_SECONDS = 1;
const WINDOW_SAMPLES = TARGET_SR * (NEW_AUDIO_SECONDS + OVERLAP_SECONDS);
const NEW_AUDIO_SAMPLES = TARGET_SR * NEW_AUDIO_SECONDS;
const OVERLAP_SAMPLES = TARGET_SR * OVERLAP_SECONDS;
const SILENCE_RMS_THRESHOLD = 0.006;
const TRANSFORMERS_CDN_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const MAX_WORDS = 3000;

export function useLiveTranscription() {
  const [status, setStatusText] = useState("idle — pick a source and press start");
  const [isLive, setIsLive] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [words, setWords] = useState([]);
  const [modelProgress, setModelProgress] = useState(null);

  // Mutable state that doesn't need to trigger re-renders
  const asrPipelineRef = useRef(null);
  const loadedModelIdRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const analyserNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const muteGainRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const runningRef = useRef(false);
  const nativeSampleRateRef = useRef(48000);

  const pcmBufferRef = useRef([]);
  const pcmBufferLenRef = useRef(0);

  const chunkQueueRef = useRef([]);
  const processingChunkRef = useRef(false);

  // Last few words actually displayed, used to detect how much of a new
  // window's transcript duplicates what the overlap region already showed.
  const recentWordsRef = useRef([]);
  const wordIdRef = useRef(0);

  function setStatus(text, live) {
    setStatusText(text);
    setIsLive(!!live);
  }

  function downsampleTo16k(float32, inputRate) {
    if (inputRate === TARGET_SR) return float32;
    const ratio = inputRate / TARGET_SR;
    const outLength = Math.round(float32.length / ratio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const srcPos = i * ratio;
      const i0 = Math.floor(srcPos);
      const i1 = Math.min(i0 + 1, float32.length - 1);
      const frac = srcPos - i0;
      out[i] = float32[i0] * (1 - frac) + float32[i1] * frac;
    }
    return out;
  }

  function concatFloat32(chunks, totalLen) {
    const out = new Float32Array(totalLen);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  }

  function rms(float32) {
    let sum = 0;
    for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
    return Math.sqrt(sum / float32.length);
  }

  async function ensureModel(modelId) {
    if (asrPipelineRef.current && loadedModelIdRef.current === modelId) {
      return asrPipelineRef.current;
    }

    setModelProgress({ percent: 0, label: "loading model… 0%" });
    setStatus("loading model — this happens once and is cached in your browser…", false);

    const { pipeline, env, LogLevel } = await import(/* webpackIgnore: true */ TRANSFORMERS_CDN_URL);
    env.allowLocalModels = false;
    // Keep the console focused on actionable signals — hide the routine
    // onnxruntime execution-provider assignment notices.
    env.logLevel = LogLevel.ERROR;
    // transformers.js defaults the wasm backend to a single thread. Whether
    // wasm ends up as the primary backend (no WebGPU) or just handles the
    // handful of ops WebGPU can't run, give it every core we've got instead
    // of leaving 11/12 idle.
    env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;

    const progressMap = new Map();
    const onProgress = (data) => {
      if (data.status === "progress" && data.file) {
        progressMap.set(data.file, data.progress || 0);
        const vals = [...progressMap.values()];
        const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
        const percent = Math.min(100, avg);
        setModelProgress({ percent, label: `loading model… ${percent.toFixed(0)}%` });
      }
    };

    // Prefer a GPU backend on macOS so Apple devices can use Metal via
    // WebGPU/ONNX Runtime instead of staying on the slower wasm fallback.
    // We still probe the adapter first and only request fp16 if the device
    // actually advertises shader-f16 support.
    const isMacLike = /Mac|iPhone|iPad|iPod/.test(navigator.userAgent || "");
    const hasWebGpu = typeof navigator !== "undefined" && !!navigator.gpu && typeof navigator.gpu.requestAdapter === "function";

    let device = "wasm";
    let dtype;
    let backendLabel = "wasm";

    if (hasWebGpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (adapter) {
          const adapterName = (adapter.name || "").toLowerCase();
          const supportsFp16 = adapter.features?.has("shader-f16");
          device = "webgpu";
          dtype = supportsFp16 ? "fp16" : "fp32";
          backendLabel = isMacLike && adapterName.includes("apple") ? "metal" : "webgpu";
        }
      } catch (err) {
        console.warn("webgpu adapter probe failed; falling back to wasm", err);
      }
    }

    let asrPipeline;
    try {
      asrPipeline = await pipeline("automatic-speech-recognition", modelId, {
        device,
        ...(dtype ? { dtype } : {}),
        progress_callback: onProgress,
      });
    } catch (err) {
      console.warn(`${device} unavailable, falling back to wasm`, err);
      device = "wasm";
      backendLabel = "wasm";
      asrPipeline = await pipeline("automatic-speech-recognition", modelId, {
        device,
        progress_callback: onProgress,
      });
    }

    asrPipelineRef.current = asrPipeline;
    loadedModelIdRef.current = modelId;
    setModelProgress(null);
    setStatus(`model ready (${backendLabel}) — listening…`, true);
    return asrPipeline;
  }

  async function getStream(sourceType) {
    if (sourceType === "mic") {
      return navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
        video: false,
      });
    } else {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
      });
      if (stream.getAudioTracks().length === 0) {
        stream.getTracks().forEach((t) => t.stop());
        throw new Error('No audio track — remember to tick "Share audio" in the picker.');
      }
      stream.getVideoTracks().forEach((t) => t.stop());
      return stream;
    }
  }

  // The pipeline runs one inference at a time on a single ONNX Runtime
  // session. If a window takes longer to transcribe than the
  // NEW_AUDIO_SECONDS it takes to fill the next one (slow hardware, a cold
  // WebGPU shader cache, the bigger Base/Small models), firing handleChunk
  // for every window as it arrives starts a second call while the first is
  // still running. Concurrent runs on one session aren't safe and produce
  // intermittent garbage/errors, and the backlog makes the visible delay
  // grow without bound. Serialize processing through a queue instead, and
  // cap it so a slow stretch causes a bounded catch-up rather than an
  // ever-growing lag.
  const MAX_QUEUED_CHUNKS = 10;

  function enqueueChunk(float32) {
    chunkQueueRef.current.push(float32);
    if (chunkQueueRef.current.length > MAX_QUEUED_CHUNKS) {
      chunkQueueRef.current.splice(0, chunkQueueRef.current.length - MAX_QUEUED_CHUNKS);
    }
    processChunkQueue();
  }

  async function processChunkQueue() {
    if (processingChunkRef.current) return;
    processingChunkRef.current = true;
    try {
      while (runningRef.current && chunkQueueRef.current.length) {
        const next = chunkQueueRef.current.shift();
        await handleChunk(next);
      }
    } finally {
      processingChunkRef.current = false;
    }
  }

  async function handleChunk(float32) {
    // Judge silence on just the newly captured tail, not the retained
    // overlap context, so trailing silence after speech doesn't suppress
    // a window that's mostly old (already-processed) signal.
    const newPortion = float32.subarray(float32.length - NEW_AUDIO_SAMPLES);
    if (rms(newPortion) < SILENCE_RMS_THRESHOLD) return;

    try {
      // No return_timestamps here — word-level timing needs cross-attention
      // DTW alignment (real extra GPU/CPU work), and chunk-level timing
      // isn't used either since overlap dedup works on the text itself.
      // Skipping it is the single biggest per-window speed win available.
      const output = await asrPipelineRef.current(float32);
      // Inference is async — the user may have hit Stop while this window
      // was still transcribing. Drop stale results instead of rendering
      // words in after the session has ended.
      if (!runningRef.current) return;
      enqueueWords(output.text);
    } catch (err) {
      console.error("transcription error", err);
    }
  }

  function normalizeWord(w) {
    return w.toLowerCase().replace(/[^\p{L}\p{N}']/gu, "");
  }

  // The overlap audio at the front of this window covers speech already
  // shown from the previous window, so whisper will very likely transcribe
  // it the same way again. Find the longest run where the tail of what's
  // already on screen matches the head of the new transcript, and treat
  // only what comes after that run as new.
  function countOverlapWords(prevWords, newWords) {
    const maxCheck = Math.min(12, prevWords.length, newWords.length);
    for (let k = maxCheck; k > 0; k--) {
      let match = true;
      for (let i = 0; i < k; i++) {
        if (normalizeWord(prevWords[prevWords.length - k + i]) !== normalizeWord(newWords[i])) {
          match = false;
          break;
        }
      }
      if (match) return k;
    }
    return 0;
  }

  function pushWords(freshWords) {
    setWords((prev) => {
      const next = [...prev, ...freshWords.map((text) => ({ id: wordIdRef.current++, text }))];
      return next.length > MAX_WORDS ? next.slice(-MAX_WORDS) : next;
    });
  }

  // Shared dedup step: figures out which words in a fresh transcript are
  // actually new vs. already-displayed overlap, and returns just the new
  // ones (or null if there's nothing new to show).
  function dedupeIncoming(rawText) {
    const text = (rawText || "").trim();
    if (!text) return null;

    const newWords = text.split(/\s+/).filter(Boolean);
    if (!newWords.length) return null;

    const overlapCount = countOverlapWords(recentWordsRef.current, newWords);
    const freshWords = newWords.slice(overlapCount);
    if (!freshWords.length) return null;

    recentWordsRef.current = [...recentWordsRef.current, ...freshWords].slice(-20);
    return freshWords;
  }

  function enqueueWords(rawText) {
    const freshWords = dedupeIncoming(rawText);
    if (!freshWords) return;
    // Inference is async — the user may have hit Stop while this window
    // was still transcribing. Drop stale results instead of rendering
    // words in after the session has ended.
    if (!runningRef.current) return;
    pushWords(freshWords);
  }

  async function start(sourceType, modelId) {
    setIsBusy(true);

    try {
      await ensureModel(modelId);

      const mediaStream = await getStream(sourceType);
      mediaStreamRef.current = mediaStream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      nativeSampleRateRef.current = audioCtx.sampleRate;
      // Model loading above can take anywhere from a second to tens of
      // seconds on first download, so by the time we get here the original
      // click's user-activation window has often expired. Some browsers
      // then create the context already suspended — everything looks like
      // it's running (status says "listening…") but onaudioprocess never
      // fires and nothing gets transcribed. Resuming explicitly is a no-op
      // if it's already running, so there's no downside to always doing it.
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }

      const sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      sourceNodeRef.current = sourceNode;

      const analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 256;
      sourceNode.connect(analyserNode);
      analyserNodeRef.current = analyserNode;

      const processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
      processorNodeRef.current = processorNode;

      const muteGain = audioCtx.createGain();
      muteGain.gain.value = 0; // never play captured audio back out loud
      muteGainRef.current = muteGain;

      sourceNode.connect(processorNode);
      processorNode.connect(muteGain);
      muteGain.connect(audioCtx.destination);

      pcmBufferRef.current = [];
      pcmBufferLenRef.current = 0;
      chunkQueueRef.current = [];
      recentWordsRef.current = [];
      setWords([]);

      processorNode.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const resampled = downsampleTo16k(input, nativeSampleRateRef.current);
        pcmBufferRef.current.push(resampled.slice());
        pcmBufferLenRef.current += resampled.length;

        if (pcmBufferLenRef.current >= WINDOW_SAMPLES) {
          const full = concatFloat32(pcmBufferRef.current, pcmBufferLenRef.current);
          // Keep the trailing OVERLAP_SAMPLES as context for the next
          // window instead of clearing entirely — this is what makes it a
          // sliding window rather than disjoint chunks.
          const tail = full.slice(full.length - OVERLAP_SAMPLES);
          pcmBufferRef.current = [tail];
          pcmBufferLenRef.current = tail.length;
          enqueueChunk(full);
        }
      };

      runningRef.current = true;
      setStatus("listening…", true);
      setIsListening(true);

      mediaStream.getTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          if (runningRef.current) stop();
        });
      });
    } catch (err) {
      console.error(err);
      let msg = err.message || String(err);
      if (msg.includes("Failed to fetch")) {
        msg =
          "could not download the model — check your internet connection, and disable any ad-blocker/VPN/firewall that might be blocking cdn.jsdelivr.net or huggingface.co, then retry.";
      }
      setStatus(`error: ${msg}`, false);
      setModelProgress(null);
      setIsBusy(false);
    }
  }

  function stop() {
    runningRef.current = false;
    // Drop anything still in flight so no more words render after stop.
    chunkQueueRef.current = [];

    // Each teardown step is isolated: a node that's already disconnected,
    // or an AudioContext that's already closing, can throw in some
    // browsers. One failure here shouldn't stop the rest of cleanup from
    // running or leave the UI stuck in a "still listening" state.
    const safely = (fn, label) => {
      try {
        fn();
      } catch (err) {
        console.warn(`stop: ${label} failed`, err);
      }
    };

    safely(() => {
      if (processorNodeRef.current) {
        processorNodeRef.current.disconnect();
        processorNodeRef.current.onaudioprocess = null;
      }
    }, "processor disconnect");
    safely(() => sourceNodeRef.current && sourceNodeRef.current.disconnect(), "source disconnect");
    safely(() => analyserNodeRef.current && analyserNodeRef.current.disconnect(), "analyser disconnect");
    safely(() => muteGainRef.current && muteGainRef.current.disconnect(), "mute gain disconnect");
    safely(() => audioCtxRef.current && audioCtxRef.current.close(), "audio context close");
    safely(
      () => mediaStreamRef.current && mediaStreamRef.current.getTracks().forEach((t) => t.stop()),
      "media stream stop"
    );

    setStatus("stopped — press start to resume", false);
    setIsBusy(false);
    setIsListening(false);
  }

  useEffect(() => {
    // stop everything if the component unmounts (e.g. navigating away)
    return () => {
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Dev-only escape hatch so display modes can be exercised with canned
  // text when a real mic isn't available (e.g. automated/sandboxed
  // browsers) — feeds the exact same batching path real transcription uses.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    window.__pushTestWords = (text) => {
      const freshWords = dedupeIncoming(text);
      if (freshWords) pushWords(freshWords);
    };
    return () => {
      delete window.__pushTestWords;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    status,
    isLive,
    isBusy,
    isListening,
    words,
    modelProgress,
    analyserNodeRef,
    start,
    stop,
  };
}
