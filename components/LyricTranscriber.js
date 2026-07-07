"use client";

import { useEffect, useRef, useState } from "react";

const TARGET_SR = 16000;
const CHUNK_SECONDS = 3.5;
const CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS;
const SILENCE_RMS_THRESHOLD = 0.006;
const VU_BARS = 20;
const TRANSFORMERS_CDN_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

export default function LyricTranscriber() {
  // DOM refs
  const statusLineRef = useRef(null);
  const recDotRef = useRef(null);
  const lyricFeedRef = useRef(null);
  const scrollWrapRef = useRef(null);
  const idleHintRef = useRef(null);
  const vuBarsRef = useRef([]);
  const progressWrapRef = useRef(null);
  const progressLabelRef = useRef(null);
  const progressFillRef = useRef(null);
  const sourceSelectRef = useRef(null);
  const modelSelectRef = useRef(null);

  // Drives the disabled state of the controls declaratively. Toggling
  // .disabled by mutating the DOM node directly (via a ref) fools the
  // visible attribute but not React's event dispatch — React tracks the
  // "disabled" prop from the last render internally and uses that to
  // decide whether to fire onClick at all, so a button re-enabled purely
  // via DOM mutation stops receiving clicks. State + a real re-render
  // avoids that trap.
  const [isBusy, setIsBusy] = useState(false);
  const [isListening, setIsListening] = useState(false);

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
  const vuRafIdRef = useRef(null);
  const nativeSampleRateRef = useRef(48000);

  const pcmBufferRef = useRef([]);
  const pcmBufferLenRef = useRef(0);

  const chunkQueueRef = useRef([]);
  const processingChunkRef = useRef(false);

  // Word-level timestamps need a model exported with cross-attentions
  // (the "_timestamped" onnx-community builds). If a model doesn't have
  // them, fall back to chunk-level timestamps instead of erroring forever.
  const wordTimestampsSupportedRef = useRef(true);

  useEffect(() => {
    // stop everything if the component unmounts (e.g. navigating away)
    return () => {
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setStatus(text, isLive) {
    if (statusLineRef.current) statusLineRef.current.textContent = text;
    if (recDotRef.current) recDotRef.current.classList.toggle("live", !!isLive);
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

    wordTimestampsSupportedRef.current = true;

    if (progressWrapRef.current) progressWrapRef.current.style.display = "flex";
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
        if (progressFillRef.current) {
          progressFillRef.current.style.width = Math.min(100, avg).toFixed(0) + "%";
        }
        if (progressLabelRef.current) {
          progressLabelRef.current.textContent = `loading model… ${Math.min(100, avg).toFixed(0)}%`;
        }
      }
    };

    // fp16 weights halve GPU memory bandwidth and compute vs the fp32
    // default, which is the main win WebGPU offers here — these
    // onnx-community "_timestamped" builds ship fp16 encoder/decoder
    // weights specifically for this. Not all GPUs/drivers expose the
    // shader-f16 feature though, so detect it rather than assume it.
    let webgpuDtype = "fp32";
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      if (adapter?.features?.has("shader-f16")) webgpuDtype = "fp16";
    } catch {
      // no WebGPU adapter — the pipeline() call below will fail and we'll
      // fall back to wasm.
    }

    let device = "webgpu";
    let asrPipeline;
    try {
      asrPipeline = await pipeline("automatic-speech-recognition", modelId, {
        device,
        dtype: webgpuDtype,
        progress_callback: onProgress,
      });
    } catch (err) {
      console.warn("webgpu unavailable, falling back to wasm", err);
      device = "wasm";
      asrPipeline = await pipeline("automatic-speech-recognition", modelId, {
        device,
        progress_callback: onProgress,
      });
    }

    asrPipelineRef.current = asrPipeline;
    loadedModelIdRef.current = modelId;
    if (progressWrapRef.current) progressWrapRef.current.style.display = "none";
    setStatus(`model ready (${device}) — listening…`, true);
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

  function startVuLoop() {
    const analyserNode = analyserNodeRef.current;
    const data = new Uint8Array(analyserNode.fftSize);
    const step = () => {
      if (!runningRef.current) return;
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
        bar.style.background = on
          ? i > VU_BARS * 0.75
            ? "#FF5C5C"
            : "#FFB238"
          : "var(--amber-dim)";
      });
      vuRafIdRef.current = requestAnimationFrame(step);
    };
    step();
  }

  // The pipeline runs one inference at a time on a single ONNX Runtime
  // session. If a chunk takes longer to transcribe than the 3.5s it
  // takes to fill the next one (slow hardware, a cold WebGPU shader
  // cache, the bigger Base model), firing handleChunk for every chunk as
  // it arrives starts a second call while the first is still running.
  // Concurrent runs on one session aren't safe and produce intermittent
  // garbage/errors, and the backlog makes the visible delay grow without
  // bound. Serialize processing through a queue instead, and cap it so a
  // slow stretch causes a bounded catch-up rather than an ever-growing lag.
  const MAX_QUEUED_CHUNKS = 2;

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
    const level = rms(float32);
    if (level < SILENCE_RMS_THRESHOLD) return;
    try {
      const output = await asrPipelineRef.current(float32, {
        return_timestamps: wordTimestampsSupportedRef.current ? "word" : true,
        chunk_length_s: 30,
      });
      // Inference is async — the user may have hit Stop while this chunk
      // was still transcribing. Drop stale results instead of animating
      // words in after the session has ended.
      if (!runningRef.current) return;
      enqueueWords(output);
    } catch (err) {
      const noCrossAttentions =
        wordTimestampsSupportedRef.current &&
        /cross attentions/i.test(err?.message || "");
      if (noCrossAttentions) {
        // This model wasn't exported with output_attentions=True, so word
        // timestamps aren't available. Don't keep retrying "word" mode —
        // fall back to chunk-level timestamps for the rest of the session.
        console.warn(
          "word-level timestamps unsupported by this model, falling back to chunk-level timestamps",
          err
        );
        wordTimestampsSupportedRef.current = false;
        if (!runningRef.current) return;
        try {
          const output = await asrPipelineRef.current(float32, {
            return_timestamps: true,
            chunk_length_s: 30,
          });
          if (!runningRef.current) return;
          enqueueWords(output);
        } catch (fallbackErr) {
          console.error("transcription error", fallbackErr);
        }
      } else {
        console.error("transcription error", err);
      }
    }
  }

  function enqueueWords(output) {
    const text = (output.text || "").trim();
    if (!text) return;

    let words = [];
    if (Array.isArray(output.chunks) && output.chunks.length) {
      words = output.chunks
        .map((c) => c.text)
        .filter((t) => t && t.trim().length)
        .map((t) => t.trim());
    }
    if (!words.length) {
      words = text.split(/\s+/).filter(Boolean);
    }

    if (!runningRef.current) return;

    if (idleHintRef.current && idleHintRef.current.parentNode) {
      idleHintRef.current.remove();
    }

    const feed = lyricFeedRef.current;
    if (!feed) return;

    for (const w of words) {
      const span = document.createElement("span");
      span.className = "word current";
      span.textContent = w;
      feed.appendChild(span);
    }

    const prevCurrent = [...feed.querySelectorAll(".word.current")].slice(0, -1);
    prevCurrent.forEach((el) => {
      el.classList.remove("current");
      el.classList.add("settled");
    });

    if (scrollWrapRef.current) {
      scrollWrapRef.current.scrollTop = scrollWrapRef.current.scrollHeight;
    }

    while (feed.children.length > 220) {
      feed.removeChild(feed.firstChild);
    }
  }

  async function start() {
    setIsBusy(true);

    try {
      await ensureModel(modelSelectRef.current.value);

      const mediaStream = await getStream(sourceSelectRef.current.value);
      mediaStreamRef.current = mediaStream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      nativeSampleRateRef.current = audioCtx.sampleRate;

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

      processorNode.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const resampled = downsampleTo16k(input, nativeSampleRateRef.current);
        pcmBufferRef.current.push(resampled.slice());
        pcmBufferLenRef.current += resampled.length;

        if (pcmBufferLenRef.current >= CHUNK_SAMPLES) {
          const full = concatFloat32(pcmBufferRef.current, pcmBufferLenRef.current);
          pcmBufferRef.current = [];
          pcmBufferLenRef.current = 0;
          enqueueChunk(full);
        }
      };

      runningRef.current = true;
      setStatus("listening…", true);
      startVuLoop();
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
      setIsBusy(false);
    }
  }

  function stop() {
    runningRef.current = false;
    // Drop anything still in flight so no more words render after stop.
    chunkQueueRef.current = [];

    if (vuRafIdRef.current) cancelAnimationFrame(vuRafIdRef.current);
    vuBarsRef.current.forEach((b) => {
      if (!b) return;
      b.style.height = "3px";
      b.style.background = "var(--amber-dim)";
    });

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

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="rec-dot" ref={recDotRef}></span>
          <span>SIGNAL</span>
        </div>
        <div className="status-line mono" ref={statusLineRef}>
          idle — pick a source and press start
        </div>
        <div className="vu">
          {Array.from({ length: VU_BARS }).map((_, i) => (
            <i key={i} ref={(el) => (vuBarsRef.current[i] = el)}></i>
          ))}
        </div>
      </header>

      <main>
        <div id="scrollWrap" ref={scrollWrapRef}>
          <div id="idleHint" ref={idleHintRef}>
            <b>No subscriptions. No cloud API.</b>
            <br />
            Whisper runs locally in your browser via transformers.js.
            <br />
            Choose a source below, hit start, and speak.
          </div>
          <div id="lyricFeed" ref={lyricFeedRef}></div>
        </div>
      </main>

      <footer>
        <label className="field">
          SOURCE
          <select ref={sourceSelectRef} defaultValue="mic" disabled={isBusy}>
            <option value="mic">Microphone</option>
            <option value="system">System / tab audio</option>
          </select>
        </label>

        <label className="field">
          MODEL
          <select ref={modelSelectRef} defaultValue="onnx-community/whisper-tiny.en_timestamped" disabled={isBusy}>
            <option value="onnx-community/whisper-tiny.en_timestamped">Whisper Tiny (fast)</option>
            <option value="onnx-community/whisper-base.en_timestamped">Whisper Base (accurate)</option>
          </select>
        </label>

        <button className="btn-primary" onClick={start} disabled={isBusy}>
          ▶ Start
        </button>
        <button className="btn-stop" onClick={stop} disabled={!isListening}>
          ■ Stop
        </button>

        <div className="spacer"></div>

        <div className="progress-wrap" ref={progressWrapRef} style={{ display: "none" }}>
          <span ref={progressLabelRef}>loading model…</span>
          <div className="progress-bar">
            <div ref={progressFillRef}></div>
          </div>
        </div>
      </footer>
    </div>
  );
}
