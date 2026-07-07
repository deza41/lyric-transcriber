# Signal — Live Lyric Transcriber (Next.js)

Turns your microphone or system audio into a live, karaoke-style transcript.
Runs entirely in the browser — Whisper (via [transformers.js](https://huggingface.co/docs/transformers.js)) does the transcription client-side, so there's no API key, no subscription, and no audio ever leaves your machine.

## Run it

```bash
npm install
npm run dev
```

Then open **http://localhost:3000**.

> Use `npm run dev` (or `npm run build && npm run start`) — don't open any of the built files directly as `file://`. The model is fetched over the network at runtime, and browsers block that from `file://` pages.

## Using it

1. Pick a **Source**:
   - **Microphone** — normal mic capture.
   - **System / tab audio** — uses the browser's screen-share picker. When it opens, make sure you tick **"Share audio"**, otherwise there's no audio track to transcribe.
2. Pick a **Model**:
   - **Whisper Tiny** — fastest, good for real-time use on most machines.
   - **Whisper Base** — more accurate, needs more compute per chunk.

   Both use the `_timestamped` ONNX builds (`onnx-community/whisper-tiny.en_timestamped` / `whisper-base.en_timestamped`), which are exported with cross-attentions so word-level timestamps actually work. The plain (non-`_timestamped`) onnx-community Whisper exports don't include cross-attentions and throw "Model outputs must contain cross attentions to extract timestamps" if used with `return_timestamps: "word"`.
3. Press **Start**. The first run downloads the model (a few hundred MB) — it's cached by the browser afterward, so subsequent runs start instantly.
4. Speak. Words appear one at a time, timed to match your speaking cadence, teleprompter/lyric-video style.

## How it works

- Audio is captured via the Web Audio API, resampled to 16kHz, and chunked into ~3.5s windows.
- Each chunk is sent to a local Whisper pipeline (`onnx-community/whisper-tiny.en_timestamped` or `whisper-base.en_timestamped`) running on WebGPU, falling back to WASM if WebGPU isn't available.
- Whisper's word-level timestamps are used to space out how each word appears on screen, instead of dumping the whole sentence at once — that's what gives it the "cadence-matched" lyric feel. If a model ever loads without cross-attentions, the app automatically falls back to chunk-level timestamps instead of erroring out.
- Near-silent chunks are skipped to avoid hallucinated text.

## Notes / limitations

- There's a few seconds of inherent latency — a chunk has to finish recording before it can be transcribed. This is near-real-time, not zero-latency captioning.
- System audio capture depends on your OS/browser's screen-share audio support (works well in Chrome on Windows/ChromeOS/Android; more limited on macOS).
- transformers.js is loaded from a CDN (`cdn.jsdelivr.net`, v4.x) at runtime rather than bundled, to sidestep some known Next.js bundling quirks with `onnxruntime-web`. If you're on a network that blocks that CDN, the app will tell you and suggest checking your connection/firewall/ad-blocker.

## Project structure

```
app/
  layout.js        root layout, fonts, metadata
  page.js           loads the transcriber client-only (no SSR — it needs browser audio APIs)
  globals.css       design tokens + styles
components/
  LyricTranscriber.js   all the capture/transcription/display logic
```
