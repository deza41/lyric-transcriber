/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // onnxruntime-web's multi-threaded wasm backend needs SharedArrayBuffer,
  // which browsers only expose in a crossOriginIsolated context. Without
  // these headers it silently falls back to single-threaded wasm — no
  // error, just every core but one sitting idle whenever wasm is in play
  // (full fallback, or just the ops WebGPU can't run).
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
