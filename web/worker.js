// Web worker: owns the wasm module and the model, and runs every decode off
// the main thread. Audio arrives in ~100 ms chunks; while a decode runs,
// chunks queue up, and the next push takes all of them at once, so the
// stream never falls further behind than one decode.

import init, { WhisperInWasm } from "./pkg/whisper_in_wasm.js";

const MODEL_URL = "models/whisper-tiny.en.safetensors";
const VOCAB_URL = "models/gpt2.tiktoken";
const CACHE = "whisper-in-wasm-models-v1";

let vw = null;
let model = null;
let vocab = null;
let options = null;

let queue = [];
let scheduled = false;
let stopping = false;
let received = 0;   // samples received this stream
let processed = 0;  // samples pushed into the model this stream
let computeMs = 0;  // total push time this stream

const post = (msg) => self.postMessage(msg);

async function fetchCached(url, label) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) {
      post({ type: "progress", label, loaded: 1, total: 1, cached: true });
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch (_) {
    cache = null; // Cache Storage unavailable (e.g. insecure origin)
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} (did you run scripts/fetch-model.sh?)`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    post({ type: "progress", label, loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) {
    bytes.set(p, o);
    o += p.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes));
    } catch (_) {}
  }
  return bytes;
}

function build() {
  const t = performance.now();
  if (vw) vw.free();
  // The constructor copies the bytes into wasm memory; keep ours for a
  // rebuild when the settings change.
  vw = new WhisperInWasm(
    model,
    vocab,
    options.windowSeconds,
    options.draftIntervalMs,
    options.minSilenceMs,
    options.vadMarginDb,
  );
  post({ type: "ready", ms: performance.now() - t, options });
}

async function load(opts) {
  options = opts;
  try {
    await init();
    if (!model) {
      const [m, v] = await Promise.all([
        fetchCached(MODEL_URL, "model"),
        fetchCached(VOCAB_URL, "vocabulary"),
      ]);
      model = m;
      vocab = new TextDecoder().decode(v);
    }
    post({ type: "status", text: "building model…" });
    build();
  } catch (e) {
    post({ type: "error", text: String(e?.message ?? e) });
  }
}

function emit(json, extra) {
  const u = JSON.parse(json);
  post({
    type: "update",
    ...u,
    lagMs: ((received - processed) / 16000) * 1000,
    audioS: processed / 16000,
    computeMs,
    ...extra,
  });
}

function schedule() {
  if (!scheduled) {
    scheduled = true;
    // A macrotask, so audio messages that arrived during the last decode
    // are queued before we drain.
    setTimeout(drain, 0);
  }
}

function drain() {
  scheduled = false;
  if (!vw) return;
  if (queue.length) {
    const n = queue.reduce((a, q) => a + q.length, 0);
    const all = new Float32Array(n);
    let o = 0;
    for (const q of queue) {
      all.set(q, o);
      o += q.length;
    }
    queue = [];
    const t = performance.now();
    try {
      const json = vw.push(all);
      const ms = performance.now() - t;
      computeMs += ms;
      processed += n;
      emit(json, { pushMs: ms, pushedS: n / 16000 });
    } catch (e) {
      post({ type: "error", text: String(e?.message ?? e) });
      queue = [];
    }
  }
  if (queue.length) {
    schedule();
  } else if (stopping) {
    stopping = false;
    const t = performance.now();
    try {
      const json = vw.finish();
      computeMs += performance.now() - t;
      emit(json, { final: true, pushMs: performance.now() - t });
    } catch (e) {
      post({ type: "error", text: String(e?.message ?? e) });
    }
    post({ type: "stopped" });
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "load":
      load(msg.options);
      break;
    case "configure":
      options = msg.options;
      if (model) {
        try {
          build();
        } catch (err) {
          post({ type: "error", text: String(err?.message ?? err) });
        }
      }
      break;
    case "start":
      received = processed = 0;
      computeMs = 0;
      queue = [];
      stopping = false;
      break;
    case "audio":
      received += msg.samples.length;
      queue.push(msg.samples);
      schedule();
      break;
    case "stop":
      stopping = true;
      schedule();
      break;
  }
};
