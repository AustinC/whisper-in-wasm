// WebGPU spike worker: see gpu-bench.html.
import init, { initGpu, GpuBench, WhisperInWasm } from "./pkg-gpu/whisper_in_wasm.js";

const say = (s) => { console.log("[gpu-bench] " + s); self.postMessage(s); };
const fmt = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  return `mean ${mean.toFixed(1)} ms, median ${s[s.length >> 1].toFixed(1)}, min ${s[0].toFixed(1)}, max ${s[s.length - 1].toFixed(1)}`;
};

async function wavSamples(url) {
  const v = new DataView(await (await fetch(url)).arrayBuffer());
  let o = 12;
  while (o < v.byteLength) {
    const id = String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
    const size = v.getUint32(o + 4, true);
    if (id === "data") {
      const out = new Float32Array(size / 2);
      for (let i = 0; i < out.length; i++) out[i] = v.getInt16(o + 8 + 2 * i, true) / 32768;
      return out;
    }
    o += 8 + size + (size & 1);
  }
  throw new Error("no data chunk");
}

self.onmessage = async ({ data: { window } }) => {
  try {
    say(`navigator.gpu: ${"gpu" in navigator}`);
    for (const pref of ["high-performance", "low-power"]) {
      const a = await navigator.gpu?.requestAdapter({ powerPreference: pref });
      const i = a?.info ?? {};
      say(`adapter (${pref}): vendor=${i.vendor} arch=${i.architecture} device=${i.device} desc=${i.description} fallback=${a?.isFallbackAdapter ?? i.isFallbackAdapter}`);
    }
    await init();
    let t = performance.now();
    const adapter = JSON.parse(await initGpu());
    say(`WebGPU up in ${(performance.now() - t).toFixed(0)} ms: ${JSON.stringify(adapter)}`);

    const [model, vocab] = await Promise.all([
      fetch("models/whisper-tiny.en.safetensors").then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
      fetch("models/gpt2.tiktoken").then((r) => r.text()),
    ]);
    t = performance.now();
    const bench = new GpuBench(model, vocab, window);
    say(`loaded both (window ${window}s) in ${(performance.now() - t).toFixed(0)} ms, GPU copy ${bench.gpuLoadMs.toFixed(0)} ms`);

    const cpu = [];
    bench.cpuEncoder(1);
    for (let i = 0; i < 5; i++) cpu.push(bench.cpuEncoder(1));
    say(`CPU encoder (flex, wasm SIMD):   ${fmt(cpu)}`);

    t = performance.now();
    const first = await bench.gpuEncoder();
    say(`GPU encoder first call (shader compile): ${first.toFixed(0)} ms`);
    const gpu = [];
    for (let i = 0; i < 20; i++) gpu.push(await bench.gpuEncoder());
    say(`GPU encoder + async readback:    ${fmt(gpu)}`);

    for (let i = 0; i < 3; i++) { const [q, w] = await bench.gpuEncoderSplit(); say(`GPU encoder split: enqueue ${q.toFixed(0)} ms, wait ${w.toFixed(0)} ms`); }
    await bench.gpuLaunches(10);
    for (const n of [100, 1000]) { const [q, a] = await bench.gpuLaunches(n); say(`GPU ${n} tiny ops: enqueue ${(q * 1000).toFixed(0)} us/op, total ${(a * 1000).toFixed(0)} us/op`); }
    for (const n of [1, 5, 20]) say(`GPU encoder x${n}, one readback: ${(await bench.gpuEncoderBatch(n)).toFixed(1)} ms/call`);
    await bench.gpuMatmul(400, 384, 1536, 1);
    say(`GPU matmul 400x384x1536 x50: ${(await bench.gpuMatmul(400, 384, 1536, 50)).toFixed(2)} ms each`);

    await bench.gpuRoundTrip();
    const rt = [];
    for (let i = 0; i < 50; i++) rt.push(await bench.gpuRoundTrip());
    say(`GPU argmax round trip (per-token floor): ${fmt(rt)}`);

    // The stock CPU app path on the same bundle: checks the reactor fix.
    {
      const samples = await wavSamples("jfk.wav");
      const vw = new WhisperInWasm(model, vocab, window, 500, 600, 9);
      const text = [];
      let t0 = performance.now();
      for (let i = 0; i < samples.length; i += 1600) {
        for (const c of JSON.parse(vw.push(samples.subarray(i, i + 1600))).committed) text.push(c.text);
      }
      for (const c of JSON.parse(vw.finish()).committed) text.push(c.text);
      const ms = performance.now() - t0;
      say(`CPU app path on jfk.wav: ${text.join(" ")}  (compute ${(ms / 1000).toFixed(1)} s for ${(samples.length / 16000).toFixed(1)} s, RTF ${(ms / 1000 / (samples.length / 16000)).toFixed(2)})`);
      vw.free();
    }

    say(`\nStock synchronous driver on WebGPU (expected to panic)…`);
    const samples = await wavSamples("jfk.wav");
    try {
      say(`  transcript: ${bench.gpuDriverSync(samples)}`);
    } catch (e) {
      say(`  failed: ${e.message ?? e}`);
    }
    say("done");
  } catch (e) {
    say(`error: ${e.message ?? e}\n${e.stack ?? ""}`);
  }
};
