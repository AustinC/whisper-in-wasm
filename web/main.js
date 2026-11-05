// Main thread: microphone, visuals and transcript. Inference lives in
// worker.js; audio flows mic -> AudioWorklet (16 kHz) -> here -> worker.

const $ = (id) => document.getElementById(id);
const micBtn = $("mic"), statusEl = $("status"), progressEl = $("progress");
const transcriptEl = $("transcript"), errorEl = $("error");
const canvas = $("scope"), g = canvas.getContext("2d");

const RATE = 16000;
const PEAK_HOP = 160;                // one waveform column per 10 ms
const VIEW_SECONDS = 12;

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

let ready = false, recording = false, stopping = false;
let ctx = null, node = null, stream = null;
let received = 0;                    // 16 kHz samples sent this stream
let peaks = [];                      // [{t, v}] per 10 ms
let segments = [];                   // committed, this stream: {start, end, text}
let committedText = "";              // whole transcript (across streams)
let stable = "", unstable = "";
let levelSmooth = 0;
let fileSource = null;
// Held for the whole recording: Chrome may garbage-collect a source node
// that only a local variable refers to, and the audio silently stops.
let sourceNode = null, sinkNode = null;
let lastAudioAt = 0;

// ---- settings --------------------------------------------------------------
const opts = {
  window: [$("o-window"), $("v-window"), (v) => `${v} s`],
  draft: [$("o-draft"), $("v-draft"), (v) => `${v} ms`],
  silence: [$("o-silence"), $("v-silence"), (v) => `${v} ms`],
  vad: [$("o-vad"), $("v-vad"), (v) => `${v} dB`],
};
try {
  const saved = JSON.parse(localStorage.getItem("wiw-opts") || "{}");
  for (const [k, [inp]] of Object.entries(opts)) if (saved[k] != null) inp.value = saved[k];
} catch (_) {}
const options = () => ({
  windowSeconds: Number(opts.window[0].value),
  draftIntervalMs: Number(opts.draft[0].value),
  minSilenceMs: Number(opts.silence[0].value),
  vadMarginDb: Number(opts.vad[0].value),
});
for (const [k, [inp, out, fmt]] of Object.entries(opts)) {
  out.textContent = fmt(inp.value);
  inp.addEventListener("input", () => (out.textContent = fmt(inp.value)));
  inp.addEventListener("change", () => {
    try {
      localStorage.setItem("wiw-opts", JSON.stringify(
        Object.fromEntries(Object.entries(opts).map(([k2, [i]]) => [k2, i.value]))));
    } catch (_) {}
    if (recording) return; // applied on the next start
    setReady(false, "rebuilding model…");
    worker.postMessage({ type: "configure", options: options() });
  });
}

// ---- worker ----------------------------------------------------------------
function setReady(ok, text) {
  ready = ok;
  micBtn.disabled = !ok && !recording;
  statusEl.textContent = text;
}

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "progress": {
      const pct = m.total ? (100 * m.loaded) / m.total : 0;
      if (m.label === "model") {
        progressEl.style.width = `${pct}%`;
        statusEl.textContent = m.cached
          ? "model from cache"
          : `downloading whisper tiny.en… ${(m.loaded / 1e6).toFixed(0)}${m.total ? ` / ${(m.total / 1e6).toFixed(0)}` : ""} MB`;
      }
      break;
    }
    case "status":
      statusEl.textContent = m.text;
      break;
    case "ready":
      progressEl.style.width = "100%";
      setTimeout(() => (progressEl.parentElement.style.visibility = "hidden"), 400);
      setReady(true, `ready · tiny.en · ${m.options.windowSeconds} s window · built in ${(m.ms / 1000).toFixed(1)} s`);
      if (autoSrc && !recording) start(autoSrc);
      break;
    case "update":
      onUpdate(m);
      break;
    case "stopped":
      stopping = false;
      micBtn.disabled = !ready;
      statusEl.textContent = "ready";
      break;
    case "error":
      errorEl.textContent = m.text;
      statusEl.textContent = "error";
      break;
  }
};
worker.postMessage({ type: "load", options: options() });

function onUpdate(m) {
  for (const s of m.committed) {
    // A segment's timestamps can run into the window's zero padding;
    // keep the timeline monotone and inside the audio heard so far.
    const now = received / RATE;
    const prev = segments.length ? segments[segments.length - 1].end : 0;
    const start = Math.min(Math.max(s.start, prev), now);
    segments.push({ ...s, start, end: Math.max(start, Math.min(s.end, now)) });
    committedText += (committedText && !/\s$/.test(committedText) ? " " : "") + s.text;
  }
  stable = m.stable;
  unstable = m.unstable;
  if (m.final) {
    stable = unstable = "";
    if (committedText) committedText += "\n";
  }
  renderTranscript();
  if (m.pushMs != null) $("s-push").textContent = `${Math.round(m.pushMs)} ms`;
  // Lag: audio captured here that the model has not yet heard.
  $("s-lag").textContent = `${Math.max(0, received / RATE - m.audioS).toFixed(1)} s`;
  if (m.audioS > 0) $("s-rtf").textContent = (m.computeMs / 1000 / m.audioS).toFixed(2);
}

function renderTranscript() {
  transcriptEl.textContent = "";
  const add = (cls, text) => {
    const s = document.createElement("span");
    s.className = cls;
    s.textContent = text;
    transcriptEl.append(s);
  };
  if (!committedText && !stable && !unstable && !recording) {
    add("hint", "Click the mic and start talking.");
    return;
  }
  const sep = (t) => (t && !/\s$/.test(t) ? " " : "");
  add("committed", committedText);
  if (stable) add("stable", sep(committedText) + stable);
  if (unstable) add("unstable", sep(committedText + stable) + unstable);
  if (recording) transcriptEl.append(Object.assign(document.createElement("span"), { className: "caret" }));
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// ---- microphone ------------------------------------------------------------
// `file`: an audio File/Blob/URL to play through the same pipeline instead
// of the microphone (it is heard, and transcribed in real time).
async function start(file = null) {
  errorEl.textContent = "";
  try {
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule(new URL("./audio-processor.js", import.meta.url));
    node = new AudioWorkletNode(ctx, "downsampler");
    node.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "audio") onAudio(m.samples, m.rms);
      else if (m.type === "flushed") finishStop();
    };
    let src;
    if (file) {
      const bytes = typeof file === "string"
        ? await (await fetch(file)).arrayBuffer()
        : await file.arrayBuffer();
      src = ctx.createBufferSource();
      src.buffer = await ctx.decodeAudioData(bytes);
      src.connect(ctx.destination);
      src.onended = () => setTimeout(stop, 300);
      src.start();
      fileSource = src;
    } else {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      src = ctx.createMediaStreamSource(stream);
      const track = stream.getAudioTracks()[0];
      track.onended = () => recording && warn("the microphone was disconnected");
      track.onmute = () => recording && warn("the microphone was muted by the browser or system");
      track.onunmute = () => recording && (statusEl.textContent = "listening…");
    }
    sourceNode = src;
    src.connect(node);
    // The worklet feeds a silent gain into the destination, so the graph
    // always pulls it (an unconnected AudioWorkletNode may stop being
    // processed); the mic itself is never played back.
    sinkNode = ctx.createGain();
    sinkNode.gain.value = 0;
    node.connect(sinkNode).connect(ctx.destination);
    if (ctx.state !== "running") await ctx.resume();
  } catch (e) {
    errorEl.textContent = file ? `Could not play that file: ${e.message ?? e}` : `Microphone unavailable: ${e.message ?? e}`;
    cleanup();
    return;
  }
  received = 0;
  lastAudioAt = performance.now();
  peaks = [];
  segments = [];
  stable = unstable = "";
  worker.postMessage({ type: "start" });
  recording = true;
  micBtn.classList.add("on");
  micBtn.setAttribute("aria-label", "Stop transcribing");
  statusEl.textContent = file ? "transcribing file…" : "listening…";
  renderTranscript();
}

function stop() {
  if (!recording) return;
  recording = false;
  stopping = true;
  micBtn.classList.remove("on");
  micBtn.setAttribute("aria-label", "Start transcribing");
  micBtn.disabled = true;
  statusEl.textContent = "finishing…";
  node?.port.postMessage("flush");
  // finishStop runs when the worklet has flushed its partial chunk.
}

function finishStop() {
  worker.postMessage({ type: "stop" });
  cleanup();
}

function cleanup() {
  stream?.getTracks().forEach((t) => t.stop());
  try { fileSource?.stop(); } catch (_) {}
  fileSource = null;
  sourceNode?.disconnect();
  sinkNode?.disconnect();
  node?.disconnect();
  ctx?.close();
  stream = node = ctx = sourceNode = sinkNode = null;
  micBtn.style.setProperty("--level", 0);
}

function onAudio(samples, rms) {
  if (!recording && !stopping) return;
  if (recording && performance.now() - lastAudioAt > 1500) statusEl.textContent = "listening…";
  lastAudioAt = performance.now();
  for (let i = 0; i < samples.length; i += PEAK_HOP) {
    let p = 0;
    const end = Math.min(i + PEAK_HOP, samples.length);
    for (let j = i; j < end; j++) p = Math.max(p, Math.abs(samples[j]));
    peaks.push({ t: (received + i) / RATE, v: p });
  }
  const cutoff = (received + samples.length) / RATE - VIEW_SECONDS - 1;
  while (peaks.length && peaks[0].t < cutoff) peaks.shift();
  received += samples.length;
  levelSmooth = Math.max(Math.min(1, rms * 6), levelSmooth * 0.8);
  micBtn.style.setProperty("--level", levelSmooth.toFixed(3));
  worker.postMessage({ type: "audio", samples }, [samples.buffer]);
}

function warn(text) {
  statusEl.textContent = `⚠ ${text}`;
}

// Watchdog: say so if the audio stops arriving, rather than freezing silently.
setInterval(() => {
  if (!recording || fileSource) return;
  const gap = (performance.now() - lastAudioAt) / 1000;
  if (gap > 1.5) {
    const track = stream?.getAudioTracks()[0];
    warn(`no audio for ${gap.toFixed(0)} s (audio context ${ctx?.state}, mic ${track?.readyState}${track?.muted ? ", muted" : ""})`);
    if (ctx?.state === "suspended") ctx.resume();
  }
}, 500);

micBtn.addEventListener("click", () => (recording ? stop() : ready && start()));
$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f && ready && !recording && !stopping) start(f);
});
// ?src=<url>: transcribe that audio file as soon as the model is ready.
const autoSrc = new URLSearchParams(location.search).get("src");
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    micBtn.click();
  }
});
$("clear").addEventListener("click", () => {
  committedText = "";
  stable = unstable = "";
  segments = [];
  renderTranscript();
});
$("copy").addEventListener("click", async () => {
  const text = [committedText, stable, unstable].filter(Boolean).join(" ").trim();
  try {
    await navigator.clipboard.writeText(text);
    $("copy").textContent = "Copied";
    setTimeout(() => ($("copy").textContent = "Copy"), 1200);
  } catch (_) {}
});

// ---- scope -----------------------------------------------------------------
function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const now = received / RATE;
  const t0 = now - VIEW_SECONDS;
  const x = (t) => ((t - t0) / VIEW_SECONDS) * w;

  // Pending region: since the last commit (what drafts are covering).
  const lastEnd = segments.length ? segments[segments.length - 1].end : 0;
  if (recording && now > lastEnd) {
    g.fillStyle = css.getPropertyValue("--draft");
    g.fillRect(Math.max(0, x(lastEnd)), 0, w - Math.max(0, x(lastEnd)), h);
  }
  // Committed segments, with their text.
  g.font = "11px system-ui, sans-serif";
  g.textBaseline = "top";
  for (const s of segments) {
    if (s.end < t0) continue;
    const a = x(s.start), b = x(s.end);
    g.fillStyle = css.getPropertyValue("--commit");
    g.fillRect(a, 0, b - a, h);
    g.fillStyle = css.getPropertyValue("--commit-edge");
    g.fillRect(a, 0, 2, h);
    g.save();
    g.beginPath();
    g.rect(a, 0, b - a, h);
    g.clip();
    g.fillStyle = css.getPropertyValue("--muted");
    g.fillText(s.text, a + 6, 6);
    g.restore();
  }
  // Waveform.
  g.fillStyle = css.getPropertyValue("--wave");
  const colW = Math.max(1, (w / (VIEW_SECONDS * 100)) * 0.75);
  for (const p of peaks) {
    if (p.t < t0) continue;
    const amp = Math.min(1, Math.sqrt(p.v) * 1.1) * (h / 2 - 8);
    g.fillRect(x(p.t), h / 2 - amp, colW, Math.max(1, amp * 2));
  }
  if (!recording && !peaks.length) {
    g.fillStyle = css.getPropertyValue("--faint");
    g.fillRect(0, h / 2, w, 1);
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
