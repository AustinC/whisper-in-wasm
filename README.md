# whisper-in-wasm

Near-real-time speech-to-text in the browser: click the mic, talk, and watch
the transcript form. Whisper **tiny.en** runs locally in WebAssembly (Rust,
[bunsen](https://zspacelabs.ai/bunsen/book/) on [burn](https://burn.dev)'s CPU
backend with wasm SIMD). No audio leaves the page.

<img width="1283" height="800" alt="whisper-wasm-test" src="https://github.com/user-attachments/assets/2fbedaa6-b0fb-4092-82e6-a69243dedcdf" />


## Run

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130   # must match Cargo.lock
./scripts/fetch-model.sh     # whisper tiny.en safetensors (145 MB) + gpt2.tiktoken
./build.sh                   # -> web/pkg
./scripts/serve.py           # http://localhost:8080
```

Open http://localhost:8080 and click the mic (or press space). **Audio file…**
plays a file through the same pipeline; `?src=jfk.wav` does that on load.
The model is cached in Cache Storage after the first load.

## How it works

```
mic ─▶ AudioWorklet ─16 kHz─▶ main thread ─▶ Web Worker ─▶ wasm (Rust)
       (box-filter             (waveform,     (queues audio    bunsen WhisperStreamContext
        downsample)             transcript)    while decoding)  + LocalAgreement
```

* **bunsen-asr's engine, as a library.** `bunsen-asr` is a CLI (cpal, file
  cache, network), so it can't run in a browser. Its engine lives in the
  `bunsen` crate, which builds for `wasm32-unknown-unknown`. This app uses
  the same `WhisperStreamDriver` that `bunsen-asr`'s *responsive* preset uses,
  but loads the weights and vocabulary from bytes instead of its file cache
  (`src/engine.rs`).
* **Overlapping sliding windows.** Under the responsive policy, every
  *draft interval* of speech (default 500 ms) the driver re-decodes all audio
  since the last commit, so consecutive drafts are overlapping windows over a
  growing buffer. A segment is *committed* when the VAD sees a pause, or at
  the last timestamp when the window fills, and the window then slides past it.
* **LocalAgreement-2** (Macháček et al., 2023): a word becomes *stable* once
  two consecutive drafts agree on it and everything before it. Stable words
  are shown solid, the rest in italics, so text stops flickering well before
  it is committed.
* **Short window.** Whisper always encodes 30 s. Truncating the encoder's
  positional table at load time (whisper.cpp's `audio_ctx` trick) gives an
  8 s window by default, and the encoder cost drops from ~1 s to ~0.3 s in wasm.
* **Live guard** (`src/filters.rs`): suppresses timestamps past the shortened
  window and ends repetition loops (`f. f. f.`, a sentence ×3) with
  `<|endoftext|>`. Tiny produces these on slivers of audio, and without the
  guard they run to the token cap.
* **Energy VAD** (`src/vad.rs`), an adaptive-noise-floor VAD behind bunsen's
  `VoiceActivityModel` trait, so the page downloads one model instead of two
  (bunsen's default is Silero).

## Tuning

**Streaming settings** in the page: window length, draft interval, silence to
commit, and VAD margin. Measured on this machine (single-threaded wasm,
Chromium): ~0.7–2 s per decode, about 1–3 s from speech to text, and a
real-time factor of about 1.0 under continuous speech. Commits after pauses
make it catch up.

Native check, the same engine without a browser:

```sh
cargo run --release --example stream_wav -- web/jfk.wav [window_seconds]
cargo run --release --example bench_encoder
```

## Next steps

* **Threads**: wasm threads plus `wasm-bindgen-rayon` (nightly `build-std`,
  COOP/COEP headers) would let burn-flex's gemm parallelize the encoder.
* **Multilingual**: `openai/whisper-tiny` with `multilingual.tiktoken` loads
  through the same path. The UI only needs a model switch.
* **f16 weights** would halve the 145 MB download. Loading already casts any
  float precision to f32.
