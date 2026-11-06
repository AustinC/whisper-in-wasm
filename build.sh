#!/usr/bin/env bash
# Builds the wasm module into web/pkg. Run scripts/fetch-model.sh once first.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")"
# GPU=1 builds the WebGPU spike (web/gpu-bench.html) into web/pkg-gpu instead.
if [[ "${GPU:-}" == 1 ]]; then features=(--features gpu); out=web/pkg-gpu; else features=(); out=web/pkg; fi
cargo build --release --target wasm32-unknown-unknown --lib "${features[@]}"
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  target/wasm32-unknown-unknown/release/whisper_in_wasm.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$out/whisper_in_wasm_bg.wasm" -o "$out/whisper_in_wasm_bg.wasm"
fi
ls -lh "$out"
