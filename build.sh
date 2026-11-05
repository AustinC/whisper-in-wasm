#!/usr/bin/env bash
# Builds the wasm module into web/pkg. Run scripts/fetch-model.sh once first.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/whisper_in_wasm.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    web/pkg/whisper_in_wasm_bg.wasm -o web/pkg/whisper_in_wasm_bg.wasm
fi
ls -lh web/pkg
