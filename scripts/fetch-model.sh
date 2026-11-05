#!/usr/bin/env bash
# Downloads Whisper tiny.en (Hugging Face transformers safetensors) and the
# GPT-2 tiktoken vocabulary into web/models/, checking the vocabulary digest.
set -euo pipefail
cd "$(dirname "$0")/.."
out=web/models
mkdir -p "$out"

model_url="https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors"
vocab_url="https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/gpt2.tiktoken"
vocab_sha="306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"

if [[ ! -s $out/whisper-tiny.en.safetensors ]]; then
  echo "fetching $model_url"
  curl -fL --progress-bar -o "$out/whisper-tiny.en.safetensors.part" "$model_url"
  mv "$out/whisper-tiny.en.safetensors.part" "$out/whisper-tiny.en.safetensors"
fi
if [[ ! -s $out/gpt2.tiktoken ]]; then
  echo "fetching $vocab_url"
  curl -fL --progress-bar -o "$out/gpt2.tiktoken" "$vocab_url"
fi
echo "$vocab_sha  $out/gpt2.tiktoken" | sha256sum -c -
ls -lh "$out"
