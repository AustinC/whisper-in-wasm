//! Times one encoder forward at a few window sizes.
//!
//! cargo run --release --example bench_encoder [--features gpu] -- [cpu|gpu|dgpu|igpu]
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};
use whisper_in_wasm::engine::{EngineOptions, device_named, load_driver_on};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "cpu".into());
    let device = device_named(&name).ok_or("unknown device (gpu ones need --features gpu)")?;
    let bytes = std::fs::read("web/models/whisper-tiny.en.safetensors")?;
    let vocab = std::fs::read_to_string("web/models/gpt2.tiktoken")?;
    println!("device {name}: {device:?}");
    for w in [5.0f32, 8.0, 10.0, 30.0] {
        let opts = EngineOptions { window_seconds: w, ..Default::default() };
        let driver = load_driver_on(bytes.clone(), &vocab, &opts, &device)?;
        let model = driver.whisper_model();
        let frames = driver.window_frames();
        let mel: Tensor<3> = Tensor::random([1, 80, frames], Distribution::Default, &device);
        // Warm-up: shader compilation and autotune on the GPU.
        let t = Instant::now();
        let _ = model.forward_encoder(mel.clone()).into_data();
        let first = t.elapsed();
        let t = Instant::now();
        for _ in 0..5 {
            let _ = model.forward_encoder(mel.clone()).into_data();
        }
        println!("window {w:4}s: encoder {:?} (first call {first:?})", t.elapsed() / 5);
    }
    Ok(())
}
