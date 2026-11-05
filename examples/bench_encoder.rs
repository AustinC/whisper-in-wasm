//! Times one encoder forward at a few window sizes.
use std::time::Instant;

use burn::tensor::{Device, Distribution, Tensor};
use whisper_in_wasm::engine::{EngineOptions, load_driver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("web/models/whisper-tiny.en.safetensors")?;
    let vocab = std::fs::read_to_string("web/models/gpt2.tiktoken")?;
    for w in [5.0f32, 10.0, 30.0] {
        let opts = EngineOptions { window_seconds: w, ..Default::default() };
        let driver = load_driver(bytes.clone(), &vocab, &opts)?;
        let model = driver.whisper_model();
        let frames = (w * 100.0) as usize;
        let mel: Tensor<3> = Tensor::random([1, 80, frames], Distribution::Default, &Device::default());
        let _ = model.forward_encoder(mel.clone()).into_data();
        let t = Instant::now();
        for _ in 0..3 {
            let _ = model.forward_encoder(mel.clone()).into_data();
        }
        println!("window {w:4}s: encoder {:?}", t.elapsed() / 3);
    }
    Ok(())
}
