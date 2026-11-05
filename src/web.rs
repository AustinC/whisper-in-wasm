//! The wasm-bindgen surface the web worker drives.
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::engine::{EngineOptions, Transcriber, Update, load_driver};
use bunsen::kits::speech::whisper::driver::WhisperStreamDriver;

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn to_json(u: &Update) -> String {
    json!({
        "committed": u.committed.iter().map(|p| json!({"start": p.start, "end": p.end, "text": p.text})).collect::<Vec<_>>(),
        "stable": u.stable,
        "unstable": u.unstable,
        "decoded": u.decoded,
    })
    .to_string()
}

/// A loaded model and one live stream over it.
#[wasm_bindgen]
pub struct WhisperInWasm {
    driver: WhisperStreamDriver,
    stream: Transcriber,
}

#[wasm_bindgen]
impl WhisperInWasm {
    /// `model`: transformers-layout Whisper safetensors; `vocab`: the
    /// tiktoken rank file's text.
    #[wasm_bindgen(constructor)]
    pub fn new(
        model: Vec<u8>,
        vocab: &str,
        window_seconds: f32,
        draft_interval_ms: u32,
        min_silence_ms: u32,
        vad_margin_db: f32,
    ) -> Result<WhisperInWasm, JsError> {
        let options = EngineOptions {
            window_seconds,
            draft_interval_ms: draft_interval_ms as u64,
            min_silence_ms: min_silence_ms as usize,
            vad_margin_db,
        };
        let driver = load_driver(model, vocab, &options).map_err(js_err)?;
        let stream = Transcriber::new(&driver).map_err(js_err)?;
        Ok(Self { driver, stream })
    }

    /// Pushes 16 kHz mono samples; returns the update as JSON.
    pub fn push(&mut self, samples: &[f32]) -> Result<String, JsError> {
        self.stream.push(samples).map(|u| to_json(&u)).map_err(js_err)
    }

    /// Ends the stream (commits what is pending) and opens a fresh one.
    pub fn finish(&mut self) -> Result<String, JsError> {
        let u = self.stream.finish().map_err(js_err)?;
        self.stream = Transcriber::new(&self.driver).map_err(js_err)?;
        Ok(to_json(&u))
    }
}

#[wasm_bindgen]
impl WhisperInWasm {
    /// Milliseconds for one encoder forward over a full window (diagnostics).
    pub fn bench_encoder(&self) -> f64 {
        use burn::tensor::{Device, Distribution, Tensor};
        let model = self.driver.whisper_model();
        let frames = self.driver.window_frames();
        let mel: Tensor<3> = Tensor::random([1, 80, frames], Distribution::Default, &Device::default());
        let t = js_sys::Date::now();
        let _ = model.forward_encoder(mel).into_data();
        js_sys::Date::now() - t
    }
}
