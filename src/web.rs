//! The wasm-bindgen surface the web worker drives.
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::engine::{EngineOptions, Transcriber, Update, load_driver};
use bunsen::kits::speech::whisper::driver::WhisperStreamDriver;

/// Exporting `_initialize` makes wasm-ld treat the module as a *reactor*.
/// Without it, the module is a *command*: wasm-ld wraps every export in a
/// call to `__wasm_call_ctors`, so each call into wasm from JS (wasm-bindgen's
/// `__externref_table_dealloc` runs whenever a JS handle is dropped, e.g.
/// every WebGPU bind group) re-runs every static constructor. cubecl and
/// pliron register thousands of `inventory` items that way, which cost
/// ~2 ms per GPU kernel launch. A reactor runs them once, here.
#[unsafe(no_mangle)]
pub extern "C" fn _initialize() {
    unsafe extern "C" {
        fn __wasm_call_ctors();
    }
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe { __wasm_call_ctors() });
}

#[wasm_bindgen(start)]
fn start() {
    _initialize();
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
        use burn::tensor::{Distribution, Tensor};
        let model = self.driver.whisper_model();
        let frames = self.driver.window_frames();
        let mel: Tensor<3> =
            Tensor::random([1, 80, frames], Distribution::Default, &crate::engine::cpu_device());
        let t = js_sys::Date::now();
        let _ = model.forward_encoder(mel).into_data();
        js_sys::Date::now() - t
    }
}


/// WebGPU spike: the encoder alone on WebGPU, read back asynchronously
/// (the browser cannot block on a GPU readback, so bunsen's synchronous
/// decode loop cannot run there as-is).
#[cfg(feature = "gpu")]
mod gpu {
    use burn::backend::{
        AutoGraphicsApi, WgpuDevice,
        wgpu::{RuntimeOptions, try_init_setup_async},
    };
    use burn::tensor::{Device, Distribution, Tensor};
    use serde_json::json;
    use wasm_bindgen::prelude::*;

    use super::js_err;
    use crate::engine::{EngineOptions, cpu_device, load_driver_on};
    use bunsen::kits::speech::whisper::driver::WhisperStreamDriver;

    /// Brings up WebGPU (async: the adapter request is a promise) and
    /// returns the adapter's description as JSON.
    #[wasm_bindgen(js_name = initGpu)]
    pub async fn init_gpu() -> Result<String, JsError> {
        let setup = try_init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), RuntimeOptions::default())
            .await
            .map_err(|e| js_err(format!("{e:?}")))?;
        let info = setup.adapter.get_info();
        Ok(json!({
            "name": info.name,
            "vendor": info.vendor,
            "device_type": format!("{:?}", info.device_type),
            "backend": format!("{:?}", info.backend),
            "driver": info.driver,
        })
        .to_string())
    }

    /// The same model on both backends, for side-by-side encoder timings.
    #[wasm_bindgen]
    pub struct GpuBench {
        cpu: WhisperStreamDriver,
        gpu: WhisperStreamDriver,
        gpu_device: Device,
        load_ms: f64,
    }

    #[wasm_bindgen]
    impl GpuBench {
        /// Call after [`init_gpu`].
        #[wasm_bindgen(constructor)]
        pub fn new(model: Vec<u8>, vocab: &str, window_seconds: f32) -> Result<GpuBench, JsError> {
            let options = EngineOptions { window_seconds, ..Default::default() };
            let gpu_device: Device = WgpuDevice::default().into();
            let cpu = load_driver_on(model.clone(), vocab, &options, &cpu_device()).map_err(js_err)?;
            let t = js_sys::Date::now();
            let gpu = load_driver_on(model, vocab, &options, &gpu_device).map_err(js_err)?;
            let load_ms = js_sys::Date::now() - t;
            Ok(Self { cpu, gpu, gpu_device, load_ms })
        }

        #[wasm_bindgen(getter, js_name = gpuLoadMs)]
        pub fn gpu_load_ms(&self) -> f64 {
            self.load_ms
        }

        /// Milliseconds per CPU encoder forward, mean of `iters`.
        #[wasm_bindgen(js_name = cpuEncoder)]
        pub fn cpu_encoder(&self, iters: u32) -> f64 {
            let model = self.cpu.whisper_model();
            let mel: Tensor<3> =
                Tensor::random([1, 80, self.cpu.window_frames()], Distribution::Default, &cpu_device());
            let t = js_sys::Date::now();
            for _ in 0..iters {
                let _ = model.forward_encoder(mel.clone()).into_data();
            }
            (js_sys::Date::now() - t) / iters as f64
        }

        /// One GPU encoder forward plus the async readback of its features
        /// (what a hybrid would hand the CPU decoder). Milliseconds.
        #[wasm_bindgen(js_name = gpuEncoder)]
        pub async fn gpu_encoder(&self) -> Result<f64, JsError> {
            let model = self.gpu.whisper_model();
            let mel: Tensor<3> =
                Tensor::random([1, 80, self.gpu.window_frames()], Distribution::Default, &self.gpu_device);
            let t = js_sys::Date::now();
            model.forward_encoder(mel).into_data_async().await.map_err(|e| js_err(format!("{e:?}")))?;
            Ok(js_sys::Date::now() - t)
        }

        /// `n` encoder forwards queued back to back, one readback at the end:
        /// per-call cost without the per-call sync. Milliseconds per call.
        #[wasm_bindgen(js_name = gpuEncoderBatch)]
        pub async fn gpu_encoder_batch(&self, n: u32) -> Result<f64, JsError> {
            let model = self.gpu.whisper_model();
            let mel: Tensor<3> =
                Tensor::random([1, 80, self.gpu.window_frames()], Distribution::Default, &self.gpu_device);
            let t = js_sys::Date::now();
            let mut acc = model.forward_encoder(mel.clone());
            for _ in 1..n {
                acc = acc + model.forward_encoder(mel.clone());
            }
            let _ = acc.sum().into_data_async().await.map_err(|e| js_err(format!("{e:?}")))?;
            Ok((js_sys::Date::now() - t) / n as f64)
        }

        /// Splits one encoder call into time to enqueue (CPU side, in wasm)
        /// and time waiting for the GPU. Returns `[enqueue_ms, wait_ms]`.
        #[wasm_bindgen(js_name = gpuEncoderSplit)]
        pub async fn gpu_encoder_split(&self) -> Result<Vec<f64>, JsError> {
            let model = self.gpu.whisper_model();
            let mel: Tensor<3> =
                Tensor::random([1, 80, self.gpu.window_frames()], Distribution::Default, &self.gpu_device);
            let _ = mel.to_data_async().await;
            let t = js_sys::Date::now();
            let fut = model.forward_encoder(mel).sum().into_data_async();
            let enqueue = js_sys::Date::now() - t;
            let _ = fut.await.map_err(|e| js_err(format!("{e:?}")))?;
            Ok(vec![enqueue, js_sys::Date::now() - t - enqueue])
        }

        /// `n` chained tiny elementwise ops, one readback: per-launch cost.
        /// Returns `[enqueue_ms_per_op, total_ms_per_op]`.
        #[wasm_bindgen(js_name = gpuLaunches)]
        pub async fn gpu_launches(&self, n: u32) -> Result<Vec<f64>, JsError> {
            let mut x: Tensor<1> = Tensor::random([1024], Distribution::Default, &self.gpu_device);
            let _ = x.to_data_async().await;
            let t = js_sys::Date::now();
            for _ in 0..n {
                x = x.mul_scalar(1.0001);
            }
            let fut = x.into_data_async();
            let q = js_sys::Date::now() - t;
            let _ = fut.await.map_err(|e| js_err(format!("{e:?}")))?;
            Ok(vec![q / n as f64, (js_sys::Date::now() - t) / n as f64])
        }

        /// One `[m, k] x [k, n]` matmul, read back: raw kernel speed.
        #[wasm_bindgen(js_name = gpuMatmul)]
        pub async fn gpu_matmul(&self, m: usize, k: usize, n: usize, reps: u32) -> Result<f64, JsError> {
            let a: Tensor<2> = Tensor::random([m, k], Distribution::Default, &self.gpu_device);
            let b: Tensor<2> = Tensor::random([k, n], Distribution::Default, &self.gpu_device);
            let t = js_sys::Date::now();
            let mut acc = a.clone().matmul(b.clone());
            for _ in 1..reps {
                acc = acc + a.clone().matmul(b.clone());
            }
            let _ = acc.sum().into_data_async().await.map_err(|e| js_err(format!("{e:?}")))?;
            Ok((js_sys::Date::now() - t) / reps as f64)
        }

        /// Round trip of a tiny GPU op plus readback: the floor on what each
        /// decoder token would cost if the decode loop ran on WebGPU.
        #[wasm_bindgen(js_name = gpuRoundTrip)]
        pub async fn gpu_round_trip(&self) -> Result<f64, JsError> {
            let x: Tensor<2> = Tensor::random([1, 51864], Distribution::Default, &self.gpu_device);
            let t = js_sys::Date::now();
            let _ = x.argmax(1).into_data_async().await.map_err(|e| js_err(format!("{e:?}")))?;
            Ok(js_sys::Date::now() - t)
        }

        /// Runs the stock synchronous driver on WebGPU. Expected to panic
        /// ("Failed to read tensor data synchronously"); kept to confirm it.
        #[wasm_bindgen(js_name = gpuDriverSync)]
        pub fn gpu_driver_sync(&self, samples: &[f32]) -> Result<String, JsError> {
            let mut tr = crate::engine::Transcriber::new(&self.gpu).map_err(js_err)?;
            let mut u = tr.push(samples).map_err(js_err)?;
            u.committed.extend(tr.finish().map_err(js_err)?.committed);
            Ok(u.committed.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join(" "))
        }
    }
}
