//! The platform-independent half of whisper-in-wasm: Whisper tiny on
//! bunsen's stream driver, fed from memory rather than a file cache.
//!
//! The pipeline per stream:
//!
//! 1. Samples (16 kHz mono) go into bunsen's [`WhisperStreamContext`] under
//!    its *responsive* emission policy: an energy VAD ([`crate::vad`]) gates
//!    the stream into speech regions, a decode commits at every region end
//!    (or when the window fills, at the last timestamp), and every
//!    `draft_interval` of speech a *draft* re-decodes everything since the
//!    last commit.
//! 2. Consecutive drafts are overlapping windows over the same growing
//!    buffer. [`LocalAgreement`] keeps the longest word prefix two drafts in
//!    a row agree on as *stable*: it is shown solid, the rest as
//!    provisional, so text stops flickering long before bunsen commits it.
//! 3. The encoder window is shortened from Whisper's 30 s to
//!    `window_seconds` by truncating the encoder's positional table at load
//!    time (whisper.cpp's `audio_ctx` trick), which is what makes a draft
//!    cheap enough to run several times a second on a CPU.

use std::{sync::Arc, time::Duration};

use bunsen::{
    data::pretrained::apply_rekeyed_tensors_as,
    errors::{BunsenError, BunsenResult},
    kits::{
        speech::{
            asr_driver::{
                EmissionPolicy, LevelPolicy, StreamClock, TranscriptEvent,
                VoiceActivityFilterConfig,
            },
            whisper::{
                WhisperApiConfig, WhisperFallbackConfig,
                driver::{
                    WhisperBundle, WhisperStreamContext, WhisperStreamDriver,
                    WhisperStreamDriverConfig,
                },
                pretrained::hf_rekey,
            },
        },
        tokens::TiktokenRanks,
    },
    burner::module::ModuleInit,
};
use burn::{
    store::{ModuleStore, SafetensorsStore, bridge},
    tensor::{DType, Device, TensorData},
};

use crate::{filters::LiveGuard, vad::EnergyVad};

/// Whisper's mel hop is 10 ms; the encoder halves the frame rate.
const FRAMES_PER_SECOND: usize = 100;
const ENCODER_STRIDE: usize = 2;
pub const SAMPLE_RATE: usize = 16_000;

/// Knobs a caller may turn. Defaults are tuned for tiny.en on a laptop CPU.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// Decode window, in seconds (Whisper's native window is 30).
    pub window_seconds: f32,
    /// Media time of speech between drafts.
    pub draft_interval_ms: u64,
    /// Silence that ends a speech region (and commits it).
    pub min_silence_ms: usize,
    /// VAD sensitivity: how far above the noise floor counts as speech, dB.
    pub vad_margin_db: f32,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            window_seconds: 8.0,
            draft_interval_ms: 500,
            min_silence_ms: 600,
            vad_margin_db: 9.0,
        }
    }
}

/// The CPU backend, named explicitly: with the `gpu` feature on,
/// `Device::default()` would pick wgpu instead.
pub fn cpu_device() -> Device {
    burn::backend::FlexDevice.into()
}

/// A device by name: `cpu` (flex), and with the `gpu` feature `gpu` (wgpu's
/// pick), `dgpu` / `igpu` (first discrete / integrated adapter).
pub fn device_named(name: &str) -> Option<Device> {
    #[cfg(feature = "gpu")]
    {
        use burn::backend::{WgpuDevice, WgpuDeviceKind};
        match name {
            "gpu" => return Some(WgpuDevice::default().into()),
            "dgpu" => return Some(WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0)).into()),
            "igpu" => return Some(WgpuDevice::new(WgpuDeviceKind::IntegratedGpu(0)).into()),
            _ => {}
        }
    }
    (name == "cpu").then(cpu_device)
}

/// Builds a driver over Whisper weights (`transformers` safetensors, any
/// float precision) and a tiktoken vocabulary, all in memory, on the CPU.
pub fn load_driver(
    safetensors: Vec<u8>,
    vocabulary: &str,
    options: &EngineOptions,
) -> BunsenResult<WhisperStreamDriver> {
    load_driver_on(safetensors, vocabulary, options, &cpu_device())
}

/// [`load_driver`] on a chosen device.
pub fn load_driver_on(
    safetensors: Vec<u8>,
    vocabulary: &str,
    options: &EngineOptions,
    device: &Device,
) -> BunsenResult<WhisperStreamDriver> {
    let device = device.clone();

    let mut store = SafetensorsStore::from_bytes(Some(safetensors));
    let tensors = store
        .get_all_tensors()
        .map_err(|e| BunsenError::invalid_resource(format!("safetensors: {e:?}")))?
        .values()
        .cloned()
        .collect::<Vec<_>>();

    let shape_of = |name: &str| {
        tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.shape.to_vec())
            .ok_or_else(|| BunsenError::invalid_resource(format!("checkpoint has no {name}")))
    };
    let conv1 = shape_of("model.encoder.conv1.weight")?;
    let (d_model, n_mels) = (conv1[0], conv1[1]);
    let vocab_size = shape_of("model.decoder.embed_tokens.weight")?[0];
    let full_audio_positions = shape_of("model.encoder.embed_positions.weight")?[0];
    let max_text_ctx = shape_of("model.decoder.embed_positions.weight")?[0];
    let layers = |side: &str| {
        let prefix = format!("model.{side}.layers.");
        tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix) && t.name.ends_with(".self_attn.k_proj.weight"))
            .count()
    };
    let (n_encoder_layers, n_decoder_layers) = (layers("encoder"), layers("decoder"));

    // The shortened window, in encoder positions, on the 20 ms grid.
    let positions = ((options.window_seconds.clamp(2.0, 30.0) as f64
        * (FRAMES_PER_SECOND / ENCODER_STRIDE) as f64)
        .round() as usize)
        .min(full_audio_positions);

    let tensors = tensors
        .into_iter()
        .map(|t| -> BunsenResult<_> {
            if t.name != "model.encoder.embed_positions.weight" || positions == full_audio_positions
            {
                return Ok(t);
            }
            let (name, id) = (t.name.clone(), t.param_id);
            let data = bridge::into_data(t)
                .map_err(|e| BunsenError::invalid_resource(format!("{name}: {e:?}")))?
                .convert::<f32>();
            let width = data.shape()[1];
            let rows: Vec<f32> = data
                .try_to_vec::<f32>()
                .map_err(|e| BunsenError::invalid_resource(format!("{name}: {e:?}")))?
                .into_iter()
                .take(positions * width)
                .collect();
            Ok(bridge::from_data(TensorData::new(rows, [positions, width]), name, id))
        })
        .collect::<BunsenResult<Vec<_>>>()?;

    let cfg = WhisperApiConfig::new(
        n_mels,
        vocab_size,
        d_model,
        positions * ENCODER_STRIDE,
        n_encoder_layers,
        max_text_ctx,
        n_decoder_layers,
    );
    let mut model = cfg.try_init(&device)?;
    apply_rekeyed_tensors_as(&mut model, tensors, &hf_rekey(), Some(DType::F32))?;

    let ranks = TiktokenRanks::parse(vocabulary)?;
    let bundle = Arc::new(WhisperBundle::from_model(model)?.with_ranks(ranks));

    let mut emission = EmissionPolicy::responsive();
    emission.triggers.interval = Some(Duration::from_millis(options.draft_interval_ms));

    let driver = WhisperStreamDriverConfig::new()
        .with_timestamps(true)
        // Room for fast speech (~4 tokens/s) plus timestamp tokens, and no
        // more: a repetition loop otherwise runs to Whisper's 224-token cap,
        // which is seconds of CPU on a draft nobody will read.
        .with_max_tokens(((options.window_seconds * 5.0) as usize + 12).min(224))
        .with_emission(emission)
        // One greedy pass per decode: the temperature ladder would multiply
        // the worst case, which matters more live than a rare hallucination.
        .with_fallback(WhisperFallbackConfig::new())
        .with_speaker_cut(None)
        .init_from_bundle(bundle.clone(), &device)?;

    // Upstream's defaults (suppress blank, suppress tokens), then the live
    // guard; the driver appends the timestamp grammar after these.
    let mut filters = bundle.default_filters();
    filters.push(Arc::new(LiveGuard::new(bundle.layout.ids(), positions)));
    let driver = driver.with_logit_filters(filters);

    let vad = EnergyVad::new(SAMPLE_RATE, 512, options.vad_margin_db);
    driver.with_vad(
        Arc::new(vad),
        VoiceActivityFilterConfig::new()
            .with_min_silence_ms(options.min_silence_ms)
            .with_speech_pad_ms(200)
            .with_max_speech_s(Some((options.window_seconds as f64 - 1.0).max(2.0))),
    )
}

/// Drops a phrase of three or more words that repeats back to back at the
/// end of `text`: what is left of a loop the live guard ended
/// ("And so my fellow Americans. And so my fellow Americans.").
pub fn collapse_repeats(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    'outer: loop {
        let n = words.len();
        for period in 3..=n / 2 {
            let (a, b) = (&words[n - 2 * period..n - period], &words[n - period..]);
            if a.iter().zip(b).all(|(x, y)| norm(x) == norm(y)) {
                words.truncate(n - period);
                continue 'outer;
            }
        }
        return words.join(" ");
    }
}

/// One committed or provisional stretch of text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Piece {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// What the display should show after a push.
#[derive(Clone, Debug, Default)]
pub struct Update {
    /// Newly committed segments, final, in order.
    pub committed: Vec<Piece>,
    /// Draft words two consecutive drafts agreed on (since the last commit).
    pub stable: String,
    /// The rest of the latest draft.
    pub unstable: String,
    /// Whether any decode ran during this push.
    pub decoded: bool,
}

/// LocalAgreement-2 over overlapping drafts (Macháček et al., 2023,
/// "Turning Whisper into Real-Time Transcription System"): a word is stable
/// once two consecutive hypotheses over the same, growing, buffer agree on
/// it and everything before it.
#[derive(Debug, Default)]
pub struct LocalAgreement {
    previous: Vec<String>,
    stable: Vec<String>,
}

fn norm(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

impl LocalAgreement {
    /// Folds in a new hypothesis; returns `(stable, unstable)` text.
    pub fn insert(&mut self, hypothesis: &str) -> (String, String) {
        let words: Vec<String> = hypothesis.split_whitespace().map(str::to_owned).collect();
        let agreed = words
            .iter()
            .zip(&self.previous)
            .take_while(|(a, b)| norm(a) == norm(b))
            .count();
        // Stability is monotone between commits: a later draft that
        // disagrees inside the stable prefix does not un-stabilize it (the
        // commit settles it); its words past the prefix are the tail.
        if agreed > self.stable.len() {
            self.stable = words[..agreed].to_vec();
        }
        let tail = words.iter().skip(self.stable.len()).cloned().collect::<Vec<_>>();
        self.previous = words;
        (self.stable.join(" "), tail.join(" "))
    }

    /// A commit covers everything drafted so far: start over.
    pub fn reset(&mut self) {
        self.previous.clear();
        self.stable.clear();
    }
}

/// A live stream: bunsen's context plus the agreement over its drafts.
pub struct Transcriber {
    ctx: WhisperStreamContext,
    agreement: LocalAgreement,
}

impl Transcriber {
    pub fn new(driver: &WhisperStreamDriver) -> BunsenResult<Self> {
        Ok(Self {
            ctx: driver.new_context(StreamClock::uniform(SAMPLE_RATE), LevelPolicy::Running)?,
            agreement: LocalAgreement::default(),
        })
    }

    /// Writes 16 kHz mono samples and runs whatever decodes are due.
    pub fn push(&mut self, samples: &[f32]) -> BunsenResult<Update> {
        let events = self.ctx.write_read(samples)?;
        Ok(self.fold(events))
    }

    /// Ends the stream: decodes and commits whatever is pending.
    pub fn finish(&mut self) -> BunsenResult<Update> {
        let events = self.ctx.end_read()?;
        let mut update = self.fold(events);
        update.stable.clear();
        update.unstable.clear();
        self.agreement.reset();
        Ok(update)
    }

    fn fold(&mut self, events: Vec<TranscriptEvent>) -> Update {
        let mut update = Update {
            decoded: !events.is_empty(),
            ..Default::default()
        };
        let mut draft = None;
        for event in events {
            let seg = event.segment();
            let piece = Piece {
                start: seg.start,
                end: seg.end,
                text: collapse_repeats(seg.text.as_deref().unwrap_or_default().trim()),
            };
            match event {
                TranscriptEvent::Committed(_) => {
                    // Silence decodes to stray punctuation ("." / ". . ."); drop it.
                    if piece.text.chars().any(char::is_alphanumeric) {
                        update.committed.push(piece);
                    }
                    self.agreement.reset();
                    draft = None;
                }
                TranscriptEvent::Draft(_) => draft = Some(piece),
            }
        }
        if let Some(d) = draft {
            let (stable, unstable) = self.agreement.insert(&d.text);
            update.stable = stable;
            update.unstable = unstable;
        } else if update.committed.is_empty() {
            // No new draft: keep showing the last one.
            let (s, u) = self.agreement.current();
            update.stable = s;
            update.unstable = u;
        }
        update
    }
}

impl LocalAgreement {
    fn current(&self) -> (String, String) {
        let tail = self.previous.iter().skip(self.stable.len()).cloned().collect::<Vec<_>>();
        (self.stable.join(" "), tail.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_trailing_repeats() {
        assert_eq!(
            collapse_repeats("And so my fellow American. And so my fellow American."),
            "And so my fellow American."
        );
        assert_eq!(collapse_repeats("a b c a b c a b c"), "a b c");
        assert_eq!(collapse_repeats("thank you thank you"), "thank you thank you");
        assert_eq!(collapse_repeats("ask not what"), "ask not what");
    }

    #[test]
    fn agreement_stabilizes_common_prefix() {
        let mut la = LocalAgreement::default();
        assert_eq!(la.insert("hello there"), ("".into(), "hello there".into()));
        assert_eq!(la.insert("hello there my"), ("hello there".into(), "my".into()));
        assert_eq!(la.insert("Hello, there, my friend"), ("Hello, there, my".into(), "friend".into()));
        la.reset();
        assert_eq!(la.insert("next"), ("".into(), "next".into()));
    }
}
