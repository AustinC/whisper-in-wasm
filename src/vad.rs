//! A small adaptive energy VAD behind bunsen's [`VoiceActivityModel`] trait.
//!
//! bunsen's real-time presets want a voice-activity model to find speech
//! endpoints. Silero is the usual one; this stands in for it so the page
//! downloads one model instead of two. It tracks a noise floor (fast down,
//! slow up) and maps each chunk's RMS level above that floor to a
//! probability through a logistic around `margin_db`. bunsen's
//! [`VoiceActivityFilter`](bunsen::kits::speech::asr_driver::VoiceActivityFilter)
//! does the hysteresis, padding and minimum-silence rules on top.

use bunsen::kits::speech::asr_driver::{VoiceActivityModel, VoiceActivityStream};

#[derive(Clone, Debug)]
pub struct EnergyVad {
    rate: usize,
    chunk: usize,
    margin_db: f32,
}

impl EnergyVad {
    pub fn new(rate: usize, chunk: usize, margin_db: f32) -> Self {
        Self {
            rate,
            chunk,
            margin_db,
        }
    }
}

impl VoiceActivityModel for EnergyVad {
    fn sample_rate(&self) -> usize {
        self.rate
    }

    fn chunk_size(&self) -> usize {
        self.chunk
    }

    fn new_stream(&self) -> Box<dyn VoiceActivityStream> {
        Box::new(EnergyVadStream {
            chunk: self.chunk,
            margin_db: self.margin_db,
            floor_db: None,
            chunk_secs: self.chunk as f32 / self.rate as f32,
        })
    }
}

#[derive(Clone, Debug)]
struct EnergyVadStream {
    chunk: usize,
    margin_db: f32,
    floor_db: Option<f32>,
    chunk_secs: f32,
}

/// The quietest a floor may sit: digital silence would otherwise make any
/// hiss "speech".
const MIN_FLOOR_DB: f32 = -70.0;

impl VoiceActivityStream for EnergyVadStream {
    fn probabilities(&mut self, samples: Vec<f32>) -> Vec<f32> {
        samples
            .chunks_exact(self.chunk)
            .map(|c| {
                let rms = (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt();
                let db = 20.0 * rms.max(1e-7).log10();
                let floor = *self.floor_db.get_or_insert(db.max(MIN_FLOOR_DB));
                // Fast attack downward (≈0.1 s), slow release upward (≈4 s),
                // so speech does not drag the floor up with it.
                let tau = if db < floor { 0.1 } else { 4.0 };
                let a = (-self.chunk_secs / tau).exp();
                let next = (a * floor + (1.0 - a) * db).max(MIN_FLOOR_DB);
                self.floor_db = Some(next);
                let above = db - floor;
                1.0 / (1.0 + (-(above - self.margin_db) / 2.0).exp())
            })
            .collect()
    }
}
