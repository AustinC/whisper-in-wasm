//! Streams a 16 kHz mono WAV through the engine in real-time-sized chunks
//! and reports what the browser would show, plus how long each decode took.
//!
//! cargo run --release --example stream_wav -- path.wav [window_seconds]
use std::time::Instant;

use whisper_in_wasm::engine::{EngineOptions, Transcriber, load_driver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let wav = args.next().expect("usage: stream_wav <16k mono wav> [window_seconds]");
    let mut options = EngineOptions::default();
    if let Some(w) = args.next() {
        options.window_seconds = w.parse()?;
    }

    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    assert_eq!((spec.sample_rate, spec.channels), (16_000, 1), "need 16 kHz mono");
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;

    let t = Instant::now();
    let driver = load_driver(
        std::fs::read("web/models/whisper-tiny.en.safetensors")?,
        &std::fs::read_to_string("web/models/gpt2.tiktoken")?,
        &options,
    )?;
    println!("loaded in {:?} ({options:?})", t.elapsed());

    let mut tr = Transcriber::new(&driver)?;
    let mut busy = 0.0;
    for (i, chunk) in samples.chunks(1600).enumerate() {
        let t = Instant::now();
        let u = tr.push(chunk)?;
        let dt = t.elapsed().as_secs_f64();
        busy += dt;
        if u.decoded {
            for c in &u.committed {
                println!("[{:5.1}s] COMMIT {:5.2}-{:5.2} {}  ({dt:.2}s)", i as f64 * 0.1, c.start, c.end, c.text);
            }
            if !u.stable.is_empty() || !u.unstable.is_empty() {
                println!("[{:5.1}s] draft  <{}> {}  ({dt:.2}s)", i as f64 * 0.1, u.stable, u.unstable);
            }
        }
    }
    let t = Instant::now();
    for c in tr.finish()?.committed {
        println!("[ end ] COMMIT {:5.2}-{:5.2} {}", c.start, c.end, c.text);
    }
    busy += t.elapsed().as_secs_f64();
    let audio = samples.len() as f64 / 16_000.0;
    println!("audio {audio:.1}s, compute {busy:.1}s, real-time factor {:.2}", busy / audio);
    Ok(())
}
