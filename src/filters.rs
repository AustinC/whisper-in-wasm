//! Logit filters for live decoding on a shortened window.

use burn::{
    Tensor,
    prelude::{Bool, TensorData},
};
use bunsen::kits::speech::whisper::{driver::WhisperSpecialIds, logit_filters::LogitFilter};

/// Two guards Whisper needs once its window is shorter than 30 s and its
/// drafts are short:
///
/// - **No timestamps past the window.** Timestamp `i` means `i` × 20 ms,
///   and the model knows 1500 of them; with an 8 s window only the first
///   401 name audio that exists. The rest are suppressed.
/// - **No repetition loops.** On a sliver of audio (a draft 400 ms into a
///   word, say) tiny likes to loop: `f. f. f. …` to the token cap, which is
///   seconds of CPU in a browser. When the sampled text tokens end in a
///   period-1..=24 pattern (a token up to a short sentence) repeated (5× for one token, 3× up to
///   four, 2× for longer phrases), only
///   `<|endoftext|>` is left. (The timestamp grammar applied after this
///   never suppresses `<|endoftext|>`.)
#[derive(Debug, Clone)]
pub struct LiveGuard {
    eot: i64,
    timestamp_begin: i64,
    max_timestamp: usize,
}

const MIN_REPEATS: usize = 3;
/// Longest repeating unit checked, in tokens: a short sentence.
const MAX_PERIOD: usize = 24;
/// A single token repeating needs more evidence ("no, no, no" is real).
const MIN_SINGLE_REPEATS: usize = 5;

impl LiveGuard {
    pub fn new(ids: &WhisperSpecialIds, window_positions: usize) -> Self {
        Self {
            eot: ids.eot,
            timestamp_begin: ids.timestamp_begin,
            max_timestamp: window_positions,
        }
    }

    fn looping(&self, sampled: &[i64]) -> bool {
        let text: Vec<i64> = sampled
            .iter()
            .copied()
            .filter(|&t| t < self.timestamp_begin)
            .collect();
        (1..=MAX_PERIOD).any(|period| {
            let need = match period {
                1 => MIN_SINGLE_REPEATS,
                2..=4 => MIN_REPEATS,
                // A phrase of five tokens or more, twice back to back in
                // one short window, is a loop far more often than speech.
                _ => 2,
            };
            let len = period * need;
            if text.len() < len {
                return false;
            }
            let tail = &text[text.len() - len..];
            tail.iter().enumerate().all(|(i, t)| *t == tail[i % period])
        })
    }
}

impl LogitFilter for LiveGuard {
    fn apply(&self, logits: Tensor<2>, tokens: &[Vec<i64>], prompt_len: usize) -> Tensor<2> {
        let [rows, vocab] = logits.dims();
        let first_late = self.timestamp_begin as usize + self.max_timestamp + 1;
        let mut mask = vec![false; rows * vocab];
        let mut any = false;
        for (r, seq) in tokens.iter().enumerate().take(rows) {
            let row = &mut mask[r * vocab..(r + 1) * vocab];
            if self.looping(seq.get(prompt_len..).unwrap_or(&[])) {
                row.iter_mut().for_each(|m| *m = true);
                row[self.eot as usize] = false;
                any = true;
            } else if first_late < vocab {
                row[first_late..].iter_mut().for_each(|m| *m = true);
                any = true;
            }
        }
        if !any {
            return logits;
        }
        let mask: Tensor<2, Bool> =
            Tensor::from_data(TensorData::new(mask, [rows, vocab]), &logits.device());
        logits.mask_fill(mask, f32::NEG_INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> LiveGuard {
        LiveGuard {
            eot: 50256,
            timestamp_begin: 50363,
            max_timestamp: 500,
        }
    }

    #[test]
    fn detects_loops() {
        let g = guard();
        assert!(g.looping(&[1, 2, 1, 2, 1, 2]));
        assert!(g.looping(&[9, 1, 2, 3, 1, 2, 3, 1, 2, 3]));
        assert!(!g.looping(&[1, 1, 1, 1]));
        assert!(g.looping(&[1, 1, 1, 1, 1]));
        // Timestamps between the repeats do not hide a loop.
        assert!(g.looping(&[1, 50400, 2, 1, 2, 50410, 1, 2]));
        assert!(!g.looping(&[1, 2, 3, 4, 5, 6]));
        // A sentence said three times in a row: "Thank you for listening."
        let s = [400, 345, 329, 8680, 13];
        assert!(g.looping(&[s, s, s].concat()));
        assert!(g.looping(&[s, s].concat()));
        // ...but a short phrase twice is speech: "thank you, thank you".
        assert!(!g.looping(&[1, 2, 1, 2]));
    }
}
