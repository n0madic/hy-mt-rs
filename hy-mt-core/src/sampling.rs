//! Sampling: temperature, top-k, top-p, repeat-penalty.
//!
//! Operates on a flat `Vec<f32>` of logits. Token-history-aware repeat
//! penalty mirrors the convention used by `llama.cpp`: divide positive
//! logits by `penalty` and multiply negative ones for tokens recently
//! generated.

use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::Result;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            repeat_window: 64,
            seed: 0xCAFEBABE,
        }
    }
}

pub struct Sampler {
    pub params: SamplingParams,
    rng: StdRng,
    /// Reusable scratch buffer for the repeat-penalty seen-set. Indexed
    /// by token id; only the indices we touched in the current call
    /// (tracked by `seen_dirty`) need clearing on the next call.
    seen: Vec<bool>,
    seen_dirty: Vec<u32>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = StdRng::seed_from_u64(params.seed);
        Self {
            params,
            rng,
            seen: Vec::new(),
            seen_dirty: Vec::new(),
        }
    }

    /// Sample one token id from `logits`. `history` is the list of tokens
    /// generated so far — only the last [`SamplingParams::repeat_window`]
    /// entries are inspected for the repeat penalty.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> Result<u32> {
        // 1) Repeat penalty (apply once per unique token in the window).
        //    Use an amortized `Vec<bool>` allocated once per `Sampler`
        //    and only clear the indices touched on the previous call —
        //    avoids 120 KiB allocation per token on a 120 k vocab.
        if self.params.repeat_penalty > 1.0 {
            if self.seen.len() < logits.len() {
                self.seen.resize(logits.len(), false);
            }
            // Clear dirty entries from the previous call.
            for &idx in &self.seen_dirty {
                if (idx as usize) < self.seen.len() {
                    self.seen[idx as usize] = false;
                }
            }
            self.seen_dirty.clear();

            let start = history.len().saturating_sub(self.params.repeat_window);
            let recent = &history[start..];
            for &tok in recent {
                let idx = tok as usize;
                if idx >= logits.len() || self.seen[idx] {
                    continue;
                }
                self.seen[idx] = true;
                self.seen_dirty.push(tok);
                let l = logits[idx];
                logits[idx] = if l >= 0.0 {
                    l / self.params.repeat_penalty
                } else {
                    l * self.params.repeat_penalty
                };
            }
        }

        // 2) Greedy short-circuit when temperature == 0.
        if self.params.temperature <= 0.0 {
            return Ok(argmax(logits));
        }

        // 3) Temperature
        let inv_t = 1.0 / self.params.temperature;
        for l in logits.iter_mut() {
            *l *= inv_t;
        }

        // 4) Top-K filter — replace dropped logits with -inf so they survive
        //    the softmax with probability 0.
        if let Some(k) = self.params.top_k {
            top_k_filter(logits, k);
        }

        // 5) Convert to probabilities via numerically-stable softmax.
        let mut probs = softmax(logits);

        // 6) Top-P filter — keep the smallest set whose cumulative mass ≥ p.
        if let Some(p) = self.params.top_p {
            top_p_filter(&mut probs, p);
        }

        // 7) Multinomial sample.
        let dist = WeightedIndex::new(&probs)
            .map_err(|e| crate::Error::Sampling(format!("WeightedIndex: {e}")))?;
        Ok(dist.sample(&mut self.rng) as u32)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    // `>` returns false for any NaN comparison, so NaN logits are
    // skipped and `best_idx` keeps the running max. If all logits are
    // NaN we end up with index 0 — degenerate but bounded, no panic.
    let mut best_idx = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // Use a quickselect (`select_nth_unstable_by`) over a clone instead of
    // a full sort: O(V) on a 120k-vocab vs O(V log V), saving ≈ 1-2 ms
    // per token at the model's vocabulary size. We need a clone because
    // quickselect mutates the slice it operates on.
    let mut buf: Vec<f32> = logits.to_vec();
    let (_, kth, _) = buf.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = *kth;
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs
}

fn top_p_filter(probs: &mut [f32], p: f32) {
    // Sort only non-zero probabilities — when top_k has already pruned the
    // tail to zeros (via `f32::NEG_INFINITY` → softmax → 0), nucleus
    // becomes effectively `O(k log k)` instead of `O(V log V)`.
    let mut order: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
    order.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cum = 0.0f32;
    let mut last_keep = 0;
    for (rank, &idx) in order.iter().enumerate() {
        cum += probs[idx];
        last_keep = rank;
        if cum >= p {
            break;
        }
    }
    // Zero everything ranked beyond `last_keep`, then renormalize.
    for &idx in order.iter().skip(last_keep + 1) {
        probs[idx] = 0.0;
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![1.0, 0.5, 3.0, -0.2];
        let tok = s.sample(&mut logits, &[]).unwrap();
        assert_eq!(tok, 2);
    }

    #[test]
    fn argmax_with_nan_logits_does_not_panic() {
        // `argmax` uses `>` so NaN comparisons return false and the
        // current best stays put. All-NaN input degenerates to index 0
        // — bounded but unhelpful; the important thing is no panic.
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![f32::NAN, 2.5, f32::NAN, 1.0];
        let tok = s.sample(&mut logits, &[]).unwrap();
        // Token 1 has the only finite max (2.5) and must be picked.
        assert_eq!(tok, 1);

        let mut all_nan = vec![f32::NAN; 4];
        let tok = s.sample(&mut all_nan, &[]).unwrap();
        assert!(tok < 4, "all-NaN should still produce a bounded id");
    }

    #[test]
    fn repeat_penalty_demotes_recent_tokens() {
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![1.0, 0.5, 3.0, -0.2];
        let tok = s.sample(&mut logits, &[2, 2, 2]).unwrap();
        // Token 2 had logit 3.0; even after /2 it becomes 1.5 (still above 1.0).
        assert_eq!(tok, 2);

        let mut logits = vec![1.0, 0.5, 1.4, -0.2];
        let params2 = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params2);
        let tok = s.sample(&mut logits, &[2]).unwrap();
        // 1.4 / 2 = 0.7 → now lower than 1.0 at index 0.
        assert_eq!(tok, 0);
    }
}
