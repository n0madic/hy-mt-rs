//! Token-level generator: per-token prefill, KV-cache-driven decode, and a
//! pluggable sampler.

use candle_core::{DType, Tensor};

use crate::model::HunyuanDense;
use crate::sampling::{Sampler, SamplingParams};
use crate::tokenizer::HyTokenizer;
use crate::{Error, Result};

/// One-shot generator that owns the model, tokenizer and sampler.
pub struct Generator<'a> {
    pub model: &'a mut HunyuanDense,
    pub tokenizer: &'a HyTokenizer,
    pub sampler: Sampler,
    pub max_new_tokens: usize,
}

/// Decoded streaming step. `text` may be empty for tokens that don't form a
/// complete UTF-8 boundary by themselves; callers should accumulate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Step {
    pub token: u32,
    pub text: String,
    pub finished: bool,
}

impl<'a> Generator<'a> {
    pub fn new(
        model: &'a mut HunyuanDense,
        tokenizer: &'a HyTokenizer,
        params: SamplingParams,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            sampler: Sampler::new(params),
            max_new_tokens,
        }
    }

    /// Encode + apply chat template + run prefill + sample new tokens until
    /// EOS or `max_new_tokens` is reached.
    ///
    /// `on_step` is invoked once per emitted token with a streaming-friendly
    /// `Step`. Returning `false` from the callback aborts generation.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        mut on_step: impl FnMut(&Step) -> bool,
    ) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Err(Error::Tokenizer("empty prompt".into()));
        }
        // Bound the *prompt* against the model's positional budget — that
        // is the physical limit (RoPE table, embedding lookup). Then bound
        // total = prompt + new tokens against the same budget but capped
        // (so a 10k prompt with `max_new_tokens=5` is accepted: it only
        // needs 10005 of the 262144 declared positions).
        let max_ctx = self.model.config.max_position_embeddings;
        if max_ctx == 0 {
            return Err(Error::Validation(
                "model max_position_embeddings is 0; refusing to run".into(),
            ));
        }
        if prompt_tokens.len() > max_ctx {
            return Err(Error::OverLimit {
                what: "prompt length",
                got: prompt_tokens.len() as u64,
                max: max_ctx as u64,
            });
        }
        let total = prompt_tokens
            .len()
            .saturating_add(self.max_new_tokens)
            .min(max_ctx);

        // Capacity == prompt + max new tokens; bounded above by the model
        // context length we just verified.
        self.model.reset_kv_cache(total)?;
        let device = self.model.device.device.clone();
        // The Hunyuan-MT 1.5 chat template terminates on the
        // `assistant_turn_end` token (config.eos_id = 120_020 in the
        // production GGUF), but defense-in-depth also stops on the raw
        // EOS_ID (120_001) — emitted on rare degenerate completions.
        let eos = self.model.config.eos_id;
        let alt_eos = crate::tokenizer::EOS_ID;

        // ---- prefill -----------------------------------------------------
        // Run the entire prompt through the model in one batched forward.
        // `HunyuanDense::forward` already returns logits for the last position
        // only, which is exactly what we need to seed sampling. The KV cache
        // ends up holding all prompt tokens after this single call.
        let mut history: Vec<u32> = prompt_tokens.to_vec();
        let prompt_tensor =
            Tensor::from_vec(prompt_tokens.to_vec(), (1, prompt_tokens.len()), &device)?;
        let mut logits = self
            .model
            .forward(&prompt_tensor)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // ---- decode ------------------------------------------------------
        // Track text already surfaced via `Step.text` so we can stream
        // only the newly-completed UTF-8 tail per token (see M3 below).
        let mut emitted_text = self.tokenizer.decode(&history).unwrap_or_default();
        let mut produced = Vec::with_capacity(self.max_new_tokens);
        for _ in 0..self.max_new_tokens {
            let next = self.sampler.sample(&mut logits, &history)?;
            history.push(next);

            let finished = next == eos || next == alt_eos;
            // Incremental UTF-8-safe decode: byte-level BPE can split a
            // multi-byte glyph (Cyrillic, CJK, …) across token boundaries,
            // so single-token `decode(&[next])` returns garbage. We
            // decode the full `history`, diff against the previously
            // emitted text, and surface only the new tail.
            //
            // Some tokenizers (those with normalisers like NFC/NFD) can
            // *rewrite* earlier output when a later token combines with
            // it — so the new `full` is not always a strict prefix of
            // `emitted_text`. We compute the longest common char-aligned
            // prefix in that case, snap to the closest UTF-8 boundary,
            // and always advance `emitted_text` so the next step
            // compares against the current state.
            let full = self.tokenizer.decode(&history).unwrap_or_default();
            let text = compute_emitted_tail(&emitted_text, &full);
            emitted_text = full;
            let step = Step {
                token: next,
                text,
                finished,
            };
            let cont = on_step(&step);
            produced.push(next);
            if finished || !cont {
                break;
            }

            let next_t = Tensor::from_vec(vec![next], (1, 1), &device)?;
            logits = self
                .model
                .forward(&next_t)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
        }

        Ok(produced)
    }

    /// Sync-friendly wrapper that returns the decoded translation. The
    /// model auto-detects the source language; only the target is needed.
    pub fn translate(&mut self, tgt_lang: &str, text: &str) -> Result<String> {
        let prompt = self.tokenizer.build_translate_prompt(tgt_lang, text)?;
        let ids = self.generate(&prompt, |_| true)?;
        let text = self.tokenizer.decode(&ids)?;
        Ok(text)
    }
}

/// Compute the streaming tail to surface in `Step.text` when the cumulative
/// `decode(history)` advances from `prev` to `full`. Handles the common
/// monotonic case (`prev` is a prefix of `full`, return the suffix) and the
/// degenerate case where the tokenizer rewrites earlier output (return the
/// suffix from the longest common UTF-8-aligned prefix).
fn compute_emitted_tail(prev: &str, full: &str) -> String {
    if let Some(stripped) = full.strip_prefix(prev) {
        return stripped.to_string();
    }
    let common = prev
        .as_bytes()
        .iter()
        .zip(full.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    // Snap `common` down to the nearest char boundary in `full` so we
    // never split a multi-byte sequence.
    let mut boundary = common;
    while boundary > 0 && !full.is_char_boundary(boundary) {
        boundary -= 1;
    }
    full[boundary..].to_string()
}

#[cfg(test)]
mod tests {
    use super::compute_emitted_tail;

    #[test]
    fn prefix_case_returns_clean_tail() {
        assert_eq!(compute_emitted_tail("Hello", "Hello, world"), ", world");
        assert_eq!(compute_emitted_tail("", "abc"), "abc");
        assert_eq!(compute_emitted_tail("abc", "abc"), "");
    }

    #[test]
    fn rewrite_case_falls_back_to_lcp() {
        // Simulate a tokenizer that rewrote "ab" → "axyz".
        let tail = compute_emitted_tail("abc", "axyz");
        assert_eq!(tail, "xyz");
    }

    #[test]
    fn lcp_snaps_to_char_boundary() {
        // "Привет" — first byte differs in the middle of a Cyrillic glyph.
        // "Прquit" diverges at the 'q' (byte index 4 — middle of 'и').
        // The byte-LCP is 4 but that's not a char boundary; must snap to 2
        // (after "Пр").
        let tail = compute_emitted_tail("Привет", "Прquit");
        // The snapped boundary in `full` is 2 ("Пр"), so the tail is "quit".
        assert_eq!(tail, "quit");
    }
}
