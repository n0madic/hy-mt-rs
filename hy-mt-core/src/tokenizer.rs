//! Thin wrapper over [`tokenizers::Tokenizer`] adding Hunyuan-MT 1.5 special
//! tokens and a translate-oriented chat template.
//!
//! Special tokens use **full-width vertical bars** `｜` (U+FF5C), not the
//! ASCII pipe — see the chat template embedded in the production GGUF:
//!
//! ```text
//! <｜hy_begin▁of▁sentence｜><｜hy_User｜>{user}<｜hy_Assistant｜>
//! ```
//!
//! Because the `tokenizers` crate's `encode(...)` byte-pair-merges the input
//! string and would split `<｜hy_User｜>` into many tokens, we look the
//! special tokens up by name and concatenate IDs directly.

use std::path::Path;

use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use crate::gguf::meta::Meta;
use crate::gguf::vendored::{Content, Value};
use crate::{Error, Result};

/// Conventional special-token IDs for Hunyuan-MT 1.5 (cross-checked against
/// the GGUF metadata and tokenizer.json `added_tokens` list).
pub const BOS_ID: u32 = 120_000;
pub const EOS_ID: u32 = 120_001;
/// Marker emitted by the assistant turn — also what the GGUF declares as
/// "EOS" and what we use to terminate generation.
pub const ASSISTANT_TURN_END: u32 = 120_020;
pub const PAD_ID: u32 = 120_002;

/// Special token strings (with full-width pipes) used by the chat template.
const TOK_BOS: &str = "<｜hy_begin▁of▁sentence｜>";
const TOK_USER: &str = "<｜hy_User｜>";
const TOK_ASSISTANT: &str = "<｜hy_Assistant｜>";
const TOK_SYSTEM_END: &str = "<｜hy_place▁holder▁no▁3｜>";
const TOK_TURN_END: &str = "<｜hy_place▁holder▁no▁2｜>";

/// Wrapper around `tokenizers::Tokenizer` for Hy-MT 1.5 models.
pub struct HyTokenizer {
    inner: Tokenizer,
    bos: u32,
    user: u32,
    assistant: u32,
    system_end: u32,
    turn_end: u32,
}

impl HyTokenizer {
    /// Load a `tokenizer.json` (HuggingFace fast-tokenizer format).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("from_file failed: {e}")))?;
        Ok(Self::with_inner(inner))
    }

    /// Build the tokenizer directly from data embedded in a GGUF file's
    /// metadata (no external `tokenizer.json` required). Supports the
    /// `gpt2` BPE flavour, which is what `tokenizer.ggml.model = "gpt2"`
    /// covers — the convention used by Hunyuan-MT 1.5 and most modern
    /// HF-derived GGUFs.
    pub fn from_gguf(content: &Content) -> Result<Self> {
        let meta = Meta::new(content);
        let model = meta.string("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(Error::Tokenizer(format!(
                "GGUF tokenizer model `{model}` not supported (expected `gpt2`)"
            )));
        }

        let tokens = read_string_array(&meta, "tokenizer.ggml.tokens")?;
        let merges_raw = read_string_array(&meta, "tokenizer.ggml.merges")?;
        let token_types = read_i32_array(&meta, "tokenizer.ggml.token_type")?;

        if tokens.len() != token_types.len() {
            return Err(Error::Tokenizer(format!(
                "tokens len {} != token_type len {}",
                tokens.len(),
                token_types.len()
            )));
        }

        // Parse "a b" merge lines into tuples.
        let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_raw.len());
        for line in &merges_raw {
            let mut parts = line.splitn(2, ' ');
            match (parts.next(), parts.next()) {
                (Some(a), Some(b)) => merges.push((a.to_string(), b.to_string())),
                _ => return Err(Error::Tokenizer(format!("malformed merge `{line}`"))),
            }
        }

        let vocab: Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| Error::Tokenizer(format!("BPE build failed: {e}")))?;

        let mut inner = Tokenizer::new(bpe);
        // Standard GPT-2 setup: byte-level pre-tokenizer / decoder /
        // post-processor. The same ByteLevel struct serves all three roles.
        inner.with_pre_tokenizer(Some(ByteLevel::default()));
        inner.with_decoder(Some(ByteLevel::default()));
        inner.with_post_processor(Some(ByteLevel::default()));

        // GGUF token-type values (mirroring llama.cpp's enum):
        //   1 NORMAL  2 UNKNOWN  3 CONTROL  4 USER_DEFINED  5 UNUSED  6 BYTE
        // Add CONTROL and USER_DEFINED entries as added tokens so the
        // tokenizer recognises them in raw input without splitting; CONTROL
        // ones get `special=true` so `decode(skip_special=true)` hides them.
        let added: Vec<AddedToken> = tokens
            .iter()
            .zip(token_types.iter())
            .filter_map(|(t, ty)| match *ty {
                3 => Some(AddedToken::from(t.clone(), true)),
                4 => Some(AddedToken::from(t.clone(), false)),
                _ => None,
            })
            .collect();
        if !added.is_empty() {
            inner.add_special_tokens(&added);
        }

        Ok(Self::with_inner(inner))
    }

    fn with_inner(inner: Tokenizer) -> Self {
        let lookup =
            |name: &str, fallback: u32| -> u32 { inner.token_to_id(name).unwrap_or(fallback) };
        Self {
            bos: lookup(TOK_BOS, BOS_ID),
            user: lookup(TOK_USER, 120_006),
            assistant: lookup(TOK_ASSISTANT, 120_007),
            system_end: lookup(TOK_SYSTEM_END, 120_021),
            turn_end: lookup(TOK_TURN_END, ASSISTANT_TURN_END),
            inner,
        }
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Token ID conventionally used to terminate generation in this model
    /// (the `<｜hy_place▁holder▁no▁2｜>` placeholder, also declared as EOS in
    /// the GGUF).
    pub fn turn_end_id(&self) -> u32 {
        self.turn_end
    }

    pub fn bos_id(&self) -> u32 {
        self.bos
    }

    pub fn assistant_id(&self) -> u32 {
        self.assistant
    }

    /// Encode a raw string with no special-token wrapping.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("encode failed: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token IDs back to a string, hiding special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(format!("decode failed: {e}")))
    }

    pub fn decode_with_special(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| Error::Tokenizer(format!("decode failed: {e}")))
    }

    /// Build the prompt token sequence for a single-turn user message,
    /// using the chat template embedded in the production GGUF:
    ///
    /// `<BOS><USER>{user}<ASSISTANT>`
    ///
    /// (Or `<BOS>{system}<SYSTEM_END><USER>{user}<ASSISTANT>` when a system
    /// message is provided.)
    pub fn apply_chat_template(&self, system: Option<&str>, user: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(8 + user.len() / 2);
        ids.push(self.bos);
        if let Some(sys) = system {
            ids.extend(self.encode_raw(sys)?);
            ids.push(self.system_end);
        }
        ids.push(self.user);
        ids.extend(self.encode_raw(user)?);
        ids.push(self.assistant);
        Ok(ids)
    }

    /// Build the standard "translate into <tgt>" prompt. The model
    /// auto-detects the source language, so no `src_lang` parameter is
    /// required.
    pub fn build_translate_prompt(&self, tgt_lang: &str, text: &str) -> Result<Vec<u32>> {
        let user = format!(
            "Translate the following segment into {tgt_lang}, without additional explanation.\n\n{text}",
        );
        self.apply_chat_template(None, &user)
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.inner.token_to_id(text)
    }
}

/// Read a typed homogeneous GGUF array, applying `extract` to each element.
/// Returns a clear, key-attributed error if any element has the wrong tag.
fn read_array<T, F>(
    meta: &Meta<'_>,
    key: &str,
    expected: &'static str,
    extract: F,
) -> Result<Vec<T>>
where
    F: Fn(&Value) -> Option<T>,
{
    let arr = meta.array(key)?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        match extract(v) {
            Some(t) => out.push(t),
            None => {
                return Err(Error::Tokenizer(format!(
                    "{key}: expected {expected} element, got {v:?}"
                )))
            }
        }
    }
    Ok(out)
}

fn read_string_array(meta: &Meta<'_>, key: &str) -> Result<Vec<String>> {
    read_array(meta, key, "String", |v| match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    })
}

fn read_i32_array(meta: &Meta<'_>, key: &str) -> Result<Vec<i32>> {
    read_array(meta, key, "I32", |v| match v {
        Value::I32(i) => Some(*i),
        _ => None,
    })
}
