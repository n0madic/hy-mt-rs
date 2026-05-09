//! Format-agnostic tensor naming for the Hunyuan-MT 1.5 architecture.
//!
//! Both the GGUF and HuggingFace safetensors layouts describe the same
//! 32-layer transformer; only the on-disk tensor *names* differ. This
//! module owns the canonical [`TensorRole`] enum plus parsers for each
//! naming convention, so loaders can convert names to roles (and back)
//! without leaking a string-based dispatch into the model code.

use std::fmt;

/// Per-block slot inside a single decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockSlot {
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnQNorm,
    AttnKNorm,
    AttnOutput,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
}

/// Semantic role of a tensor in the model. Loaders translate to/from
/// format-specific names through [`gguf_name`], [`hf_name`] and
/// [`from_gguf_name`] / [`from_hf_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    TokenEmbedding,
    OutputNorm,
    Output,
    Block { idx: usize, slot: BlockSlot },
}

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenEmbedding => f.write_str("token_embd"),
            Self::OutputNorm => f.write_str("output_norm"),
            Self::Output => f.write_str("output"),
            Self::Block { idx, slot } => write!(f, "blk.{idx}.{slot:?}"),
        }
    }
}

impl TensorRole {
    /// Iterate every role that should be present in a `n_layers`-deep model.
    /// `tied_lm_head = true` omits [`TensorRole::Output`].
    pub fn enumerate(n_layers: usize, tied_lm_head: bool) -> Vec<Self> {
        let mut out = vec![Self::TokenEmbedding, Self::OutputNorm];
        if !tied_lm_head {
            out.push(Self::Output);
        }
        for idx in 0..n_layers {
            for slot in [
                BlockSlot::AttnNorm,
                BlockSlot::AttnQ,
                BlockSlot::AttnK,
                BlockSlot::AttnV,
                BlockSlot::AttnQNorm,
                BlockSlot::AttnKNorm,
                BlockSlot::AttnOutput,
                BlockSlot::FfnNorm,
                BlockSlot::FfnGate,
                BlockSlot::FfnUp,
                BlockSlot::FfnDown,
            ] {
                out.push(Self::Block { idx, slot });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// GGUF naming (`token_embd.weight`, `blk.N.attn_q.weight`, …)
// ---------------------------------------------------------------------------

/// Convert a [`TensorRole`] to its GGUF tensor name (without the
/// `.weight` suffix, which is universal for this architecture).
pub fn gguf_name(role: TensorRole) -> String {
    match role {
        TensorRole::TokenEmbedding => "token_embd.weight".into(),
        TensorRole::OutputNorm => "output_norm.weight".into(),
        TensorRole::Output => "output.weight".into(),
        TensorRole::Block { idx, slot } => {
            let stem = match slot {
                BlockSlot::AttnNorm => "attn_norm",
                BlockSlot::AttnQ => "attn_q",
                BlockSlot::AttnK => "attn_k",
                BlockSlot::AttnV => "attn_v",
                BlockSlot::AttnQNorm => "attn_q_norm",
                BlockSlot::AttnKNorm => "attn_k_norm",
                BlockSlot::AttnOutput => "attn_output",
                BlockSlot::FfnNorm => "ffn_norm",
                BlockSlot::FfnGate => "ffn_gate",
                BlockSlot::FfnUp => "ffn_up",
                BlockSlot::FfnDown => "ffn_down",
            };
            format!("blk.{idx}.{stem}.weight")
        }
    }
}

/// Inverse of [`gguf_name`]; returns `None` for unknown names.
pub fn from_gguf_name(name: &str) -> Option<TensorRole> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    match stem {
        "token_embd" => return Some(TensorRole::TokenEmbedding),
        "output_norm" => return Some(TensorRole::OutputNorm),
        "output" => return Some(TensorRole::Output),
        _ => {}
    }
    let rest = stem.strip_prefix("blk.")?;
    let mut parts = rest.splitn(2, '.');
    let idx: usize = parts.next()?.parse().ok()?;
    let slot = match parts.next()? {
        "attn_norm" => BlockSlot::AttnNorm,
        "attn_q" => BlockSlot::AttnQ,
        "attn_k" => BlockSlot::AttnK,
        "attn_v" => BlockSlot::AttnV,
        "attn_q_norm" => BlockSlot::AttnQNorm,
        "attn_k_norm" => BlockSlot::AttnKNorm,
        "attn_output" => BlockSlot::AttnOutput,
        "ffn_norm" => BlockSlot::FfnNorm,
        "ffn_gate" => BlockSlot::FfnGate,
        "ffn_up" => BlockSlot::FfnUp,
        "ffn_down" => BlockSlot::FfnDown,
        _ => return None,
    };
    Some(TensorRole::Block { idx, slot })
}

// ---------------------------------------------------------------------------
// HuggingFace naming (`model.embed_tokens.weight`,
//                    `model.layers.N.self_attn.q_proj.weight`, …)
// ---------------------------------------------------------------------------

/// Convert a [`TensorRole`] to its HuggingFace tensor name.
///
/// `HunYuanDenseV1ForCausalLM` (the AngelSlim/tencent base) uses
/// `query_layernorm`/`key_layernorm` for QK-norm (not the more common
/// `q_norm`/`k_norm`); both forms are accepted on input by [`from_hf_name`].
pub fn hf_name(role: TensorRole) -> String {
    match role {
        TensorRole::TokenEmbedding => "model.embed_tokens.weight".into(),
        TensorRole::OutputNorm => "model.norm.weight".into(),
        TensorRole::Output => "lm_head.weight".into(),
        TensorRole::Block { idx, slot } => {
            let suffix = match slot {
                BlockSlot::AttnNorm => "input_layernorm.weight",
                BlockSlot::AttnQ => "self_attn.q_proj.weight",
                BlockSlot::AttnK => "self_attn.k_proj.weight",
                BlockSlot::AttnV => "self_attn.v_proj.weight",
                BlockSlot::AttnQNorm => "self_attn.query_layernorm.weight",
                BlockSlot::AttnKNorm => "self_attn.key_layernorm.weight",
                BlockSlot::AttnOutput => "self_attn.o_proj.weight",
                BlockSlot::FfnNorm => "post_attention_layernorm.weight",
                BlockSlot::FfnGate => "mlp.gate_proj.weight",
                BlockSlot::FfnUp => "mlp.up_proj.weight",
                BlockSlot::FfnDown => "mlp.down_proj.weight",
            };
            format!("model.layers.{idx}.{suffix}")
        }
    }
}

/// Inverse of [`hf_name`]; returns `None` for unknown names.
///
/// Accepts either `query_layernorm`/`key_layernorm` (Hunyuan-MT 1.5 names)
/// or the shorter `q_norm`/`k_norm` (used by some other models) so that
/// fine-tuned variants don't trip the loader.
pub fn from_hf_name(name: &str) -> Option<TensorRole> {
    match name {
        "model.embed_tokens.weight" => return Some(TensorRole::TokenEmbedding),
        "model.norm.weight" => return Some(TensorRole::OutputNorm),
        "lm_head.weight" => return Some(TensorRole::Output),
        _ => {}
    }
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let (idx_str, after) = rest.split_at(dot);
    let idx: usize = idx_str.parse().ok()?;
    let suffix = &after[1..];
    let slot = match suffix {
        "input_layernorm.weight" => BlockSlot::AttnNorm,
        "self_attn.q_proj.weight" => BlockSlot::AttnQ,
        "self_attn.k_proj.weight" => BlockSlot::AttnK,
        "self_attn.v_proj.weight" => BlockSlot::AttnV,
        "self_attn.query_layernorm.weight" | "self_attn.q_norm.weight" => BlockSlot::AttnQNorm,
        "self_attn.key_layernorm.weight" | "self_attn.k_norm.weight" => BlockSlot::AttnKNorm,
        "self_attn.o_proj.weight" => BlockSlot::AttnOutput,
        "post_attention_layernorm.weight" => BlockSlot::FfnNorm,
        "mlp.gate_proj.weight" => BlockSlot::FfnGate,
        "mlp.up_proj.weight" => BlockSlot::FfnUp,
        "mlp.down_proj.weight" => BlockSlot::FfnDown,
        _ => return None,
    };
    Some(TensorRole::Block { idx, slot })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(role: TensorRole) {
        let g = gguf_name(role);
        assert_eq!(from_gguf_name(&g), Some(role), "GGUF round-trip {g:?}");
        let h = hf_name(role);
        assert_eq!(from_hf_name(&h), Some(role), "HF round-trip {h:?}");
    }

    #[test]
    fn round_trip_top_level_roles() {
        for r in [
            TensorRole::TokenEmbedding,
            TensorRole::OutputNorm,
            TensorRole::Output,
        ] {
            round_trip(r);
        }
    }

    #[test]
    fn round_trip_per_block_slots() {
        for &slot in &[
            BlockSlot::AttnNorm,
            BlockSlot::AttnQ,
            BlockSlot::AttnK,
            BlockSlot::AttnV,
            BlockSlot::AttnQNorm,
            BlockSlot::AttnKNorm,
            BlockSlot::AttnOutput,
            BlockSlot::FfnNorm,
            BlockSlot::FfnGate,
            BlockSlot::FfnUp,
            BlockSlot::FfnDown,
        ] {
            for idx in [0usize, 1, 7, 31] {
                round_trip(TensorRole::Block { idx, slot });
            }
        }
    }

    #[test]
    fn enumerate_counts_match() {
        let roles = TensorRole::enumerate(32, true);
        // 2 top-level (token_embd + output_norm) + 32*11 block slots.
        assert_eq!(roles.len(), 2 + 32 * 11);

        let with_lm_head = TensorRole::enumerate(2, false);
        assert_eq!(with_lm_head.len(), 3 + 2 * 11);
        assert!(with_lm_head.contains(&TensorRole::Output));
    }
}
