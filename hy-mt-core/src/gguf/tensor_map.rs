//! GGUF-specific name mapping is now a thin wrapper around the
//! format-agnostic [`crate::model::layout`].

pub use crate::model::layout::{BlockSlot, TensorRole};

use crate::model::layout::from_gguf_name;

/// Parse a GGUF tensor name into a [`TensorRole`], returning `None` for
/// unknown names so the caller can decide whether to log/skip/error.
pub fn parse_gguf_name(name: &str) -> Option<TensorRole> {
    from_gguf_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_names() {
        assert_eq!(
            parse_gguf_name("token_embd.weight"),
            Some(TensorRole::TokenEmbedding)
        );
        assert_eq!(
            parse_gguf_name("output_norm.weight"),
            Some(TensorRole::OutputNorm)
        );
        assert_eq!(parse_gguf_name("output.weight"), Some(TensorRole::Output));
        assert_eq!(
            parse_gguf_name("blk.0.attn_q.weight"),
            Some(TensorRole::Block {
                idx: 0,
                slot: BlockSlot::AttnQ
            })
        );
        assert_eq!(
            parse_gguf_name("blk.31.ffn_down.weight"),
            Some(TensorRole::Block {
                idx: 31,
                slot: BlockSlot::FfnDown
            })
        );
    }

    #[test]
    fn unknown_names_become_none() {
        assert_eq!(parse_gguf_name("foo"), None);
        assert_eq!(parse_gguf_name("blk.notnum.attn_q"), None);
        assert_eq!(parse_gguf_name("blk.0.unknown_slot"), None);
    }
}
