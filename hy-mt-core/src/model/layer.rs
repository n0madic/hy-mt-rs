//! One Hunyuan-MT 1.5 decoder layer (pre-norm residual scheme).

use candle_core::Tensor;

use super::attention::Attention;
use super::ffn::SwiGluFfn;
use super::kv_cache::KvCache;
use super::rms_norm::RmsNorm;
use crate::Result;

#[derive(Clone)]
pub struct DecoderLayer {
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub ffn_norm: RmsNorm,
    pub ffn: SwiGluFfn,
}

impl DecoderLayer {
    pub fn forward(&self, x: &Tensor, pos_offset: usize, cache: &mut KvCache) -> Result<Tensor> {
        // Pre-norm attention.
        let h = self.attn_norm.forward(x)?;
        let h = self.attn.forward(&h, pos_offset, cache)?;
        let x = x.add(&h)?;

        // Pre-norm FFN.
        let h = self.ffn_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        let x = x.add(&h)?;

        Ok(x)
    }
}
