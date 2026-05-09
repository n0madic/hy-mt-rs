//! Rotary position embedding.
//!
//! Hunyuan-MT 1.5 keeps the HF "rotate-half" convention in GGUF (the
//! converter does **not** permute Q/K weights for the `hunyuan-dense` arch),
//! so we apply rotation as `(x[..d/2], x[d/2..])` rather than the
//! interleaved GPT-J style. This matters: the interleaved convention
//! works on most tokens of most languages but pushes argmax to nearby —
//! often wrong — subword candidates on morphologically rich languages
//! such as Russian. Empirically, switching to `rope` (rotate-half) restored
//! correct Russian translations end-to-end.
//!
//! `cos`/`sin` are computed **fresh on every `apply` call** rather than
//! pre-computed up to a fixed capacity. The kept state is just the
//! `head_dim/2`-element `inv_freq` vector. This removes any artificial
//! `max_seq_len` cap (the model claims a 262 144-token context) and adds
//! only microseconds per forward step — pre-computation isn't worth its
//! memory budget at megabyte scale, and a growable cache would put a
//! `Mutex` on the attention hot path.
//!
//! NTK-aware base rescaling (`new_base = base * alpha^(d/(d-2))`) is
//! folded into `rope_theta` at config-load time, not here. The production
//! GGUF stores the already-rescaled base (`11_158_840`) in metadata;
//! `HunyuanConfig::from_hf_config` applies the same formula to the raw
//! `10_000` base when reading HF safetensors. By the time `RopeCache::new`
//! sees `rope_theta`, both paths have an identical effective value, so
//! the cache has nothing scaling-related to handle itself.

use candle_core::{DType, Device, Tensor};

use crate::{Error, Result};

#[derive(Clone)]
pub struct RopeCache {
    inv_freq: Tensor,
    head_dim: usize,
}

impl RopeCache {
    /// Build the `inv_freq` table for the given head dimension and base.
    /// `head_dim` must be even.
    pub fn new(head_dim: usize, theta: f32, device: &Device) -> Result<Self> {
        if head_dim % 2 != 0 {
            return Err(Error::Gguf(format!(
                "head_dim {head_dim} must be even for RoPE"
            )));
        }
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv, (1, half), device)?.to_dtype(DType::F32)?;
        Ok(Self { inv_freq, head_dim })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Apply rotate-half RoPE to a `[B, H, T, head_dim]` tensor starting
    /// from absolute position `pos_offset`. Output dtype matches the input.
    pub fn apply(&self, x: &Tensor, pos_offset: usize) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 4 {
            return Err(Error::Gguf(format!(
                "RoPE expects rank 4 input, got shape {dims:?}"
            )));
        }
        let seq_len = dims[2];
        let last = dims[3];
        if last != self.head_dim {
            return Err(Error::Gguf(format!(
                "RoPE input last dim {last} != head_dim {}",
                self.head_dim
            )));
        }

        // Build positions [pos_offset, pos_offset + seq_len) on the same
        // device as `inv_freq`, then outer-product into `freqs`.
        let positions: Vec<f32> = (pos_offset..pos_offset + seq_len)
            .map(|p| p as f32)
            .collect();
        let positions = Tensor::from_vec(positions, (seq_len, 1), self.inv_freq.device())?;
        let freqs = positions.broadcast_mul(&self.inv_freq)?;

        let x_in = x.contiguous()?;
        let cos = freqs.cos()?.to_dtype(x_in.dtype())?;
        let sin = freqs.sin()?.to_dtype(x_in.dtype())?;
        let y = candle_nn::rotary_emb::rope(&x_in, &cos, &sin)?;
        Ok(y)
    }
}
