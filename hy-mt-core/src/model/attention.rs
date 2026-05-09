//! Attention block: Q/K/V projections, optional QK-RMSNorm, RoPE, GQA,
//! KV-cache, and causal scaled dot-product attention.

use candle_core::Tensor;

use super::kv_cache::KvCache;
use super::linear::QuantLinear;
use super::rms_norm::RmsNormPerHead;
use crate::rope::RopeCache;
use crate::Result;

#[derive(Clone)]
pub struct Attention {
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    q_norm: Option<RmsNormPerHead>,
    k_norm: Option<RmsNormPerHead>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_groups: usize,
    rope: RopeCache,
}

#[allow(clippy::too_many_arguments)]
impl Attention {
    pub fn new(
        q_proj: QuantLinear,
        k_proj: QuantLinear,
        v_proj: QuantLinear,
        o_proj: QuantLinear,
        q_norm: Option<RmsNormPerHead>,
        k_norm: Option<RmsNormPerHead>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: RopeCache,
    ) -> Self {
        debug_assert!(
            n_kv_heads != 0 && n_heads % n_kv_heads == 0,
            "n_heads ({n_heads}) must be a positive multiple of n_kv_heads ({n_kv_heads}) for GQA"
        );
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_groups: n_heads / n_kv_heads,
            rope,
        }
    }

    /// Forward pass through the attention block.
    ///
    /// - `x`: `[B, T, hidden]`
    /// - `pos_offset`: absolute position of the first query token (= number of
    ///   tokens already in the KV cache before this call)
    /// - `cache`: per-layer KV cache (mutated in place)
    pub fn forward(&self, x: &Tensor, pos_offset: usize, cache: &mut KvCache) -> Result<Tensor> {
        let (b, t, _h) = x.dims3()?;

        // Q/K/V projections: [B, T, n_heads*head_dim] / [B, T, n_kv_heads*head_dim]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, T, H, D].
        let q = q.reshape((b, t, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b, t, self.n_kv_heads, self.head_dim))?;

        // Optional QK-RMSNorm (per head, weight has length head_dim).
        let q = match &self.q_norm {
            Some(n) => n.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => n.forward(&k)?,
            None => k,
        };

        // Transpose to [B, H, T, D] for RoPE/attention.
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Rotary embedding.
        let q = self.rope.apply(&q, pos_offset)?;
        let k = self.rope.apply(&k, pos_offset)?;

        // Append to KV cache and obtain the full K/V covering all past + new positions.
        let (k_full, v_full) = cache.append(&k, &v)?;

        // GQA: repeat K/V along the head dim to match Q.
        let k_full = repeat_kv(&k_full, self.kv_groups)?;
        let v_full = repeat_kv(&v_full, self.kv_groups)?;

        // Scaled dot-product attention.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k_full.transpose(2, 3)?)?;
        let scores = (scores * scale)?;

        // Causal mask: query at position `pos_offset + i` may only attend to
        // keys at positions `[0 .. pos_offset + i]`. We always mask in F32 to
        // keep numerical stability before the softmax.
        let masked = if t == 1 {
            // Pure decode step: every key is in the past — no mask needed.
            scores
        } else {
            let kv_len = pos_offset + t;
            let mask = build_causal_mask(t, kv_len, pos_offset, scores.device())?
                .to_dtype(scores.dtype())?;
            let mask = mask.broadcast_as(scores.shape())?;
            scores.broadcast_add(&mask)?
        };

        let weights = candle_nn::ops::softmax_last_dim(&masked)?;
        let attn = weights.matmul(&v_full.to_dtype(weights.dtype())?)?;

        // [B, H, T, D] → [B, T, H, D] → [B, T, hidden]
        let y =
            attn.transpose(1, 2)?
                .contiguous()?
                .reshape((b, t, self.n_heads * self.head_dim))?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

/// Repeat each KV head `groups` times along the head axis (axis 1) so that
/// `n_kv_heads * groups == n_heads`. Matches PyTorch's `repeat_kv` from the
/// reference HF Llama implementation.
fn repeat_kv(t: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(t.clone());
    }
    let (b, h_kv, seq, d) = t.dims4()?;
    let expanded = t
        .unsqueeze(2)?
        .expand((b, h_kv, groups, seq, d))?
        .reshape((b, h_kv * groups, seq, d))?;
    Ok(expanded)
}

/// Build an additive causal mask of shape `[1, 1, t, kv_len]` where
/// positions strictly past the query receive `-inf`. The mask broadcasts
/// across the batch and head dimensions.
fn build_causal_mask(
    t: usize,
    kv_len: usize,
    pos_offset: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * kv_len];
    for q in 0..t {
        let allowed_end = pos_offset + q + 1;
        for k in allowed_end..kv_len {
            data[q * kv_len + k] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(data, (1, 1, t, kv_len), device)?;
    Ok(mask)
}
