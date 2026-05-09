//! RMSNorm and a per-head variant used for QK normalization in Hunyuan-MT 1.5.
//!
//! Both delegate to `candle_nn::ops::rms_norm`, which handles broadcasting
//! the 1-D `weight` against the input's last dimension and runs the inner
//! variance reduction in F32 internally for numerical stability.

use candle_core::Tensor;

use crate::Result;

/// Standard RMSNorm: `y = x * weight / sqrt(mean(x^2) + eps)`.
///
/// `weight` is a 1-D tensor of length `hidden_size`.
#[derive(Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply normalization. Input shape `[..., hidden_size]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(x, &self.weight, self.eps)?)
    }
}

/// Per-head RMSNorm used by `use_qk_norm = true`. The input is
/// `[B, H, T, head_dim]` (or `[B, T, H, head_dim]` — both are accepted as
/// long as the *last* dim is `head_dim`); normalization runs over the last
/// dimension and the learnable scale has length `head_dim` and is broadcast
/// across all heads / positions / batches.
#[derive(Clone)]
pub struct RmsNormPerHead {
    weight: Tensor,
    eps: f32,
}

impl RmsNormPerHead {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Same op as `RmsNorm` — `rms_norm` already broadcasts a 1-D
        // `[head_dim]` weight across the leading `[B, H, T]` axes.
        Ok(candle_nn::ops::rms_norm(x, &self.weight, self.eps)?)
    }
}
