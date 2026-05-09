//! SwiGLU feed-forward block.
//!
//! `down(silu(gate(x)) * up(x))` — exactly the pattern used by Llama / Mistral
//! / Hunyuan's dense FFN.

use candle_core::Tensor;

use super::linear::QuantLinear;
use crate::Result;

#[derive(Clone)]
pub struct SwiGluFfn {
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl SwiGluFfn {
    pub fn new(gate: QuantLinear, up: QuantLinear, down: QuantLinear) -> Self {
        Self { gate, up, down }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let g = candle_nn::ops::silu(&g)?;
        let act = g.mul(&u)?;
        let y = self.down.forward(&act)?;
        Ok(y)
    }
}
