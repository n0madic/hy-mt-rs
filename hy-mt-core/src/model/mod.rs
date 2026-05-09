//! Hunyuan-MT 1.5 dense decoder transformer.
//!
//! Architecture summary (`HunYuanDenseV1ForCausalLM`):
//! - 32 transformer layers with pre-norm residual scheme
//! - GQA: 16 query heads, 4 key/value heads, head_dim 128
//! - RMSNorm (eps 1e-5) including QK-norm on the per-head Q and K
//! - SwiGLU FFN (`intermediate_size = 6144`)
//! - Rotary positional embedding, base 10000, dynamic-NTK scaling
//! - Tied input/output embeddings (`tie_word_embeddings = true`)

mod attention;
pub mod config;
mod ffn;
mod kv_cache;
mod layer;
pub mod layout;
mod linear;
mod rms_norm;
mod transformer;

pub use config::HunyuanConfig;
pub use kv_cache::KvCache;
pub use layout::{BlockSlot, TensorRole};
pub use linear::QuantLinear;
pub use rms_norm::{RmsNorm, RmsNormPerHead};
pub use transformer::HunyuanDense;
