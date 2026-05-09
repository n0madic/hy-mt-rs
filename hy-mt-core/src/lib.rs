//! Rust+Candle inference for Tencent Hunyuan-MT 1.5 models stored in the
//! 1.25-bit STQ1_0 GGUF format introduced by `ggml-org/llama.cpp` PR #22836.
//!
//! The crate provides:
//! - the `STQ1_0` codec (block layout, codebook, dequantization, custom matmul);
//! - a thin GGUF loader vendored from `candle-core::quantized::gguf_file`
//!   extended with the new `Stq1_0 = 42` ggml type;
//! - the `HunyuanDense` decoder transformer with GQA, QK-norm and dynamic-NTK
//!   RoPE, plus a token-streaming generator backed by the `tokenizers` crate.
//!
//! Both CPU (in-place STQ1_0 matmul) and Metal (eager dequantization to F16)
//! execution paths are supported via Cargo features.

pub mod device;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod hub;
pub mod model;
pub mod quant;
pub mod rope;
pub mod safetensors_loader;
pub mod sampling;
pub mod source;
pub mod tokenizer;
pub(crate) mod util;
pub mod weights;

pub use error::{Error, LoadingKind, Result};
pub use source::ModelSource;
