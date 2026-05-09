//! STQ1_0 (Sherry Ternary Quant, 1.25 bpw) codec.
//!
//! Ported from the reference implementation in `ggml-org/llama.cpp` PR #22836
//! (`ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c`).

mod block;
mod codebook;
mod dequant;
mod matmul;
#[cfg(target_arch = "x86_64")]
mod matmul_avx2;
#[cfg(target_arch = "aarch64")]
mod matmul_neon;
mod q8;
mod quantize;

pub use block::{BlockStq1_0, BLOCK_BYTES, QK_K};
pub use codebook::{STQ1_0_CODEBOOK, STQ1_0_LUT_F32, STQ1_0_QPACK_TO_SIGN, STQ1_0_QPACK_TO_SLOT};
pub use dequant::{dequantize_row_stq1_0_f16, dequantize_row_stq1_0_f32};
pub use matmul::{stq_matmul_f32, stq_matvec_f32};
pub use q8::{quantize_row_q8, stq_matmul_q8, stq_matvec_q8, BlockQ8, STQ1_0_LUT_I8};
pub use quantize::quantize_row_stq1_0_ref;
