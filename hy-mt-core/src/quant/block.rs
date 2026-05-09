use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::{Error, Result};

/// Number of weights packed into a single STQ1_0 block (== `QK_K` in ggml).
pub const QK_K: usize = 256;

/// Number of bytes occupied by one [`BlockStq1_0`].
///
/// Layout: 32 (`qs`) + 8 (`sign`) + 2 (`d`) = **42 bytes** for 256 weights,
/// i.e. exactly **1.3125 bits per weight**.
pub const BLOCK_BYTES: usize = 42;

/// One STQ1_0 block, mirroring the C definition from
/// `ggml/src/ggml-common.h` (PR ggml-org/llama.cpp#22836):
///
/// ```text
/// typedef struct {
///     uint8_t qs[QK_K/8];     // 4-bit code per group of 4 weights
///     uint8_t sign[QK_K/32];  // 1-bit table-select per group of 4 weights
///     ggml_half d;            // shared fp16 scale
/// } block_stq1_0;
/// ```
///
/// Each block decodes to 256 weights; one weight in every group of four is
/// guaranteed to be zero (3:4 sparsity pattern), the remaining three take
/// values in `{-d, +d}` as decoded through the codebook.
///
/// Stored on disk in little-endian byte order. The struct uses
/// `#[repr(C, packed)]` so a `&[BlockStq1_0]` can be obtained from raw GGUF
/// bytes via `bytemuck::cast_slice`.
#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockStq1_0 {
    /// 32 bytes; each byte stores two 4-bit codes for two consecutive
    /// 4-weight groups. The low nibble is the lower-index group.
    pub qs: [u8; QK_K / 8],
    /// 8 bytes; bit `g` (LSB-first) selects the codebook table for
    /// 4-weight group `g`.
    pub sign: [u8; QK_K / 32],
    /// fp16 absmax scale shared across the 256 weights of this block.
    pub d: f16,
}

const _: () = {
    // Compile-time guarantee that the block matches the GGUF on-disk size.
    assert!(std::mem::size_of::<BlockStq1_0>() == BLOCK_BYTES);
    assert!(std::mem::align_of::<BlockStq1_0>() == 1);
};

impl BlockStq1_0 {
    /// Reinterpret a contiguous byte slice as a slice of `BlockStq1_0`.
    ///
    /// The slice length must be a multiple of [`BLOCK_BYTES`] (42).
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<&[Self]> {
        if bytes.len() % BLOCK_BYTES != 0 {
            return Err(Error::Stq1_0(format!(
                "byte length {} is not a multiple of {BLOCK_BYTES}",
                bytes.len()
            )));
        }
        // Safe: BlockStq1_0 is `#[repr(C, packed)]` with align 1 and impls Pod.
        Ok(bytemuck::cast_slice(bytes))
    }

    /// Number of blocks needed to hold `n_weights` quantized values.
    pub const fn count_for(n_weights: usize) -> usize {
        n_weights / QK_K
    }
}
