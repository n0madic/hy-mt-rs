//! Reference dequantization for STQ1_0 blocks.
//!
//! Mirrors `dequantize_row_stq1_0` from `ggml/src/ggml-quants.c` in
//! `ggml-org/llama.cpp` PR #22836.

use half::f16;

use super::block::{BlockStq1_0, QK_K};
use super::codebook::STQ1_0_CODEBOOK;
use crate::{Error, Result};

/// Dequantize one block into 256 ternary `f32` values multiplied by `block.d`.
#[inline]
pub fn dequantize_block_stq1_0_f32(block: &BlockStq1_0, out: &mut [f32; QK_K]) {
    // `{ block.d }` is a block expression that copies out of the
    // `#[repr(C, packed)]` field by value (allowed for `Copy` types) — no
    // unaligned reference is ever created.
    let d = { block.d }.to_f32();
    let qs = block.qs;
    let sign = block.sign;

    // Each block holds QK_K / 4 = 64 four-weight groups.
    for g in 0..(QK_K / 4) {
        let code = (qs[g / 2] >> (4 * (g & 1))) & 0x0F;
        let s = (sign[g / 8] >> (g % 8)) & 0x01;
        let qpack = STQ1_0_CODEBOOK[((s as usize) << 4) | code as usize];

        // Lane p of `qpack` is a 2-bit code in {0, 1, 2}, mapping to
        // {-1, 0, +1} after subtracting 1.
        let base = g * 4;
        for p in 0..4 {
            let q = ((qpack >> (2 * p)) & 0x3) as i32;
            out[base + p] = (q - 1) as f32 * d;
        }
    }
}

/// Generic worker that dequantizes a row of blocks and writes each value
/// into `dst` after converting it via `convert`. Both [`dequantize_row_stq1_0_f32`]
/// and [`dequantize_row_stq1_0_f16`] are thin wrappers around this helper —
/// this avoids the per-dtype duplication that previously caused two
/// identical decode loops to drift.
fn dequantize_row_into<T, F>(blocks: &[BlockStq1_0], dst: &mut [T], convert: F) -> Result<()>
where
    F: Fn(f32) -> T,
{
    let expected = blocks.len() * QK_K;
    if dst.len() != expected {
        return Err(Error::Stq1_0(format!(
            "dst length {} != blocks.len() * QK_K = {expected}",
            dst.len()
        )));
    }
    let mut tmp = [0.0f32; QK_K];
    for (block, chunk) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
        dequantize_block_stq1_0_f32(block, &mut tmp);
        for (s, d) in tmp.iter().zip(chunk.iter_mut()) {
            *d = convert(*s);
        }
    }
    Ok(())
}

/// Dequantize a contiguous tensor row into `f32`.
///
/// `dst.len()` must be `blocks.len() * QK_K`.
pub fn dequantize_row_stq1_0_f32(blocks: &[BlockStq1_0], dst: &mut [f32]) -> Result<()> {
    dequantize_row_into(blocks, dst, |x| x)
}

/// Same as [`dequantize_row_stq1_0_f32`] but writes directly into a `f16`
/// buffer; convenient when dequantizing weights for upload to a Metal/CUDA
/// device that prefers half-precision storage.
pub fn dequantize_row_stq1_0_f16(blocks: &[BlockStq1_0], dst: &mut [f16]) -> Result<()> {
    dequantize_row_into(blocks, dst, f16::from_f32)
}
