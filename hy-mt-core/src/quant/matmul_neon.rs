//! NEON-vectorised inner loop for the STQ1_0 dot product on AArch64.
//!
//! Mirrors the scalar [`super::matmul::stq_dot_row_f32`] bit-for-bit but
//! replaces the per-group 4-wide scalar FMAs with `vfmaq_f32`. The decoded
//! ternary lanes already live in [`super::codebook::STQ1_0_LUT_F32`] as
//! `[f32; 4]`, so each group is a single 128-bit load and one fused
//! multiply-add.
//!
//! NEON is part of the AArch64 base ABI (every `aarch64-*` target Rust ships
//! has it enabled by default), so we deliberately do **not** add
//! `#[target_feature(enable = "neon")]` — that attribute would prevent the
//! function from being inlined into callers that do not opt into the same
//! feature gate, which costs a measurable function-call per row on the
//! hottest path.

use std::arch::aarch64::*;

use super::block::{BlockStq1_0, QK_K};
use super::codebook::STQ1_0_LUT_F32;

#[inline]
pub unsafe fn stq_dot_row_f32_neon(blocks: &[BlockStq1_0], x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), blocks.len() * QK_K);

    // Two accumulators per block to extract instruction-level parallelism;
    // the loop body's lo/hi groups are independent so the FMAs can issue
    // out-of-order without a serial dependency through the carry.
    let mut acc = 0.0f32;
    for (block_idx, block) in blocks.iter().enumerate() {
        let d = { block.d }.to_f32();
        let qs = block.qs;
        let sign = block.sign;
        let row_base = x.as_ptr().add(block_idx * QK_K);

        let mut acc_lo = vdupq_n_f32(0.0);
        let mut acc_hi = vdupq_n_f32(0.0);

        // 32 iterations; each handles two consecutive 4-weight groups.
        for gp in 0..(QK_K / 8) {
            let qs_byte = *qs.get_unchecked(gp);
            let sign_byte = *sign.get_unchecked(gp / 4);
            let g_lo = gp * 2;
            let g_hi = g_lo + 1;
            let s_lo = (sign_byte >> (g_lo & 7)) & 0x01;
            let s_hi = (sign_byte >> (g_hi & 7)) & 0x01;

            let idx_lo = ((s_lo as usize) << 4) | (qs_byte & 0x0F) as usize;
            let idx_hi = ((s_hi as usize) << 4) | ((qs_byte >> 4) & 0x0F) as usize;

            // 128-bit aligned-by-natural-alignment loads: STQ1_0_LUT_F32 is a
            // global `[[f32; 4]; 32]` so every entry sits on a 16-byte
            // boundary; `row_base` came from `x: &[f32]` which guarantees
            // 4-byte alignment (sufficient for `vld1q_f32`).
            let lut_lo = vld1q_f32(STQ1_0_LUT_F32.as_ptr().cast::<f32>().add(idx_lo * 4));
            let lut_hi = vld1q_f32(STQ1_0_LUT_F32.as_ptr().cast::<f32>().add(idx_hi * 4));

            let row_lo = vld1q_f32(row_base.add(g_lo * 4));
            let row_hi = vld1q_f32(row_base.add(g_hi * 4));

            acc_lo = vfmaq_f32(acc_lo, lut_lo, row_lo);
            acc_hi = vfmaq_f32(acc_hi, lut_hi, row_hi);
        }

        // Horizontal sum across both lanes; multiply by the block scale.
        let block_sum = vaddvq_f32(vaddq_f32(acc_lo, acc_hi));
        acc += d * block_sum;
    }
    acc
}
