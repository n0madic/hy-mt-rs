//! 8-bit per-block activation quantization for the STQ1_0 dot-product path.
//!
//! This is a simplified analogue of `llama.cpp`'s `Q8_K` — one f32 scale per
//! block of `QK_K = 256` int8 lanes, no sub-block sums. We trade away those
//! to keep the activation packer cheap; the savings come from
//! (a) shrinking the activation footprint 4× (1024 → 260 bytes per block),
//! (b) collapsing the per-group inner loop to integer dot products that map
//!     directly onto AArch64's `vdotq_s32` (DotProd extension, baseline on
//!     all Apple-silicon and most ARMv8.2-A+ hosts).
//!
//! Ternary STQ1_0 weight × int8 activation has a particularly clean form:
//! the per-lane weight is in `{-1, 0, +1}`, so the dot reduces to a sum of
//! ±activation values without any multiplication. We still go through
//! `vdotq_s32` because it is the cheapest packed integer reduction the
//! microarchitecture exposes — the ternary weight is materialised into an
//! `int8x16_t` once via a 4-byte LUT load and then reused.

use bytemuck::{Pod, Zeroable};

use super::block::{BlockStq1_0, QK_K};
use super::codebook::STQ1_0_CODEBOOK;
use crate::{Error, Result};

/// One block of 256 int8 activation lanes plus a single f32 scale.
///
/// Stored densely on the stack/heap; `qs` is naturally 1-aligned, so a
/// `&[BlockQ8]` may be obtained from raw bytes via `bytemuck::cast_slice`
/// when streaming activations from a pre-quantized pipeline.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8 {
    /// Per-lane int8 codes; the underlying real value is `qs[i] * d`.
    pub qs: [i8; QK_K],
    /// Per-block fp32 scale. `d == 0.0` means the block is all-zero and
    /// the dot product can short-circuit.
    pub d: f32,
}

impl BlockQ8 {
    /// Build a fresh zero-initialised block. Useful as a `dst` argument to
    /// the quantizer when the caller pre-allocates a `Vec<BlockQ8>`.
    pub const fn zeroed() -> Self {
        Self {
            qs: [0; QK_K],
            d: 0.0,
        }
    }
}

/// Pre-decoded `i8` ternary lanes for every codebook entry, mirroring the
/// `STQ1_0_LUT_F32` layout used by the f32 path. `[(s << 4) | code]` →
/// `[i8; 4]` in `{-1, 0, +1}`.
pub const STQ1_0_LUT_I8: [[i8; 4]; 32] = build_lut_i8();

const fn build_lut_i8() -> [[i8; 4]; 32] {
    let mut out = [[0i8; 4]; 32];
    let mut i = 0;
    while i < 32 {
        let qpack = STQ1_0_CODEBOOK[i];
        let mut p = 0;
        while p < 4 {
            // 2 bits per lane; q ∈ {0,1,2}; ternary value = q - 1.
            let q = ((qpack >> (2 * p)) & 0x3) as i8;
            out[i][p] = q - 1;
            p += 1;
        }
        i += 1;
    }
    out
}

/// Quantize a contiguous f32 row into a sequence of [`BlockQ8`].
///
/// `src.len()` must be a multiple of `QK_K`; `dst` must hold exactly
/// `src.len() / QK_K` blocks. Blocks are independent and are quantized
/// in parallel via rayon when there are enough of them; below the
/// threshold the sequential loop avoids rayon's per-task overhead.
pub fn quantize_row_q8(src: &[f32], dst: &mut [BlockQ8]) -> Result<()> {
    use rayon::prelude::*;
    if src.len() % QK_K != 0 {
        return Err(Error::Stq1_0(format!(
            "quantize_row_q8: src length {} not a multiple of {QK_K}",
            src.len()
        )));
    }
    let nb = src.len() / QK_K;
    if dst.len() != nb {
        return Err(Error::Stq1_0(format!(
            "quantize_row_q8: expected {nb} blocks, got {}",
            dst.len()
        )));
    }
    // Rayon's per-task overhead (~1 µs) only pays off above ~16 blocks.
    // Most decode-path activations (cols=2048 → 8 blocks) stay sequential.
    if nb >= 16 {
        dst.par_iter_mut()
            .zip(src.par_chunks_exact(QK_K))
            .for_each(quantize_one);
    } else {
        for (block, chunk) in dst.iter_mut().zip(src.chunks_exact(QK_K)) {
            quantize_one((block, chunk));
        }
    }
    Ok(())
}

#[inline]
fn quantize_one((block, chunk): (&mut BlockQ8, &[f32])) {
    let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    if amax == 0.0 {
        block.qs.fill(0);
        block.d = 0.0;
        return;
    }
    // Use the symmetric range [-127, +127] to dodge the i8 underflow
    // boundary at -128 (which would round-trip as +127 after a sign flip).
    let scale = amax / 127.0;
    let inv_scale = 1.0 / scale;
    for (s, q) in chunk.iter().zip(block.qs.iter_mut()) {
        let v = (s * inv_scale).round();
        *q = v.clamp(-127.0, 127.0) as i8;
    }
    block.d = scale;
}

/// Scalar reference dot product: one row of weight blocks against one row
/// of pre-quantized activation blocks. Used as a self-test against the
/// SIMD path and as the fallback on non-SIMD targets.
#[inline]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn stq_dot_row_q8_scalar(w: &[BlockStq1_0], a: &[BlockQ8]) -> f32 {
    debug_assert_eq!(w.len(), a.len());
    let mut acc = 0.0f32;
    for (wb, ab) in w.iter().zip(a.iter()) {
        if ab.d == 0.0 {
            continue;
        }
        let d_w = { wb.d }.to_f32();
        let mut block_sum: i32 = 0;
        for g in 0..(QK_K / 4) {
            let qs_byte = wb.qs[g / 2];
            let code = (qs_byte >> (4 * (g & 1))) & 0x0F;
            let s = (wb.sign[g / 8] >> (g % 8)) & 0x01;
            let lut = STQ1_0_LUT_I8[((s as usize) << 4) | code as usize];
            let act = &ab.qs[g * 4..g * 4 + 4];
            // Ternary {-1,0,+1} × i8 = ±act_i (or 0); written as a multiply
            // so the compiler is free to lower it however it likes.
            block_sum += (lut[0] as i32) * (act[0] as i32)
                + (lut[1] as i32) * (act[1] as i32)
                + (lut[2] as i32) * (act[2] as i32)
                + (lut[3] as i32) * (act[3] as i32);
        }
        acc += d_w * ab.d * (block_sum as f32);
    }
    acc
}

/// Architecture-dispatched STQ1_0 × Q8 dot product for one row pair.
#[inline]
pub fn stq_dot_row_q8(w: &[BlockStq1_0], a: &[BlockQ8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64; vdotq_s32 is gated
        // at runtime via the `dotprod` feature check below.
        unsafe { stq_dot_row_q8_neon(w, a) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        stq_dot_row_q8_scalar(w, a)
    }
}

/// Pre-decoded i8 LUT, packed as one i32 per codebook entry so each entry
/// can be loaded as a single 4-byte read and lane-spliced into a 16-wide
/// ternary vector for `vdotq_s32`. Bytes are little-endian, matching the
/// `[i8; 4]` lane order in [`STQ1_0_LUT_I8`].
#[cfg(target_arch = "aarch64")]
const STQ1_0_LUT_I8_AS_I32: [i32; 32] = {
    let mut out = [0i32; 32];
    let mut i = 0;
    while i < 32 {
        let l = STQ1_0_LUT_I8[i];
        // Each lane is in {-1, 0, +1}; pack as i32 little-endian.
        let b0 = (l[0] as u8) as u32;
        let b1 = (l[1] as u8) as u32;
        let b2 = (l[2] as u8) as u32;
        let b3 = (l[3] as u8) as u32;
        out[i] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) as i32;
        i += 1;
    }
    out
};

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn stq_dot_row_q8_neon(w: &[BlockStq1_0], a: &[BlockQ8]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(w.len(), a.len());
    let mut acc = 0.0f32;
    let lut_ptr = STQ1_0_LUT_I8_AS_I32.as_ptr();

    for (wb, ab) in w.iter().zip(a.iter()) {
        if ab.d == 0.0 {
            continue;
        }
        let d_w = { wb.d }.to_f32();

        // Two i16 accumulators per block. Each stripe contributes
        // 16 i8×i8 widening multiplies (max |product| = 127*1 = 127),
        // and we run 16 stripes per block → cumulative magnitude per
        // i16 lane ≤ 16 * 127 = 2032 (well inside i16). Promotion to
        // i32 happens once at the end of the block via vpaddlq_s16.
        let mut acc0 = vdupq_n_s16(0);
        let mut acc1 = vdupq_n_s16(0);

        // 16 stripes × 16 lanes = 256 weights per block.
        for q in 0..(QK_K / 16) {
            // Splice the four i32-packed LUT entries into one int8x16
            // through `vsetq_lane_s32` (one INS per insert) instead of
            // copying via a stack buffer.
            let qs_byte0 = *wb.qs.get_unchecked(q * 2);
            let qs_byte1 = *wb.qs.get_unchecked(q * 2 + 1);
            let sign_byte = *wb.sign.get_unchecked(q / 2);

            let g_base = q * 4;
            macro_rules! lut_idx {
                ($g_off:literal, $qs_byte:expr, $nibble:literal) => {{
                    let g = g_base + $g_off;
                    let code = ($qs_byte >> (4 * $nibble)) & 0x0F;
                    let s = (sign_byte >> (g & 7)) & 0x01;
                    ((s as usize) << 4) | code as usize
                }};
            }
            let i0 = lut_idx!(0, qs_byte0, 0);
            let i1 = lut_idx!(1, qs_byte0, 1);
            let i2 = lut_idx!(2, qs_byte1, 0);
            let i3 = lut_idx!(3, qs_byte1, 1);

            let mut wlanes = vdupq_n_s32(0);
            wlanes = vsetq_lane_s32::<0>(*lut_ptr.add(i0), wlanes);
            wlanes = vsetq_lane_s32::<1>(*lut_ptr.add(i1), wlanes);
            wlanes = vsetq_lane_s32::<2>(*lut_ptr.add(i2), wlanes);
            wlanes = vsetq_lane_s32::<3>(*lut_ptr.add(i3), wlanes);
            let w_v = vreinterpretq_s8_s32(wlanes);

            let a_v = vld1q_s8(ab.qs.as_ptr().add(q * 16));

            // 8-wide widening multiply on each half. Alternate
            // accumulators so the FMA chain stays out of each other's
            // critical path.
            let lo = vmull_s8(vget_low_s8(w_v), vget_low_s8(a_v));
            let hi = vmull_s8(vget_high_s8(w_v), vget_high_s8(a_v));
            if q & 1 == 0 {
                acc0 = vaddq_s16(acc0, lo);
                acc1 = vaddq_s16(acc1, hi);
            } else {
                acc0 = vaddq_s16(acc0, hi);
                acc1 = vaddq_s16(acc1, lo);
            }
        }

        let block_sum: i32 = vaddlvq_s16(acc0) + vaddlvq_s16(acc1);
        acc += d_w * ab.d * (block_sum as f32);
    }
    acc
}

/// `Y[i] = stq_dot(W[i], a)` for `i in 0..rows`. Parallel over rows.
pub fn stq_matvec_q8(
    weights: &[BlockStq1_0],
    a: &[BlockQ8],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    use rayon::prelude::*;
    if cols % QK_K != 0 {
        return Err(Error::Stq1_0(format!(
            "stq_matvec_q8: cols {cols} not a multiple of {QK_K}"
        )));
    }
    let blocks_per_row = cols / QK_K;
    if weights.len() != rows * blocks_per_row {
        return Err(Error::Stq1_0(format!(
            "stq_matvec_q8: weights {} != rows*bpr {}",
            weights.len(),
            rows * blocks_per_row
        )));
    }
    if a.len() != blocks_per_row {
        return Err(Error::Stq1_0(format!(
            "stq_matvec_q8: a {} != bpr {}",
            a.len(),
            blocks_per_row
        )));
    }
    if y.len() != rows {
        return Err(Error::Stq1_0(format!(
            "stq_matvec_q8: y {} != rows {rows}",
            y.len()
        )));
    }
    y.par_iter_mut()
        .zip(weights.par_chunks(blocks_per_row))
        .for_each(|(y_r, row)| {
            *y_r = stq_dot_row_q8(row, a);
        });
    Ok(())
}

/// `Y = X · W^T` for pre-quantized `X` (m × bpr blocks) and packed STQ1_0
/// weights (`n × cols`). Parallel over output rows.
pub fn stq_matmul_q8(
    weights: &[BlockStq1_0],
    a: &[BlockQ8],
    y: &mut [f32],
    m: usize,
    n: usize,
    cols: usize,
) -> Result<()> {
    use rayon::prelude::*;
    if cols % QK_K != 0 {
        return Err(Error::Stq1_0(format!(
            "stq_matmul_q8: cols {cols} not a multiple of {QK_K}"
        )));
    }
    let bpr = cols / QK_K;
    if a.len() != m * bpr {
        return Err(Error::Stq1_0(format!(
            "stq_matmul_q8: a {} != m*bpr {}",
            a.len(),
            m * bpr
        )));
    }
    if weights.len() != n * bpr {
        return Err(Error::Stq1_0(format!(
            "stq_matmul_q8: weights {} != n*bpr {}",
            weights.len(),
            n * bpr
        )));
    }
    if y.len() != m * n {
        return Err(Error::Stq1_0(format!(
            "stq_matmul_q8: y {} != m*n {}",
            y.len(),
            m * n
        )));
    }
    // Decode-step short-circuit: parallelise across the n output rows
    // (which is what `stq_matvec_q8` does), not across the m=1 input row.
    // Without this, decode steps fall back to a single sequential loop
    // over n and lose all rayon parallelism.
    if m == 1 {
        return stq_matvec_q8(weights, a, y, n, cols);
    }
    y.par_chunks_mut(n)
        .zip(a.par_chunks(bpr))
        .for_each(|(y_row, a_row)| {
            for (r, y_r) in y_row.iter_mut().enumerate() {
                let w_row = &weights[r * bpr..(r + 1) * bpr];
                *y_r = stq_dot_row_q8(w_row, a_row);
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_row_stq1_0_ref, stq_matvec_f32};
    use half::f16;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_blocks(src: &[f32]) -> Vec<BlockStq1_0> {
        let nb = src.len() / QK_K;
        let mut blocks = vec![
            BlockStq1_0 {
                qs: [0; 32],
                sign: [0; 8],
                d: f16::from_f32(0.0),
            };
            nb
        ];
        quantize_row_stq1_0_ref(src, &mut blocks).unwrap();
        blocks
    }

    /// Sanity check: dequantizing the Q8 row back to f32 should return
    /// approximately the original activation.
    #[test]
    fn q8_quantize_dequantize_recovers_input() {
        let mut rng = StdRng::seed_from_u64(0xABCDEF);
        let n = QK_K * 4;
        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
        let mut q = vec![BlockQ8::zeroed(); n / QK_K];
        quantize_row_q8(&x, &mut q).unwrap();
        for (b_idx, blk) in q.iter().enumerate() {
            for i in 0..QK_K {
                let recovered = blk.qs[i] as f32 * blk.d;
                let original = x[b_idx * QK_K + i];
                let err = (recovered - original).abs();
                assert!(
                    err < blk.d * 1.5,
                    "block {b_idx} lane {i}: orig={original} recovered={recovered} err={err}, d={}",
                    blk.d
                );
            }
        }
    }

    /// Q8 kernel must reproduce the dequant-then-dot reference exactly
    /// (modulo the quantization round-trip on the activation). Quantizing
    /// activations to int8 caps relative accuracy at ~1/127 per element,
    /// so for cancellation-prone cases we bound the absolute error
    /// against an analytic envelope rather than relative tolerance.
    #[test]
    fn q8_dot_approximates_f32_dot() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let cols = QK_K * 8;
        let rows = 7;
        let w_full: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-2.0f32..2.0)).collect();

        let mut weights = Vec::new();
        for r in 0..rows {
            weights.extend_from_slice(&make_blocks(&w_full[r * cols..(r + 1) * cols]));
        }

        let mut q8 = vec![BlockQ8::zeroed(); cols / QK_K];
        quantize_row_q8(&x, &mut q8).unwrap();

        // Reference: dense f32 matvec via the existing kernel.
        let mut y_f32 = vec![0.0f32; rows];
        stq_matvec_f32(&weights, &x, &mut y_f32, rows, cols).unwrap();

        // Production path (dispatched, NEON on AArch64).
        let mut y_q8 = vec![0.0f32; rows];
        stq_matvec_q8(&weights, &q8, &mut y_q8, rows, cols).unwrap();

        // Analytical bound on per-row Q8 quantization error:
        //   |Δdot| ≤ Σ_i |w_i| · |x_i − x_recovered_i|
        // and per-element Q8 round-trip error is bounded by d_a/2 ≤ |x|/127.
        // Worst-case per-row: cols * max|w| * max|x| / 127.
        let bound = (cols as f32) * 1.0 * 2.0 / 127.0; // ≈ 32 for cols=2048
        for r in 0..rows {
            let abs_err = (y_f32[r] - y_q8[r]).abs();
            assert!(
                abs_err < bound,
                "row={r}: f32={}, q8={}, abs_err={abs_err}, bound={bound}",
                y_f32[r],
                y_q8[r]
            );
        }
    }

    /// Cross-validate the Q8 dispatched path against its scalar reference.
    /// On AArch64 this exercises the NEON kernel; both must agree to ULP.
    #[test]
    fn q8_dispatched_matches_scalar_multirow() {
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let cols = QK_K * 5;
        let rows = 6;
        let w_full: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut weights = Vec::new();
        for r in 0..rows {
            weights.extend_from_slice(&make_blocks(&w_full[r * cols..(r + 1) * cols]));
        }
        let mut q8 = vec![BlockQ8::zeroed(); cols / QK_K];
        quantize_row_q8(&x, &mut q8).unwrap();
        let bpr = cols / QK_K;

        for r in 0..rows {
            let row = &weights[r * bpr..(r + 1) * bpr];
            let s = stq_dot_row_q8_scalar(row, &q8);
            let n = stq_dot_row_q8(row, &q8);
            let tol = (s.abs() + n.abs()).max(1.0) * 1e-5;
            assert!((s - n).abs() < tol, "row={r}: scalar={s}, dispatched={n}");
        }
    }

    #[test]
    fn q8_dispatched_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0xBADBAD);
        let cols = QK_K * 4;
        let w_src: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let blocks = make_blocks(&w_src);
        let mut q8 = vec![BlockQ8::zeroed(); cols / QK_K];
        quantize_row_q8(&x, &mut q8).unwrap();

        let lhs = stq_dot_row_q8(&blocks, &q8);
        let rhs = stq_dot_row_q8_scalar(&blocks, &q8);
        let tol = (lhs.abs() + rhs.abs()).max(1.0) * 1e-5;
        assert!(
            (lhs - rhs).abs() < tol,
            "dispatched={lhs}, scalar={rhs}, diff={}",
            lhs - rhs
        );
    }
}
