//! On-the-fly STQ1_0 × f32 matmul.
//!
//! The kernel mirrors `ggml_vec_dot_stq1_0_q8_K_generic` but accumulates
//! against `f32` activations directly (we skip the Q8_K activation packing
//! since Candle keeps activations in `f32`/`f16`). It is the inference-time
//! equivalent of "decode each ternary lane and add ±x[i]".

use rayon::prelude::*;

use super::block::{BlockStq1_0, QK_K};
use super::codebook::STQ1_0_LUT_F32;
use crate::{Error, Result};

/// Inner loop: dot one quantized weight row (`blocks`) with a dense `f32`
/// activation row of length `blocks.len() * QK_K`.
///
/// Architecture dispatch:
/// - **AArch64** → hand-written NEON (`vfmaq_f32`, dual accumulators).
///   NEON is part of the AArch64 base ABI so dispatch is unconditional.
/// - **x86_64 with AVX2 + FMA** → 256-bit AVX2 kernel selected once at
///   process start via `is_x86_feature_detected!`, cached in `AVX2_OK`.
/// - **everything else** → the scalar fallback (still SIMD-friendly to LLVM's
///   auto-vectoriser).
#[inline(always)]
pub fn stq_dot_row_f32(blocks: &[BlockStq1_0], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on AArch64; the function does in-bounds
        // pointer arithmetic via the same indices the scalar version uses.
        unsafe { super::matmul_neon::stq_dot_row_f32_neon(blocks, x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: gated on a runtime check that the host has AVX2 + FMA.
            unsafe { super::matmul_avx2::stq_dot_row_f32_avx2(blocks, x) }
        } else {
            stq_dot_row_f32_scalar(blocks, x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        stq_dot_row_f32_scalar(blocks, x)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0 = unknown, 1 = yes, 2 = no. Single AcqRel load on the hot path.
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let ok = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
            CACHE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Scalar fallback used on non-AArch64 targets and as the reference
/// implementation in tests. Behaviour-equivalent to the NEON path.
#[inline]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn stq_dot_row_f32_scalar(blocks: &[BlockStq1_0], x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), blocks.len() * QK_K);
    let mut acc = 0.0f32;
    for (block_idx, block) in blocks.iter().enumerate() {
        // `{ block.d }` is a by-value copy out of the `#[repr(C, packed)]`
        // field, allowed because `f16: Copy` and never producing an
        // unaligned reference.
        let d = { block.d }.to_f32();
        let qs = block.qs;
        let sign = block.sign;
        let row = &x[block_idx * QK_K..][..QK_K];

        let mut block_sum = 0.0f32;
        // Process two adjacent groups (one `qs` byte) per iteration: the
        // shared byte fetch + sign decode is amortised over 8 weights and
        // the 4-wide ternary FMA is fully autovectorisable.
        for gp in 0..(QK_K / 8) {
            let g_lo = gp * 2;
            let g_hi = g_lo + 1;
            let qs_byte = qs[gp];
            let sign_byte = sign[gp / 4];
            let s_lo = (sign_byte >> (g_lo & 7)) & 0x01;
            let s_hi = (sign_byte >> (g_hi & 7)) & 0x01;

            let lut_lo = &STQ1_0_LUT_F32[((s_lo as usize) << 4) | (qs_byte & 0x0F) as usize];
            let lut_hi = &STQ1_0_LUT_F32[((s_hi as usize) << 4) | ((qs_byte >> 4) & 0x0F) as usize];

            let base_lo = g_lo * 4;
            let base_hi = g_hi * 4;
            // Two FMA-friendly 4-wide ternary dot products.
            block_sum += lut_lo[0] * row[base_lo]
                + lut_lo[1] * row[base_lo + 1]
                + lut_lo[2] * row[base_lo + 2]
                + lut_lo[3] * row[base_lo + 3];
            block_sum += lut_hi[0] * row[base_hi]
                + lut_hi[1] * row[base_hi + 1]
                + lut_hi[2] * row[base_hi + 2]
                + lut_hi[3] * row[base_hi + 3];
        }
        acc += d * block_sum;
    }
    acc
}

/// Matrix-vector product `y = W * x`, where `W` is stored row-major as
/// `rows` × `cols` with `cols % QK_K == 0` (each row uses `cols / QK_K`
/// blocks). Rows are processed in parallel.
///
/// `weights.len() == rows * cols / QK_K`, `x.len() == cols`,
/// `y.len() == rows`.
pub fn stq_matvec_f32(
    weights: &[BlockStq1_0],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_shape(weights, x, y, rows, cols)?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let blocks_per_row = cols / QK_K;
    y.par_iter_mut()
        .zip(weights.par_chunks(blocks_per_row))
        .for_each(|(y_r, row_blocks)| {
            *y_r = stq_dot_row_f32(row_blocks, x);
        });
    Ok(())
}

/// Sequential variant of [`stq_matvec_f32`] used by the matmul outer loop
/// to avoid nesting two layers of rayon parallelism.
fn stq_matvec_f32_seq(weights: &[BlockStq1_0], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    let blocks_per_row = cols / QK_K;
    for (r, y_r) in y.iter_mut().enumerate() {
        let row_blocks = &weights[r * blocks_per_row..(r + 1) * blocks_per_row];
        *y_r = stq_dot_row_f32(row_blocks, x);
    }
}

/// Matrix-matrix product `Y = X * W^T`, where:
/// - `X` is `m × cols` (row-major, `f32`),
/// - `W` is `n × cols` (row-major STQ1_0, `cols % QK_K == 0`),
/// - `Y` is `m × n` (row-major, `f32`).
///
/// This is the decoder linear-layer shape (`x @ w.T`): each output row is
/// independent, so we parallelise over output rows of `X` and run each
/// per-row matvec sequentially to avoid nested rayon fan-out.
pub fn stq_matmul_f32(
    weights: &[BlockStq1_0],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    cols: usize,
) -> Result<()> {
    if x.len() != m * cols {
        return Err(Error::Stq1_0(format!(
            "x has {} elems, expected {} (m*cols)",
            x.len(),
            m * cols
        )));
    }
    if y.len() != m * n {
        return Err(Error::Stq1_0(format!(
            "y has {} elems, expected {} (m*n)",
            y.len(),
            m * n
        )));
    }
    if m == 0 || cols == 0 {
        return Ok(());
    }
    if cols % QK_K != 0 {
        return Err(Error::Stq1_0(format!(
            "cols {cols} not a multiple of QK_K={QK_K}"
        )));
    }
    let blocks_per_row = cols / QK_K;
    let expected_blocks = n * blocks_per_row;
    if weights.len() != expected_blocks {
        return Err(Error::Stq1_0(format!(
            "weights has {} blocks, expected {expected_blocks}",
            weights.len()
        )));
    }

    if m == 1 {
        return stq_matvec_f32(weights, x, y, n, cols);
    }

    // Single layer of rayon: parallelise the m output rows; each row runs
    // the matvec sequentially over n.
    y.par_chunks_mut(n)
        .zip(x.par_chunks(cols))
        .for_each(|(y_row, x_row)| stq_matvec_f32_seq(weights, x_row, y_row, n, cols));

    Ok(())
}

fn validate_shape(
    weights: &[BlockStq1_0],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if cols % QK_K != 0 {
        return Err(Error::Stq1_0(format!(
            "cols {cols} not a multiple of QK_K={QK_K}"
        )));
    }
    let blocks_per_row = cols / QK_K;
    let expected_blocks = rows * blocks_per_row;
    if weights.len() != expected_blocks {
        return Err(Error::Stq1_0(format!(
            "weights has {} blocks, expected {expected_blocks}",
            weights.len()
        )));
    }
    if x.len() != cols {
        return Err(Error::Stq1_0(format!(
            "x has length {} != cols {cols}",
            x.len()
        )));
    }
    if y.len() != rows {
        return Err(Error::Stq1_0(format!(
            "y has length {} != rows {rows}",
            y.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{stq_dot_row_f32, stq_dot_row_f32_scalar, QK_K};
    use crate::quant::{quantize_row_stq1_0_ref, BlockStq1_0};
    use half::f16;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Cross-validate the architecture-dispatched kernel against the scalar
    /// reference. On AArch64 this exercises the NEON path; elsewhere it
    /// degenerates to a self-consistency check.
    #[test]
    fn dispatched_dot_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        for &nb in &[1usize, 2, 5, 12, 24] {
            let n = nb * QK_K;
            let src: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();

            let mut blocks = vec![
                BlockStq1_0 {
                    qs: [0; 32],
                    sign: [0; 8],
                    d: f16::from_f32(0.0),
                };
                nb
            ];
            quantize_row_stq1_0_ref(&src, &mut blocks).unwrap();

            let lhs = stq_dot_row_f32(&blocks, &x);
            let rhs = stq_dot_row_f32_scalar(&blocks, &x);
            // Strict bit-equality is not guaranteed because NEON uses a
            // tree-reduction across two accumulators while the scalar
            // path is a left-fold; both are exact-rounded FMAs of the
            // same set of values, so any divergence is sub-ULP.
            let tol = (lhs.abs() + rhs.abs()).max(1.0) * 1e-5;
            assert!(
                (lhs - rhs).abs() < tol,
                "nb={nb}: dispatched={lhs}, scalar={rhs}, diff={}",
                lhs - rhs
            );
        }
    }
}
