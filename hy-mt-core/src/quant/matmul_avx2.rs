//! AVX2 + FMA inner loop for the STQ1_0 dot product on x86_64.
//!
//! 256-bit registers fit eight `f32` lanes — twice the natural NEON width.
//! Each iteration concatenates the lo/hi LUT entries into a single
//! `__m256` and the matching row slice into another, then issues one
//! `_mm256_fmadd_ps`. Compared to the scalar fallback this halves the
//! instruction count on the hot loop.
//!
//! Gated on a runtime check (`is_x86_feature_detected!("avx2")` and `"fma"`)
//! by the dispatch in `super::matmul`, so non-AVX2 x86_64 hosts stay on the
//! auto-vectorisation-friendly scalar path.

use std::arch::x86_64::*;

use super::block::{BlockStq1_0, QK_K};
use super::codebook::STQ1_0_LUT_F32;

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn stq_dot_row_f32_avx2(blocks: &[BlockStq1_0], x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), blocks.len() * QK_K);

    let mut acc = 0.0f32;
    for (block_idx, block) in blocks.iter().enumerate() {
        let d = { block.d }.to_f32();
        let qs = block.qs;
        let sign = block.sign;
        let row_base = x.as_ptr().add(block_idx * QK_K);

        let mut acc8 = _mm256_setzero_ps();

        // 32 iterations × 8 weights = 256 weights per block.
        for gp in 0..(QK_K / 8) {
            let qs_byte = *qs.get_unchecked(gp);
            let sign_byte = *sign.get_unchecked(gp / 4);
            let g_lo = gp * 2;
            let g_hi = g_lo + 1;
            let s_lo = (sign_byte >> (g_lo & 7)) & 0x01;
            let s_hi = (sign_byte >> (g_hi & 7)) & 0x01;

            let idx_lo = ((s_lo as usize) << 4) | (qs_byte & 0x0F) as usize;
            let idx_hi = ((s_hi as usize) << 4) | ((qs_byte >> 4) & 0x0F) as usize;

            // Concatenate the two 4-lane LUT entries into one 8-lane vector.
            // The `[[f32; 4]; 32]` layout means `idx_hi == idx_lo + 1` is
            // *not* guaranteed (the indices are independent), so we splice
            // via `_mm_loadu_ps` halves rather than a single 256-bit load.
            let lo128 = _mm_loadu_ps(STQ1_0_LUT_F32.as_ptr().cast::<f32>().add(idx_lo * 4));
            let hi128 = _mm_loadu_ps(STQ1_0_LUT_F32.as_ptr().cast::<f32>().add(idx_hi * 4));
            let lut = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo128), hi128);

            // The 8 weights of this iteration are contiguous in the row.
            let row = _mm256_loadu_ps(row_base.add(g_lo * 4));

            acc8 = _mm256_fmadd_ps(lut, row, acc8);
        }

        // Horizontal sum of 8 lanes → scalar.
        let lo = _mm256_castps256_ps128(acc8);
        let hi = _mm256_extractf128_ps::<1>(acc8);
        let sum128 = _mm_add_ps(lo, hi);
        // sum128 = [a, b, c, d]; collapse to a+b+c+d.
        let sh1 = _mm_movehl_ps(sum128, sum128); // [c, d, c, d]
        let s2 = _mm_add_ps(sum128, sh1); // [a+c, b+d, ., .]
        let sh2 = _mm_shuffle_ps::<0x55>(s2, s2); // [b+d, b+d, ., .]
        let s3 = _mm_add_ss(s2, sh2);
        let block_sum = _mm_cvtss_f32(s3);

        acc += d * block_sum;
    }
    acc
}
