//! Reference quantizer for STQ1_0 — used in tests and tooling, not on the
//! hot inference path.
//!
//! Direct port of `quantize_row_stq1_0_ref` from `ggml/src/ggml-quants.c` in
//! `ggml-org/llama.cpp` PR #22836.

use half::f16;

use super::block::{BlockStq1_0, QK_K};
use super::codebook::{STQ1_0_QPACK_TO_SIGN, STQ1_0_QPACK_TO_SLOT};
use crate::{Error, Result};

/// Quantize a contiguous row of `f32` weights into STQ1_0 blocks.
///
/// `src.len()` must be a multiple of `QK_K` (256). The destination slice must
/// hold exactly `src.len() / QK_K` blocks.
pub fn quantize_row_stq1_0_ref(src: &[f32], dst: &mut [BlockStq1_0]) -> Result<()> {
    if src.len() % QK_K != 0 {
        return Err(Error::Stq1_0(format!(
            "src length {} is not a multiple of {QK_K}",
            src.len()
        )));
    }
    let nb = src.len() / QK_K;
    if dst.len() != nb {
        return Err(Error::Stq1_0(format!(
            "expected {nb} blocks, got {}",
            dst.len()
        )));
    }

    for (block_idx, block) in dst.iter_mut().enumerate() {
        let x = &src[block_idx * QK_K..][..QK_K];

        // Reset the block to all zeros — the loop below ORs in the encoded bits.
        block.qs.fill(0);
        block.sign.fill(0);

        // d = absmax of the block.
        let mut amax = 0.0f32;
        for v in x.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        block.d = f16::from_f32(amax);

        for g in 0..(QK_K / 4) {
            let xv = &x[g * 4..][..4];

            // Pick the lane with the smallest |x| as the zero lane (3:4 sparsity).
            let mut zero_pos = 0;
            let mut min_abs = xv[0].abs();
            for (p, v) in xv.iter().enumerate().skip(1) {
                let a = v.abs();
                if a < min_abs {
                    min_abs = a;
                    zero_pos = p;
                }
            }

            // Build the 8-bit packed lane pattern (2 bits per lane):
            // zero lane → 0b01, negative non-zero → 0b00, positive non-zero → 0b10.
            let mut qpack: u8 = 0;
            for (p, v) in xv.iter().enumerate() {
                let lane: u8 = if p == zero_pos {
                    0x1
                } else if *v < 0.0 {
                    0x0
                } else {
                    0x2
                };
                qpack |= lane << (2 * p);
            }

            let code = STQ1_0_QPACK_TO_SLOT[qpack as usize];
            let sign = STQ1_0_QPACK_TO_SIGN[qpack as usize];
            if code == 0xFF {
                return Err(Error::Stq1_0(format!(
                    "unencodable lane pattern qpack=0x{qpack:02X} (no zero lane)"
                )));
            }

            block.qs[g / 2] |= (code & 0x0F) << (4 * (g & 1));
            block.sign[g / 8] |= sign << (g % 8);
        }
    }

    Ok(())
}
