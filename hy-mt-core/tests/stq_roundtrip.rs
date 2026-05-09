//! Round-trip property: quantize → dequantize must reproduce the input
//! within the `d` (block absmax) tolerance, since each weight is replaced by
//! the closest value in `{-d, 0, +d}` after picking the lane with the
//! smallest absolute value as the "zero lane".

use half::f16;
use hy_mt_core::quant::{dequantize_row_stq1_0_f32, quantize_row_stq1_0_ref, BlockStq1_0, QK_K};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn roundtrip(src: &[f32]) -> Vec<f32> {
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
    let mut out = vec![0.0f32; src.len()];
    dequantize_row_stq1_0_f32(&blocks, &mut out).unwrap();
    out
}

#[test]
fn roundtrip_three_lanes_recoverable() {
    // 3:4 sparsity assumption: if exactly one of every four weights is zero
    // and the others are ±d, the codec is lossless modulo fp16 scale rounding.
    let mut src = vec![0.0f32; QK_K];
    for chunk in src.chunks_exact_mut(4) {
        chunk[0] = 0.0;
        chunk[1] = 0.5;
        chunk[2] = -0.5;
        chunk[3] = 0.5;
    }

    let out = roundtrip(&src);
    for (s, o) in src.iter().zip(out.iter()) {
        assert!((s - o).abs() < 1e-3, "{s} vs {o}");
    }
}

#[test]
fn roundtrip_max_abs_error_bounded_by_d() {
    // For arbitrary inputs the codec snaps every weight to {-d, 0, +d}.
    // The largest possible error is therefore d (when a true |x| ≈ d/2 is
    // rounded to either 0 or d). We assert exactly that bound.
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let nb = 4;
    let mut src = vec![0.0f32; nb * QK_K];
    for v in src.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    let out = roundtrip(&src);
    let amax = src.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let err = src
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // d is stored as fp16, so allow a small fp16-round-off slack.
    assert!(err <= amax + 1e-2, "max abs err {err} > d {amax}");
}

#[test]
fn roundtrip_all_zero_block() {
    let src = vec![0.0f32; QK_K];
    let out = roundtrip(&src);
    for v in out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn roundtrip_preserves_sign_pattern() {
    // Every group has 1 zero + 3 same-sign non-zeros. The codec must
    // preserve the signs (since the sign bit selector is part of the encoding).
    let mut src = vec![0.0f32; QK_K];
    for (i, chunk) in src.chunks_exact_mut(4).enumerate() {
        chunk[0] = 0.0;
        let s = if i % 2 == 0 { 1.0 } else { -1.0 };
        chunk[1] = s * 0.7;
        chunk[2] = s * 0.7;
        chunk[3] = s * 0.7;
    }
    let out = roundtrip(&src);
    for (s, o) in src.iter().zip(out.iter()) {
        if *s == 0.0 {
            assert_eq!(*o, 0.0);
        } else {
            assert_eq!(s.signum(), o.signum(), "sign mismatch: {s} → {o}");
        }
    }
}
