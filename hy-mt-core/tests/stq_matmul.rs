//! End-to-end check of the custom STQ1_0 matmul against a dequantize→matmul
//! reference. Both paths must agree exactly (the matmul kernel uses the
//! same codebook decoding as the dequantizer).

use half::f16;
use hy_mt_core::quant::{
    dequantize_row_stq1_0_f32, quantize_row_stq1_0_ref, stq_matmul_f32, stq_matvec_f32,
    BlockStq1_0, QK_K,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn quantize(src: &[f32]) -> Vec<BlockStq1_0> {
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

#[test]
fn matvec_matches_dequant_then_dot() {
    let rows = 7;
    let cols = QK_K * 3;

    let mut rng = StdRng::seed_from_u64(42);
    let w_full: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-2.0..2.0)).collect();

    // Quantize per row to keep each row's `d` independent (this matches the
    // layout used in real GGUF weights).
    let mut weights = Vec::with_capacity(rows * cols / QK_K);
    let mut dequant = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &w_full[r * cols..(r + 1) * cols];
        let blocks = quantize(row);
        let mut row_dq = vec![0.0f32; cols];
        dequantize_row_stq1_0_f32(&blocks, &mut row_dq).unwrap();
        dequant[r * cols..(r + 1) * cols].copy_from_slice(&row_dq);
        weights.extend_from_slice(&blocks);
    }

    // Reference: dense f32 matvec.
    let mut y_ref = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &dequant[r * cols..(r + 1) * cols];
        y_ref[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }

    let mut y = vec![0.0f32; rows];
    stq_matvec_f32(&weights, &x, &mut y, rows, cols).unwrap();

    for (a, b) in y.iter().zip(y_ref.iter()) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn matmul_matches_matvec_per_row() {
    let m = 3;
    let n = 5;
    let cols = QK_K * 2;

    let mut rng = StdRng::seed_from_u64(123);
    let w_full: Vec<f32> = (0..n * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x: Vec<f32> = (0..m * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let mut weights = Vec::with_capacity(n * cols / QK_K);
    for r in 0..n {
        let row = &w_full[r * cols..(r + 1) * cols];
        weights.extend_from_slice(&quantize(row));
    }

    // Compute via matmul.
    let mut y = vec![0.0f32; m * n];
    stq_matmul_f32(&weights, &x, &mut y, m, n, cols).unwrap();

    // Compute via per-row matvec.
    let mut y_ref = vec![0.0f32; m * n];
    for r in 0..m {
        let xr = &x[r * cols..(r + 1) * cols];
        let dst = &mut y_ref[r * n..(r + 1) * n];
        stq_matvec_f32(&weights, xr, dst, n, cols).unwrap();
    }

    assert_eq!(y, y_ref);
}
