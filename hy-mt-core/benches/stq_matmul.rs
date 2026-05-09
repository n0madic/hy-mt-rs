//! Microbenchmarks for the STQ1_0 dot-product / matvec / matmul kernels.
//!
//! Shapes mirror the real Hy-MT 1.5 1.8B model:
//!   - hidden = 2048, intermediate = 6144
//!   - n_heads = 16, n_kv_heads = 4, head_dim = 128
//!
//! Run with:
//!   cargo bench -p hy-mt-core --bench stq_matmul

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;
use hy_mt_core::quant::{
    quantize_row_q8, quantize_row_stq1_0_ref, stq_matmul_f32, stq_matmul_q8, stq_matvec_f32,
    stq_matvec_q8, BlockQ8, BlockStq1_0, QK_K,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_blocks(n: usize, seed: u64) -> Vec<BlockStq1_0> {
    let mut rng = StdRng::seed_from_u64(seed);
    let src: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let nb = n / QK_K;
    let mut blocks = vec![
        BlockStq1_0 {
            qs: [0; 32],
            sign: [0; 8],
            d: f16::from_f32(0.0),
        };
        nb
    ];
    quantize_row_stq1_0_ref(&src, &mut blocks).unwrap();
    blocks
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

/// Single-row dot product across cols of varying length. Touches only the
/// inner loop; no rayon. Useful for measuring the SIMD-isation effect on the
/// hottest function in the codec.
fn bench_dot_row(c: &mut Criterion) {
    let mut group = c.benchmark_group("stq_dot_row_f32");
    for &cols in &[QK_K, 2048, 6144] {
        let blocks = make_blocks(cols, 0x42);
        let x = make_x(cols, 0x43);
        group.throughput(Throughput::Elements(cols as u64));
        group.bench_with_input(BenchmarkId::from_parameter(cols), &cols, |b, _| {
            b.iter(|| {
                // Use the public matvec with rows=1; rayon short-circuits to
                // a single par_iter element which is essentially the inner loop.
                let mut y = [0.0f32; 1];
                stq_matvec_f32(&blocks, black_box(&x), &mut y, 1, cols).unwrap();
                y[0]
            });
        });
    }
    group.finish();
}

/// Matvec for the projection shapes used during decode (m=1).
fn bench_matvec_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("stq_matvec_f32/decode");
    // (rows, cols, label)
    let shapes = [
        (2048usize, 2048usize, "q_proj/o_proj"),
        (512, 2048, "k_proj/v_proj"),
        (6144, 2048, "ffn_gate/up"),
        (2048, 6144, "ffn_down"),
    ];
    for (rows, cols, label) in shapes {
        let blocks = make_blocks(rows * cols, 0x100 + (rows + cols) as u64);
        let x = make_x(cols, 0x200 + cols as u64);
        let mut y = vec![0.0f32; rows];
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{label} [{rows}x{cols}]")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    stq_matvec_f32(&blocks, black_box(&x), &mut y, rows, cols).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Matmul for prefill shapes (m > 1).
fn bench_matmul_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("stq_matmul_f32/prefill");
    let cases = [
        (16usize, 2048usize, 2048usize, "q_proj T=16"),
        (64, 2048, 2048, "q_proj T=64"),
        (16, 6144, 2048, "ffn_gate T=16"),
        (64, 2048, 6144, "ffn_down T=64"),
    ];
    for (m, n, cols, label) in cases {
        let blocks = make_blocks(n * cols, 0x300 + (n + cols) as u64);
        let x = make_x(m * cols, 0x400 + m as u64);
        let mut y = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((m * n * cols) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, n, cols), |b, _| {
            b.iter(|| {
                stq_matmul_f32(&blocks, black_box(&x), &mut y, m, n, cols).unwrap();
            });
        });
    }
    group.finish();
}

/// Q8 path equivalent of [`bench_matvec_decode`]: same shapes, integer
/// activations packed once per call. The activation packing runs inside
/// the timed region — that's the realistic per-call cost on the hot
/// inference path, not just the kernel.
fn bench_matvec_decode_q8(c: &mut Criterion) {
    let mut group = c.benchmark_group("stq_matvec_q8/decode");
    let shapes = [
        (2048usize, 2048usize, "q_proj/o_proj"),
        (512, 2048, "k_proj/v_proj"),
        (6144, 2048, "ffn_gate/up"),
        (2048, 6144, "ffn_down"),
    ];
    for (rows, cols, label) in shapes {
        let blocks = make_blocks(rows * cols, 0x100 + (rows + cols) as u64);
        let x = make_x(cols, 0x200 + cols as u64);
        let mut y = vec![0.0f32; rows];
        let mut q8 = vec![BlockQ8::zeroed(); cols / QK_K];
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{label} [{rows}x{cols}]")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    quantize_row_q8(black_box(&x), &mut q8).unwrap();
                    stq_matvec_q8(&blocks, &q8, &mut y, rows, cols).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Q8 path equivalent of [`bench_matmul_prefill`].
fn bench_matmul_prefill_q8(c: &mut Criterion) {
    let mut group = c.benchmark_group("stq_matmul_q8/prefill");
    let cases = [
        (16usize, 2048usize, 2048usize, "q_proj T=16"),
        (64, 2048, 2048, "q_proj T=64"),
        (16, 6144, 2048, "ffn_gate T=16"),
        (64, 2048, 6144, "ffn_down T=64"),
    ];
    for (m, n, cols, label) in cases {
        let blocks = make_blocks(n * cols, 0x300 + (n + cols) as u64);
        let x = make_x(m * cols, 0x400 + m as u64);
        let mut y = vec![0.0f32; m * n];
        let mut q8 = vec![BlockQ8::zeroed(); m * cols / QK_K];
        group.throughput(Throughput::Elements((m * n * cols) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, n, cols), |b, _| {
            b.iter(|| {
                quantize_row_q8(black_box(&x), &mut q8).unwrap();
                stq_matmul_q8(&blocks, &q8, &mut y, m, n, cols).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_row,
    bench_matvec_decode,
    bench_matmul_prefill,
    bench_matvec_decode_q8,
    bench_matmul_prefill_q8,
);
criterion_main!(benches);
