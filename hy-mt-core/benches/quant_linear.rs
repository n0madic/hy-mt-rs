//! End-to-end benchmark for `QuantLinear::forward` on STQ1_0 weights.
//!
//! Measures the full path including activation copies, scratch buffers and
//! tensor wrapping — the stq_matmul microbench measures only the kernel.
//!
//! Run with:
//!   cargo bench -p hy-mt-core --bench quant_linear

use candle_core::{Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;
use hy_mt_core::model::QuantLinear;
use hy_mt_core::quant::{quantize_row_stq1_0_ref, BlockStq1_0, QK_K};
use hy_mt_core::weights::{BlockSource, WeightStore};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_weight(rows: usize, cols: usize, seed: u64) -> WeightStore {
    let mut rng = StdRng::seed_from_u64(seed);
    let src: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    let nb = (rows * cols) / QK_K;
    let mut blocks = vec![
        BlockStq1_0 {
            qs: [0; 32],
            sign: [0; 8],
            d: f16::from_f32(0.0),
        };
        nb
    ];
    quantize_row_stq1_0_ref(&src, &mut blocks).unwrap();
    WeightStore::Stq1_0 {
        blocks: BlockSource::from(blocks),
        rows,
        cols,
    }
}

fn bench_forward_decode(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("QuantLinear::forward/decode");
    let shapes = [
        (2048usize, 2048usize, "q_proj"),
        (512, 2048, "k_proj"),
        (6144, 2048, "ffn_gate"),
        (2048, 6144, "ffn_down"),
    ];
    for (rows, cols, label) in shapes {
        let w = make_weight(rows, cols, 0x500 + (rows + cols) as u64);
        let lin = QuantLinear::new(w).unwrap();
        // Single-token decode shape: [1, 1, cols]
        let mut rng = StdRng::seed_from_u64(0x600);
        let x_data: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let x = Tensor::from_vec(x_data, (1, 1, cols), &device).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{label} [{rows}x{cols}]")),
            &(),
            |b, _| {
                b.iter(|| {
                    let y = lin.forward(black_box(&x)).unwrap();
                    y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_forward_prefill(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("QuantLinear::forward/prefill");
    let cases = [
        (16usize, 2048usize, 2048usize, "q_proj T=16"),
        (64, 2048, 2048, "q_proj T=64"),
        (16, 6144, 2048, "ffn_gate T=16"),
    ];
    for (t, rows, cols, label) in cases {
        let w = make_weight(rows, cols, 0x700 + (rows + cols) as u64);
        let lin = QuantLinear::new(w).unwrap();
        let mut rng = StdRng::seed_from_u64(0x800);
        let x_data: Vec<f32> = (0..t * cols).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let x = Tensor::from_vec(x_data, (1, t, cols), &device).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, _| {
            b.iter(|| {
                let y = lin.forward(black_box(&x)).unwrap();
                y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_forward_decode, bench_forward_prefill);
criterion_main!(benches);
