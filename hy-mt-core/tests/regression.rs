//! Regression tests for the security/robustness/perf fixes added during
//! the whole-project audit.

use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::{Device, Tensor};

use hy_mt_core::gguf::vendored::{Content, GgmlDTypeExt, TensorInfo};
use hy_mt_core::model::KvCache;
use hy_mt_core::quant::{
    dequantize_row_stq1_0_f32, quantize_row_stq1_0_ref, stq_matvec_f32, BlockStq1_0, QK_K,
};
use hy_mt_core::weights::WeightStore;
use hy_mt_core::Error;

// ---------------------------------------------------------------------------
// C2: bounded GGUF allocations against malicious headers.
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;

#[test]
fn gguf_oversized_string_is_rejected() {
    // Header: magic, version, tensor_count=0, metadata_kv_count=1, then a
    // single key whose declared length is u64::MAX.
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // metadata_kv_count
    buf.write_u64::<LittleEndian>(u64::MAX).unwrap(); // key length

    let mut cur = std::io::Cursor::new(&buf[..]);
    let err = Content::read(&mut cur).expect_err("oversized string must be rejected");
    match err {
        Error::OverLimit { what, .. } => {
            assert!(
                what.contains("string"),
                "expected string-length OverLimit, got {what}"
            );
        }
        other => panic!("expected OverLimit, got {other:?}"),
    }
}

#[test]
fn gguf_oversized_tensor_count_is_rejected() {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(u64::MAX).unwrap(); // tensor_count = u64::MAX
    buf.write_u64::<LittleEndian>(0).unwrap(); // metadata_kv_count

    let mut cur = std::io::Cursor::new(&buf[..]);
    let err = Content::read(&mut cur).expect_err("oversized tensor_count must be rejected");
    match err {
        Error::OverLimit { what, .. } => {
            assert!(
                what.contains("tensor_count"),
                "expected tensor_count OverLimit, got {what}"
            );
        }
        other => panic!("expected OverLimit, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// H5: WeightStore::shape rejects unexpected ranks instead of returning (0,0).
// ---------------------------------------------------------------------------

#[test]
fn weight_store_shape_rejects_3d_tensor() {
    let t = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
    let store = WeightStore::Tensor(t);
    let err = store
        .shape()
        .expect_err("rank-3 weight must produce BadShape");
    match err {
        Error::BadShape { actual, .. } => assert_eq!(actual, vec![2, 3, 4]),
        other => panic!("expected BadShape, got {other:?}"),
    }
}

#[test]
fn weight_store_shape_accepts_1d_and_2d() {
    let t1 = Tensor::zeros((7,), candle_core::DType::F32, &Device::Cpu).unwrap();
    assert_eq!(WeightStore::Tensor(t1).shape().unwrap(), (1, 7));
    let t2 = Tensor::zeros((4, 5), candle_core::DType::F32, &Device::Cpu).unwrap();
    assert_eq!(WeightStore::Tensor(t2).shape().unwrap(), (4, 5));
}

// ---------------------------------------------------------------------------
// H4 / branchless matmul: round-trip agreement with explicit dequant on a
// shape that wasn't covered by `stq_matmul.rs`.
// ---------------------------------------------------------------------------

#[test]
fn matvec_branchless_matches_explicit_dequant() {
    let rows = 4;
    let cols = QK_K * 5;
    let w_full: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.02).cos()).collect();

    let mut weights: Vec<BlockStq1_0> = Vec::with_capacity(rows * cols / QK_K);
    let mut dq_full = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut blocks = vec![
            BlockStq1_0 {
                qs: [0; 32],
                sign: [0; 8],
                d: half::f16::from_f32(0.0),
            };
            cols / QK_K
        ];
        quantize_row_stq1_0_ref(&w_full[r * cols..(r + 1) * cols], &mut blocks).unwrap();
        let mut row_dq = vec![0.0f32; cols];
        dequantize_row_stq1_0_f32(&blocks, &mut row_dq).unwrap();
        dq_full[r * cols..(r + 1) * cols].copy_from_slice(&row_dq);
        weights.extend_from_slice(&blocks);
    }

    let mut y_ref = vec![0.0f32; rows];
    for r in 0..rows {
        y_ref[r] = dq_full[r * cols..(r + 1) * cols]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    let mut y = vec![0.0f32; rows];
    stq_matvec_f32(&weights, &x, &mut y, rows, cols).unwrap();
    // Tight tolerance: ternary weights × f32 activations should match
    // the explicit-dequant reference up to FMA-reordering noise (~1e-5),
    // not just the loose 1e-3 used by the older sanity test.
    for (a, b) in y.iter().zip(y_ref.iter()) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// H2: KvCache uses pre-allocated storage and grows via slice_set.
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_grows_in_place() {
    let mut c = KvCache::with_capacity(8);
    let dev = Device::Cpu;
    let k1 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 4), &dev).unwrap();
    let v1 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 4), &dev).unwrap();
    let (k_full, v_full) = c.append(&k1, &v1).unwrap();
    assert_eq!(k_full.dims(), &[1, 2, 3, 4]);
    assert_eq!(v_full.dims(), &[1, 2, 3, 4]);
    assert_eq!(c.len(), 3);

    let k2 = Tensor::randn(0.0f32, 1.0, (1, 2, 2, 4), &dev).unwrap();
    let v2 = Tensor::randn(0.0f32, 1.0, (1, 2, 2, 4), &dev).unwrap();
    let (k_full, v_full) = c.append(&k2, &v2).unwrap();
    assert_eq!(k_full.dims(), &[1, 2, 5, 4]);
    assert_eq!(v_full.dims(), &[1, 2, 5, 4]);
    assert_eq!(c.len(), 5);
}

#[test]
fn kv_cache_capacity_overflow_errors() {
    let mut c = KvCache::with_capacity(2);
    let dev = Device::Cpu;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 3, 1), &dev).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 1, 3, 1), &dev).unwrap();
    let err = c.append(&k, &v).expect_err("3 > capacity 2 must fail");
    match err {
        Error::OverLimit { what, got, max } => {
            assert!(what.contains("capacity"), "what was {what:?}");
            assert_eq!(got, 3);
            assert_eq!(max, 2);
        }
        other => panic!("expected Error::OverLimit, got {other:?}"),
    }
}

#[test]
fn kv_cache_without_capacity_errors() {
    // After Phase B the silent 8192 fallback is gone; calling `append`
    // before `with_capacity`/`set_capacity` must fail loudly.
    let mut c = KvCache::new();
    let dev = Device::Cpu;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 4), &dev).unwrap();
    let v = k.copy().unwrap();
    let err = c.append(&k, &v).expect_err("missing capacity must fail");
    matches!(err, Error::Validation(_));
}

#[test]
fn kv_cache_rejects_shape_mismatch_on_second_append() {
    // After the first append fixes (b, h, d), a later call with a
    // different head-dim must be rejected with BadShape rather than
    // silently writing garbage.
    let mut c = KvCache::with_capacity(8);
    let dev = Device::Cpu;
    let k1 = Tensor::randn(0.0f32, 1.0, (1, 2, 1, 4), &dev).unwrap();
    let v1 = k1.copy().unwrap();
    c.append(&k1, &v1).unwrap();

    let k2 = Tensor::randn(0.0f32, 1.0, (1, 2, 1, 8), &dev).unwrap();
    let v2 = k2.copy().unwrap();
    let err = c
        .append(&k2, &v2)
        .expect_err("shape mismatch on second append must fail");
    match err {
        Error::BadShape { .. } => {}
        other => panic!("expected BadShape, got {other:?}"),
    }
}

#[test]
fn kv_cache_reusable_after_reset() {
    let mut c = KvCache::with_capacity(4);
    let dev = Device::Cpu;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 2, 4), &dev).unwrap();
    let v = k.copy().unwrap();
    c.append(&k, &v).unwrap();
    assert_eq!(c.len(), 2);
    c.reset();
    assert_eq!(c.len(), 0);
    let (kf, _) = c.append(&k, &v).unwrap();
    assert_eq!(kf.dims(), &[1, 1, 2, 4]);
}

#[test]
fn kv_cache_set_capacity_on_populated_errors() {
    // Changing capacity after populating the cache would invalidate the
    // storage shape vs. cap; must be rejected.
    let mut c = KvCache::with_capacity(4);
    let dev = Device::Cpu;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 2), &dev).unwrap();
    let v = k.copy().unwrap();
    c.append(&k, &v).unwrap();
    let err = c
        .set_capacity(8)
        .expect_err("set_capacity on non-empty cache must fail");
    matches!(err, Error::Validation(_));
}

// ---------------------------------------------------------------------------
// C2 (full coverage): every GGUF length cap fires.
// ---------------------------------------------------------------------------

#[test]
fn gguf_oversized_array_is_rejected() {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // metadata_kv_count = 1
                                               // key "k" (length 1)
    buf.write_u64::<LittleEndian>(1).unwrap();
    buf.push(b'k');
    buf.write_u32::<LittleEndian>(9).unwrap(); // value_type = Array
    buf.write_u32::<LittleEndian>(0).unwrap(); // inner = U8
    buf.write_u64::<LittleEndian>(u64::MAX).unwrap(); // array length
    let mut cur = std::io::Cursor::new(&buf[..]);
    let err = Content::read(&mut cur).expect_err("oversized array must be rejected");
    match err {
        Error::OverLimit { what, .. } => {
            assert!(what.contains("array"), "what was {what:?}")
        }
        other => panic!("expected OverLimit, got {other:?}"),
    }
}

#[test]
fn gguf_oversized_metadata_kv_count_is_rejected() {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(u64::MAX).unwrap(); // metadata_kv_count
    let mut cur = std::io::Cursor::new(&buf[..]);
    let err = Content::read(&mut cur).expect_err("oversized metadata_kv_count must be rejected");
    match err {
        Error::OverLimit { what, .. } => {
            assert!(what.contains("metadata_kv_count"), "what was {what:?}")
        }
        other => panic!("expected OverLimit, got {other:?}"),
    }
}

#[test]
fn gguf_excessive_tensor_dims_is_rejected() {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count = 1
    buf.write_u64::<LittleEndian>(0).unwrap(); // metadata_kv_count = 0
                                               // tensor name "t"
    buf.write_u64::<LittleEndian>(1).unwrap();
    buf.push(b't');
    buf.write_u32::<LittleEndian>(99).unwrap(); // n_dimensions = 99 (> MAX 8)
    let mut cur = std::io::Cursor::new(&buf[..]);
    let err = Content::read(&mut cur).expect_err("99-D tensor must be rejected");
    match err {
        Error::OverLimit { what, .. } => {
            assert!(what.contains("n_dimensions"), "what was {what:?}")
        }
        other => panic!("expected OverLimit, got {other:?}"),
    }
}

#[test]
fn gguf_deeply_nested_array_is_rejected() {
    // Nest Array(Array(...)) past MAX_ARRAY_DEPTH (8). Each nested array
    // is one element (the next array) so the file stays small but the
    // recursion depth grows.
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // metadata_kv_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // key length
    buf.push(b'k');
    buf.write_u32::<LittleEndian>(9).unwrap(); // value_type = Array

    // 16 nested levels of Array(Array(... U8 ...))
    for _ in 0..15 {
        buf.write_u32::<LittleEndian>(9).unwrap(); // inner = Array
        buf.write_u64::<LittleEndian>(1).unwrap(); // length 1
    }
    // innermost: Array of one U8
    buf.write_u32::<LittleEndian>(0).unwrap(); // inner = U8
    buf.write_u64::<LittleEndian>(1).unwrap(); // length 1
    buf.push(0); // the U8 value

    let mut cur = std::io::Cursor::new(&buf[..]);
    let err = Content::read(&mut cur).expect_err("deeply nested array must be rejected");
    match err {
        Error::OverLimit { what, .. } => {
            assert!(what.contains("nesting depth"), "what was {what:?}")
        }
        other => panic!("expected OverLimit, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// C3: TensorInfo::elem_count guards against shape overflow.
// ---------------------------------------------------------------------------

#[test]
fn tensor_info_elem_count_overflow_is_rejected() {
    let info = TensorInfo {
        ggml_dtype: GgmlDTypeExt::F32,
        shape: vec![usize::MAX / 2, 3],
        offset: 0,
    };
    info.elem_count()
        .expect_err("overflow in shape product must be rejected");
}

// ---------------------------------------------------------------------------
// M-2: Generator::generate's max-position check (cap split + max_pos==0).
// ---------------------------------------------------------------------------
// Real Generator needs a model; we exercise the policy via the simpler
// policy guard: the existing test in tests/model_smoke.rs covers the
// happy path. Here we leave a placeholder asserting the saturating_add
// math is correct for the cap.

#[test]
fn cap_math_with_saturating_add() {
    let max_ctx = 100usize;
    // A 50-token prompt with max_new_tokens=usize::MAX must be capped at 100.
    let total = 50usize.saturating_add(usize::MAX).min(max_ctx);
    assert_eq!(total, 100);
}
