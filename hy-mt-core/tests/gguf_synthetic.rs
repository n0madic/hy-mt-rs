//! Build a minimal synthetic GGUF v3 file with one F32 tensor and one
//! STQ1_0 tensor, then load it through `HyGgufFile` and verify the parsed
//! metadata, shapes, and tensor views.

use std::io::Write;
use std::path::PathBuf;

use byteorder::{LittleEndian, WriteBytesExt};
use half::f16;

use hy_mt_core::gguf::vendored::GgmlDTypeExt;
use hy_mt_core::gguf::{HyGgufFile, TensorView};
use hy_mt_core::quant::{quantize_row_stq1_0_ref, BlockStq1_0, BLOCK_BYTES, QK_K};

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;

const VAL_U32: u32 = 4;
const VAL_STRING: u32 = 8;

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

fn pad_to_alignment(buf: &mut Vec<u8>) {
    let pos = buf.len() as u64;
    let aligned = pos.div_ceil(ALIGNMENT) * ALIGNMENT;
    let pad = aligned - pos;
    buf.extend(std::iter::repeat_n(0u8, pad as usize));
}

#[test]
fn parses_synthetic_gguf_with_stq1_0_and_f32() {
    let cols: usize = 2 * QK_K; // 512
    let rows: usize = 3;
    let n_weights = rows * cols;

    // Quantize a deterministic ramp into STQ1_0 blocks.
    let mut src = vec![0.0f32; n_weights];
    for (i, v) in src.iter_mut().enumerate() {
        // Pattern: 3:4 sparsity within each group of 4 (matches encoder
        // expectation that exactly one lane be the smallest |x|).
        *v = match i % 4 {
            0 => 0.0,
            1 => 0.5,
            2 => -0.5,
            _ => 0.5,
        };
    }
    let mut blocks = vec![
        BlockStq1_0 {
            qs: [0; 32],
            sign: [0; 8],
            d: f16::from_f32(0.0),
        };
        n_weights / QK_K
    ];
    quantize_row_stq1_0_ref(&src, &mut blocks).unwrap();
    let stq_bytes: &[u8] = bytemuck::cast_slice(&blocks);
    assert_eq!(stq_bytes.len(), blocks.len() * BLOCK_BYTES);

    // Make a small dense F32 tensor.
    let dense_shape = vec![4usize, 8];
    let dense: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let dense_bytes: &[u8] = bytemuck::cast_slice(&dense);

    // ---- header ----------------------------------------------------------
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(2).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(8).unwrap(); // metadata_kv_count

    // metadata: enough fields for HunyuanConfig::from_gguf to succeed.
    // The numeric values are arbitrary — this test only verifies that the
    // header parses end-to-end, not the model's behaviour.
    write_string(&mut buf, "general.architecture");
    buf.write_u32::<LittleEndian>(VAL_STRING).unwrap();
    write_string(&mut buf, "hunyuan-dense");

    write_string(&mut buf, "hunyuan-dense.block_count");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(32).unwrap();

    write_string(&mut buf, "hunyuan-dense.embedding_length");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(2048).unwrap();

    write_string(&mut buf, "hunyuan-dense.feed_forward_length");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(6144).unwrap();

    write_string(&mut buf, "hunyuan-dense.attention.head_count");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(16).unwrap();

    write_string(&mut buf, "hunyuan-dense.attention.head_count_kv");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(4).unwrap();

    write_string(&mut buf, "hunyuan-dense.context_length");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(8192).unwrap();

    write_string(&mut buf, "hunyuan-dense.vocab_size");
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(120818).unwrap();

    // tensor 0: STQ1_0 weight, shape [rows, cols] (GGUF stores reversed → write [cols, rows])
    write_string(&mut buf, "blk.0.attn_q.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(cols as u64).unwrap();
    buf.write_u64::<LittleEndian>(rows as u64).unwrap();
    buf.write_u32::<LittleEndian>(GgmlDTypeExt::Stq1_0 as u32)
        .unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    // tensor 1: F32 dense tensor at offset = stq_bytes.len() + alignment padding
    write_string(&mut buf, "token_embd.weight");
    buf.write_u32::<LittleEndian>(2).unwrap();
    buf.write_u64::<LittleEndian>(dense_shape[1] as u64)
        .unwrap();
    buf.write_u64::<LittleEndian>(dense_shape[0] as u64)
        .unwrap();
    buf.write_u32::<LittleEndian>(GgmlDTypeExt::F32 as u32)
        .unwrap();
    let stq_padded = stq_bytes.len().div_ceil(ALIGNMENT as usize) * ALIGNMENT as usize;
    buf.write_u64::<LittleEndian>(stq_padded as u64).unwrap();

    // pad header up to alignment so tensor data starts on the boundary.
    pad_to_alignment(&mut buf);

    // ---- tensor data ----------------------------------------------------
    buf.extend_from_slice(stq_bytes);
    pad_to_alignment(&mut buf);
    buf.extend_from_slice(dense_bytes);

    // ---- write to a temp file and load -----------------------------------
    let tmp_dir = std::env::temp_dir();
    let path: PathBuf = tmp_dir.join("hy_mt_synth_test.gguf");
    std::fs::write(&path, &buf).unwrap();

    let gguf = HyGgufFile::load(&path).unwrap();

    // Metadata round-trip.
    let meta = gguf.meta();
    assert_eq!(meta.architecture().unwrap(), "hunyuan-dense");
    assert_eq!(meta.u32("hunyuan-dense.block_count").unwrap(), 32);

    // Tensor count + names.
    assert_eq!(gguf.content().tensor_infos.len(), 2);
    let names: std::collections::HashSet<&str> = gguf.tensor_names().collect();
    assert!(names.contains("blk.0.attn_q.weight"));
    assert!(names.contains("token_embd.weight"));

    // STQ1_0 tensor view.
    match gguf.tensor_view("blk.0.attn_q.weight").unwrap() {
        TensorView::Stq1_0 { shape, data } => {
            assert_eq!(shape, vec![rows, cols]);
            assert_eq!(data.len(), n_weights / QK_K);

            // Round-trip dequant must match what we encoded.
            let mut got = vec![0.0f32; n_weights];
            hy_mt_core::quant::dequantize_row_stq1_0_f32(data, &mut got).unwrap();
            for (a, b) in src.iter().zip(got.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }
        other => panic!("expected STQ1_0 view, got {other:?}"),
    }

    // F32 tensor view.
    match gguf.tensor_view("token_embd.weight").unwrap() {
        TensorView::F32 { shape, data } => {
            assert_eq!(shape, dense_shape);
            assert_eq!(data, dense.as_slice());
        }
        other => panic!("expected F32 view, got {other:?}"),
    }

    let _ = std::fs::remove_file(path);
}
