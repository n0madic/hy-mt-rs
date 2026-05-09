//! Build a tiny synthetic Hunyuan-MT-shaped GGUF in memory, load it as a
//! model on CPU and run a forward pass. Validates tensor-name routing,
//! shape arithmetic in attention/FFN, RoPE shape handling, and the tied
//! lm_head path.

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use hy_mt_core::device::DeviceCtx;
use hy_mt_core::gguf::vendored::GgmlDTypeExt;
use hy_mt_core::gguf::HyGgufFile;
use hy_mt_core::model::{HunyuanConfig, HunyuanDense};

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;

const VAL_U32: u32 = 4;
const VAL_F32: u32 = 6;
const VAL_BOOL: u32 = 7;
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

struct PendingTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn write_meta_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    write_string(buf, key);
    buf.write_u32::<LittleEndian>(VAL_STRING).unwrap();
    write_string(buf, value);
}

fn write_meta_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    write_string(buf, key);
    buf.write_u32::<LittleEndian>(VAL_U32).unwrap();
    buf.write_u32::<LittleEndian>(value).unwrap();
}

fn write_meta_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    write_string(buf, key);
    buf.write_u32::<LittleEndian>(VAL_F32).unwrap();
    buf.write_f32::<LittleEndian>(value).unwrap();
}

fn write_meta_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
    write_string(buf, key);
    buf.write_u32::<LittleEndian>(VAL_BOOL).unwrap();
    buf.write_u8(if value { 1 } else { 0 }).unwrap();
}

fn random_tensor(rng: &mut StdRng, shape: &[usize]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let x: f32 = rng.gen_range(-0.1..0.1);
        v.write_f32::<LittleEndian>(x).unwrap();
    }
    v
}

#[test]
fn forward_pass_on_synthetic_tiny_model() {
    // Tiny Hunyuan-MT-like model that fits in seconds and exercises every
    // architectural feature: GQA, QK-norm, SwiGLU, tied LM head.
    let cfg = HunyuanConfig {
        n_layers: 2,
        hidden_size: 32,
        intermediate_size: 64,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        vocab_size: 64,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling_alpha: 1.0,
        use_qk_norm: true,
        tie_word_embeddings: true,
        bos_id: 0,
        eos_id: 1,
        pad_id: 2,
    };

    let mut rng = StdRng::seed_from_u64(12345);

    // ---- collect tensors -------------------------------------------------
    let mut tensors: Vec<PendingTensor> = Vec::new();

    tensors.push(PendingTensor {
        name: "token_embd.weight".into(),
        shape: vec![cfg.vocab_size, cfg.hidden_size],
        data: random_tensor(&mut rng, &[cfg.vocab_size, cfg.hidden_size]),
    });
    tensors.push(PendingTensor {
        name: "output_norm.weight".into(),
        shape: vec![cfg.hidden_size],
        data: random_tensor(&mut rng, &[cfg.hidden_size]),
    });

    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    for layer in 0..cfg.n_layers {
        let prefix = format!("blk.{layer}.");
        for (suffix, shape) in [
            ("attn_norm.weight", vec![cfg.hidden_size]),
            ("attn_q.weight", vec![q_dim, cfg.hidden_size]),
            ("attn_k.weight", vec![kv_dim, cfg.hidden_size]),
            ("attn_v.weight", vec![kv_dim, cfg.hidden_size]),
            ("attn_q_norm.weight", vec![cfg.head_dim]),
            ("attn_k_norm.weight", vec![cfg.head_dim]),
            ("attn_output.weight", vec![cfg.hidden_size, q_dim]),
            ("ffn_norm.weight", vec![cfg.hidden_size]),
            (
                "ffn_gate.weight",
                vec![cfg.intermediate_size, cfg.hidden_size],
            ),
            (
                "ffn_up.weight",
                vec![cfg.intermediate_size, cfg.hidden_size],
            ),
            (
                "ffn_down.weight",
                vec![cfg.hidden_size, cfg.intermediate_size],
            ),
        ] {
            tensors.push(PendingTensor {
                name: format!("{prefix}{suffix}"),
                shape: shape.clone(),
                data: random_tensor(&mut rng, &shape),
            });
        }
    }

    // ---- assemble GGUF ---------------------------------------------------
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();

    let metadata: Vec<(&str, MetaValue)> = vec![
        ("general.architecture", MetaValue::String("hunyuan-dense")),
        (
            "hunyuan-dense.block_count",
            MetaValue::U32(cfg.n_layers as u32),
        ),
        (
            "hunyuan-dense.embedding_length",
            MetaValue::U32(cfg.hidden_size as u32),
        ),
        (
            "hunyuan-dense.feed_forward_length",
            MetaValue::U32(cfg.intermediate_size as u32),
        ),
        (
            "hunyuan-dense.attention.head_count",
            MetaValue::U32(cfg.n_heads as u32),
        ),
        (
            "hunyuan-dense.attention.head_count_kv",
            MetaValue::U32(cfg.n_kv_heads as u32),
        ),
        (
            "hunyuan-dense.attention.key_length",
            MetaValue::U32(cfg.head_dim as u32),
        ),
        ("hunyuan-dense.context_length", MetaValue::U32(1024)),
        (
            "hunyuan-dense.vocab_size",
            MetaValue::U32(cfg.vocab_size as u32),
        ),
        (
            "hunyuan-dense.attention.layer_norm_rms_epsilon",
            MetaValue::F32(cfg.rms_norm_eps),
        ),
        (
            "hunyuan-dense.rope.freq_base",
            MetaValue::F32(cfg.rope_theta),
        ),
        ("hunyuan-dense.attention.use_qk_norm", MetaValue::Bool(true)),
        ("hunyuan-dense.tie_word_embeddings", MetaValue::Bool(true)),
        ("tokenizer.ggml.bos_token_id", MetaValue::U32(cfg.bos_id)),
        ("tokenizer.ggml.eos_token_id", MetaValue::U32(cfg.eos_id)),
        (
            "tokenizer.ggml.padding_token_id",
            MetaValue::U32(cfg.pad_id),
        ),
    ];
    buf.write_u64::<LittleEndian>(metadata.len() as u64)
        .unwrap();

    for (k, v) in &metadata {
        match v {
            MetaValue::String(s) => write_meta_kv_string(&mut buf, k, s),
            MetaValue::U32(u) => write_meta_kv_u32(&mut buf, k, *u),
            MetaValue::F32(f) => write_meta_kv_f32(&mut buf, k, *f),
            MetaValue::Bool(b) => write_meta_kv_bool(&mut buf, k, *b),
        }
    }

    // Tensor descriptors with placeholder offsets (we fix them up next).
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut running = 0usize;
    for t in &tensors {
        offsets.push(running);
        running = (running + t.data.len()).div_ceil(ALIGNMENT as usize) * ALIGNMENT as usize;
    }

    for (t, off) in tensors.iter().zip(offsets.iter()) {
        write_string(&mut buf, &t.name);
        buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
        for &d in t.shape.iter().rev() {
            buf.write_u64::<LittleEndian>(d as u64).unwrap();
        }
        buf.write_u32::<LittleEndian>(GgmlDTypeExt::F32 as u32)
            .unwrap();
        buf.write_u64::<LittleEndian>(*off as u64).unwrap();
    }

    pad_to_alignment(&mut buf);

    // Tensor data, padded to alignment between tensors.
    let data_start = buf.len();
    for (t, off) in tensors.iter().zip(offsets.iter()) {
        let target = data_start + *off;
        if buf.len() < target {
            buf.resize(target, 0);
        }
        buf.extend_from_slice(&t.data);
    }

    let tmp = std::env::temp_dir().join("hy_mt_smoke.gguf");
    std::fs::write(&tmp, &buf).unwrap();

    let gguf = HyGgufFile::load(&tmp).unwrap();
    let dev = DeviceCtx::cpu();
    let mut model = HunyuanDense::load_with_config(&gguf, &dev, cfg).unwrap();

    // Wire KV cache capacity for prefill + one decode step.
    model.reset_kv_cache(4).unwrap();

    // Run a forward pass with 3 tokens.
    let token_ids: Vec<u32> = vec![5, 7, 11];
    let tokens = candle_core::Tensor::from_vec(token_ids.clone(), (1, 3), &dev.device).unwrap();
    let logits = model.forward(&tokens).unwrap();
    let dims = logits.dims().to_vec();
    assert_eq!(dims, vec![1, cfg.vocab_size]);

    // Run an additional decode step using the KV cache and check shape again.
    let next = candle_core::Tensor::from_vec(vec![13u32], (1, 1), &dev.device).unwrap();
    let logits2 = model.forward(&next).unwrap();
    assert_eq!(logits2.dims(), &[1, cfg.vocab_size]);

    // Sanity: logits should not be all NaN/Inf.
    let stats = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(stats.iter().all(|v| v.is_finite()), "non-finite logits");

    let _ = std::fs::remove_file(tmp);
}

enum MetaValue {
    String(&'static str),
    U32(u32),
    F32(f32),
    Bool(bool),
}
