# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & test commands

```sh
cargo build --release                            # CPU only
cargo build --release --features metal           # Apple Silicon GPU
cargo build --release --features cuda            # NVIDIA GPU

cargo test --workspace                           # all 49 tests (unit + integration)
cargo bench -p hy-mt-core --bench stq_matmul     # raw kernel microbench
cargo bench -p hy-mt-core --bench quant_linear   # full QuantLinear forward (incl. activation copies)
cargo test -p hy-mt-core --test stq_matmul       # one integration test file
cargo test -p hy-mt-core --test regression       # post-fix regression suite
cargo test -p hy-mt-core --lib layout::tests     # one module's unit tests

cargo clippy --workspace --all-targets -- -D warnings                   # CPU
cargo clippy --workspace --all-targets --features metal -- -D warnings  # Metal — must also be clean
```

`--features metal` propagates from `hy-mt-cli` to `hy-mt-core` automatically. MSRV is 1.82 (uses `Iterator::repeat_n`, `i64::div_ceil`).

End-to-end smoke test against the real model lives via the CLI:

```sh
# Auto-fetches files into ~/.cache/huggingface/hub on first call.
./target/release/hy-mt translate --repo AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF \
    --tgt Spanish --prompt "Hello, world!"   # → ¡Hola, mundo!

./target/release/hy-mt inspect  --repo <ID>   # works on any of the 3 supported repos;
                                              # prints metadata + tensor index
./target/release/hy-mt debug-forward --repo <ID> --prompt "..." \
    --chat-translate Spanish --decode-steps 8 --top-k 6   # diagnostic single-step
```

The HF cache is **shared with other tools** — never `rm -rf ~/.cache/huggingface/hub`.

## High-level architecture

The crate inferences Tencent's Hunyuan-MT 1.5 (1.8B) translation model. Three model formats are supported through a single trait:

```
hf_hub fetch / local path → DiscoveredFormat ──┬→ HyGgufFile     ─┐
                                               │                  ├→ ModelSource → HunyuanDense::load_from
                                               └→ HySafetensors  ─┘
```

`ModelSource` (`hy-mt-core/src/source.rs`) is the only thing the model loader sees — it asks for `load_role(role, dev)` and gets back a `WeightStore`. **Adding a new format = one new file implementing the trait, no model code changes.** Tensor naming differences live in `model/layout.rs` as bidirectional `TensorRole ↔ name` functions for both GGUF (`blk.N.attn_q.weight`) and HF (`model.layers.N.self_attn.q_proj.weight`) conventions.

### STQ1_0 (1.25-bit) codec

Three-quarters of the project's complexity. Block layout (`quant/block.rs`): 256 weights → 42 bytes (`qs[32]` + `sign[8]` + fp16 scale `d`); ternary values `{-d, 0, +d}` with 3:4 sparsity decoded through a 32-entry codebook (`quant/codebook.rs`, copied verbatim from llama.cpp PR #22836). The codec is exercised by unit tests across `tests/stq_*.rs` plus an in-module cross-validator (`quant/matmul.rs::tests`) that pins the dispatched SIMD kernel to the scalar reference.

**`WeightStore` has two variants** (`weights.rs`):

- `Stq1_0 { blocks: BlockSource, rows, cols }` — packed 1.25-bit. The inner `BlockSource` is either an `Arc<Mmap>` view (production GGUF path, **zero-copy** — no 280 MB of `to_vec` per load) or an `Arc<[BlockStq1_0]>` (synthetic / non-mmap loaders).
- `Tensor(Tensor)` — already a Candle tensor; used for everything on Metal/CUDA, all F32/F16/BF16 GGUF tensors, the embedding (which is Q6K in GGUF, BF16 in safetensors, dequantized at load), and norms.

Dispatch happens at load time based on `DeviceCtx::supports_stq1_0_native()` (true only for CPU). On Metal/CUDA, `WeightStore::from_view` eagerly dequantizes STQ1_0 → F16 before uploading. On the GGUF + CPU path, `HyGgufFile::load_role` builds a zero-copy `BlockSource::Mmap` directly from the file's tensor data offset.

**SIMD kernels** (`quant/matmul_neon.rs`, `quant/matmul_avx2.rs`):

- `aarch64`: hand-written NEON via `vfmaq_f32` + dual accumulators. Always dispatched (NEON is part of the AArch64 base ABI, so no runtime check). On Apple-silicon hosts NEON gives ~15-30 % over the auto-vectorised scalar path — `#[inline]` is required because `#[target_feature]` would otherwise force a function call across the per-row hot path.
- `x86_64`: AVX2 + FMA via `_mm256_fmadd_ps` with 256-bit accumulator. Dispatched once per process via `is_x86_feature_detected!` (cached in `AtomicU8`).
- non-SIMD targets fall back to `stq_dot_row_f32_scalar` (auto-vectorisation-friendly).

**Q8 activation packing** (`quant/q8.rs`, opt-in via `HY_MT_USE_Q8=1`): pre-quantizes activations to per-block int8 + f32 scale before the matmul, shrinking activation memory traffic 4× and replacing per-lane f32 FMAs with `vmull_s8`/`vaddq_s16` widening multiplies. The kernel is correct and parallelised across output rows for the m=1 decode path (matches `stq_matmul_f32`'s short-circuit), but on Apple-silicon hosts it is currently ~10 % slower end-to-end than the f32 path because Rust stable does not yet expose `vdotq_s32` (issue #117224); flip the flag back on once `stdarch_neon_dotprod` stabilises.

### Compute dtype policy

Picked once in `transformer.rs::HunyuanDense::load_from`:

- **CPU** → F32 (matches the STQ1_0 custom kernel)
- **Metal/CUDA** → F16

All embeddings, norm weights, RoPE tables, and (on Metal) dequantized linear weights are cast to `compute_dtype` at load. `QuantLinear::forward` (`model/linear.rs`) preserves input dtype on output so residuals don't blow up — historically a source of "dtype mismatch in add" errors. The `safetensors_loader` casts BF16 → compute_dtype eagerly because Candle has no CPU BF16 matmul.

### Architecture-specific quirks

These are **not optional** — getting any of them wrong silently produces broken-but-plausible output (caught during real-model testing):

1. **RoPE is rotate-half (`candle_nn::rotary_emb::rope`), not interleaved.** The interleaved variant works for most languages but breaks Russian/Ukrainian morphology. See `rope.rs` module docstring.
2. **Special tokens use full-width vertical bars `｜` (U+FF5C)**, not ASCII `|`. The chat template format is `<｜hy_begin▁of▁sentence｜><｜hy_User｜>{user}<｜hy_Assistant｜>`. ASCII `<|extra_0|>` from older Hunyuan models is wrong — the BPE tokenizer would split it into characters. `tokenizer.rs` looks up the real IDs by their full-width name.
3. **QK-norm is per-head with weight `[head_dim]`** (length 128 for this model), applied to Q and K *after* the head-reshape and *before* RoPE. HF tensor names are `query_layernorm` / `key_layernorm`, not `q_norm` / `k_norm` — `from_hf_name` accepts both.
4. **`rope_theta` must be the NTK-aware scaled base (~11_158_840), not the raw 10_000.** Hunyuan was trained with alpha-NTK base scaling **always** applied (not just past `max_position_embeddings`), so the model expects the scaled base on every position — mixing it up shifts attention scores enough to flip argmax on close calls (e.g. Ukrainian `друже` vs `другу` for vocative). Two paths feed `RopeCache::new` an already-effective base:
   - **GGUF** — `rope_freq_base` in metadata is `11_158_840`, baked in by AngelSlim using `base * alpha^(d/(d-2))` with `alpha=1000`, `d=128`.
   - **HF safetensors** — `config.json` carries the raw `rope_theta=10_000` plus `rope_scaling.alpha=1000`. `HunyuanConfig::from_hf_config` applies the same formula at load time, producing `~11_158_844` (sub-ULP rounding diff vs the GGUF metadata, harmless).
5. **Tied LM head** is the default; `output.weight` is absent in GGUF and silently uses the embedding transpose. The HF safetensors file *also* contains `lm_head.weight` even when `tie_word_embeddings = true` — we follow the config flag, not the tensor's presence.

### GGUF reader

Vendored from `candle_core::quantized::gguf_file` (Apache-2.0 attribution preserved at the top of `gguf/vendored.rs`). The local `GgmlDTypeExt` adds `Stq1_0` (type id `40`, with `42` accepted as a fallback for early PR drafts) — Candle's upstream enum is closed and rejects type 40 outright. K-quants (Q4_K, Q6_K, etc.) are passed through to Candle's `qtensor_from_ggml` decoder rather than reimplemented; only STQ1_0 has its own codec.

### Tokenizer

Two construction paths in `tokenizer.rs`:

- `HyTokenizer::from_file(path)` — loads HF `tokenizer.json` (used by safetensors).
- `HyTokenizer::from_gguf(content)` — builds the BPE tokenizer entirely from `tokenizer.ggml.tokens` / `merges` / `token_type` arrays in the GGUF header (assumes `tokenizer.ggml.model = "gpt2"`). Standard ByteLevel pre/post/decoder; CONTROL tokens (type 3) registered as `special`, USER_DEFINED (type 4) as non-special.

CLI's `build_tokenizer(format)` chooses automatically: GGUF without an external `tokenizer.json` → embedded path (zero-config); safetensors always uses the `tokenizer.json` next to it (HF format requires it).

### Generator

`generate.rs` does **batched prefill**: the entire prompt is fed through `model.forward([1, T])` in one call (no per-token loop), since `HunyuanDense::forward` already returns logits for the last position only. Decode steps continue with single-token forwards on the warm KV cache. KvCache::append works for arbitrary `[1, K]` shapes because it's just `Tensor::cat` along the time axis.

### CLI shape (`hy-mt-cli/src/main.rs`)

A single `SourceArgs` struct (flattened into `TranslateArgs` / `InspectArgs` / `DebugForwardArgs`) handles `--repo` / `--revision` / `--model` / `--tokenizer` and resolves into a `DiscoveredFormat`. `DiscoveredFormat::Gguf::tokenizer` is `Option<PathBuf>` — `None` means use the GGUF-embedded vocab. `--device auto` (default) picks Metal → CUDA → CPU based on compile-time features and runtime availability. Translation streams tokens to stdout via incremental batch decode (decode the full id list each step, write only the new tail) — keeps UTF-8 boundaries intact for Cyrillic/CJK without duplicating the output.

## Defaults that matter

`--temperature 0.0` (greedy) is the default for translation — anything higher introduces stochastic word-form drift in morphologically rich languages. `--repeat-penalty 1.1` (greedy needs it; with 1.0 the model loops on long inputs by repeating the same stem indefinitely).

**The three supported repos are not interchangeable** even after the RoPE fix above:
- `tencent/HY-MT1.5-1.8B` — original Tencent BF16 checkpoint.
- `AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF` — AngelSlim's STQ1_0 1.25-bit packing of their own QAT (quantization-aware) fine-tune.
- `AngelSlim/Hy-MT1.5-1.8B-1.25bit` — same QAT fine-tuned weights, dequantized to BF16 safetensors.

The two AngelSlim repos contain **the same model**; their outputs match token-for-token after the RoPE fix. They differ from `tencent/HY-MT1.5-1.8B` because AngelSlim ran a separate QAT pass — that's where output divergences between Tencent BF16 and AngelSlim-anything come from, not from quantization itself.
