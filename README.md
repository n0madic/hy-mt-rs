# Hy-MT1.5-rs

Pure-Rust translation engine for **Tencent Hunyuan-MT 1.5 1.8B**, running the
ultra-compressed **1.25-bit STQ1_0** GGUF released by AngelSlim
([HF model](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF) –
**440 MB on disk**).

Built on top of [`huggingface/candle`](https://github.com/huggingface/candle),
with a custom port of the STQ1_0 codec from
[ggml-org/llama.cpp#22836](https://github.com/ggml-org/llama.cpp/pull/22836).
Both **CPU** and **macOS Metal** are supported out of the box.

## Quick start

```sh
# Build
cargo build --release                    # CPU only
cargo build --release --features metal   # Apple Silicon GPU

# Translate — files are auto-fetched into ~/.cache/huggingface/hub on first use
./target/release/hy-mt translate \
    --repo AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF \
    --tgt Spanish \
    --prompt "Hello, world!"
# → ¡Hola, mundo!
```

### Choosing a model variant

| Repository | Format | Size | Use when |
|---|---|---|---|
| `AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF` (default) | GGUF, 1.25-bit STQ1_0 | 440 MB | RAM-constrained, batched server |
| `AngelSlim/Hy-MT1.5-1.8B-1.25bit` | safetensors, BF16 | 3.6 GB | want full BF16 weights via HF tooling |
| `tencent/HY-MT1.5-1.8B` | safetensors, BF16 (official) | 3.6 GB | best quality, 1.25-bit not needed |

```sh
hy-mt translate --repo tencent/HY-MT1.5-1.8B          --tgt Ukrainian --prompt "Good morning, my friend."
hy-mt translate --repo AngelSlim/Hy-MT1.5-1.8B-1.25bit --tgt Spanish   --prompt "Artificial intelligence is changing the world rapidly."
```

If you'd rather download manually:

```sh
# GGUF: zero-config — the tokenizer is embedded in the file.
curl -L -o Hy-MT1.5-1.8B-1.25bit.gguf \
  https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF/resolve/main/Hy-MT1.5-1.8B-1.25bit.gguf
hy-mt translate --model ./Hy-MT1.5-1.8B-1.25bit.gguf --tgt Spanish --prompt "Hello, world!"

# safetensors: needs tokenizer.json + config.json next to the .safetensors file.
```

## Translation examples

```sh
hy-mt translate ... --tgt Japanese --prompt "Where is the train station?"
# → 駅はどこですか？

hy-mt translate ... --tgt Spanish --prompt "Artificial intelligence is changing the world rapidly."
# → La inteligencia artificial está cambiando el mundo de manera rápida.

hy-mt translate ... --tgt French --prompt "I love you very much."
# → Je t'aime beaucoup.

hy-mt translate ... --tgt Chinese  --prompt "I love programming in Rust because it is fast and safe."
# → 我喜欢用 Rust 进行编程，因为它既快速又安全。
```

## Supported languages

The model supports **33 main languages** plus **5 Chinese-region dialects /
ethnic-minority languages**, covering 1 056 translation directions in total.
Pass the language *name* (or any of the listed aliases) to `--tgt`.

### Main languages

| Language          | `--tgt` value     | Code | | Language    | `--tgt` value | Code |
|-------------------|-------------------|------|-|-------------|---------------|------|
| Chinese           | `Chinese`         | zh   | | Korean      | `Korean`      | ko   |
| English           | `English`         | en   | | Thai        | `Thai`        | th   |
| French            | `French`          | fr   | | Italian     | `Italian`     | it   |
| Portuguese        | `Portuguese`      | pt   | | German      | `German`      | de   |
| Spanish           | `Spanish`         | es   | | Vietnamese  | `Vietnamese`  | vi   |
| Japanese          | `Japanese`        | ja   | | Malay       | `Malay`       | ms   |
| Turkish           | `Turkish`         | tr   | | Indonesian  | `Indonesian`  | id   |
| Russian           | `Russian`         | ru   | | Filipino    | `Filipino`    | tl   |
| Arabic            | `Arabic`          | ar   | | Hindi       | `Hindi`       | hi   |
| Polish            | `Polish`          | pl   | | Khmer       | `Khmer`       | km   |
| Czech             | `Czech`           | cs   | | Burmese     | `Burmese`     | my   |
| Dutch             | `Dutch`           | nl   | | Persian     | `Persian`     | fa   |
| Hebrew            | `Hebrew`          | he   | | Gujarati    | `Gujarati`    | gu   |
| Bengali           | `Bengali`         | bn   | | Urdu        | `Urdu`        | ur   |
| Tamil             | `Tamil`           | ta   | | Telugu      | `Telugu`      | te   |
| Ukrainian         | `Ukrainian`       | uk   | | Marathi     | `Marathi`     | mr   |

### Dialects & ethnic-minority languages

| Language            | `--tgt` value          | Code     |
|---------------------|------------------------|----------|
| Traditional Chinese | `Traditional Chinese`  | zh-Hant  |
| Cantonese           | `Cantonese`            | yue      |
| Tibetan             | `Tibetan`              | bo       |
| Kazakh              | `Kazakh`               | kk       |
| Mongolian           | `Mongolian`            | mn       |
| Uyghur              | `Uyghur`               | ug       |

> The model accepts any natural-language target name — it translates
> *into* whatever language you ask, in mostly any source language. The
> table is the official set the authors trained on; results may vary for
> low-resource pairs and exotic register/dialect requests.

## CLI reference

```
hy-mt translate
    --tgt <LANG>               # Target language name (see table above)
  [ --repo <ID> ]              # HuggingFace repo id (auto-fetched into the
                               # standard ~/.cache/huggingface/hub cache)
  [ --revision <REV> ]         # Repo branch/tag/commit (defaults to main)
  [ --model <PATH> ]           # Local .gguf file or directory with safetensors
  [ --tokenizer <PATH> ]       # Override tokenizer.json (auto-resolved otherwise)
  [ --device auto|cpu|metal|cuda ] # Compute backend. `auto` (default) picks
                               # the best available — Metal → CUDA → CPU —
                               # gated by the features the binary was compiled
                               # with (`--features metal` / `--features cuda`).
  [ --prompt <TEXT> ]          # Source text; if omitted, reads stdin
  [ --max-new-tokens N ]       # Hard cap on output length (default 1024)
  [ --temperature F ]          # Sampling temperature (default 0.0 = greedy,
                               # the standard for translation quality;
                               # raise to ~0.7 only for stylistic variety)
  [ --top-k N ] [ --top-p F ]  # Truncation samplers (off by default)
  [ --repeat-penalty F ]       # Defaults to 1.1 (greedy needs ≥ 1.05 to
                               # avoid loops on long morphologically-rich text)
  [ --repeat-window N ]        # How many recent tokens the penalty looks at
                               # (default 64)
  [ --instruction "TMPL" ]     # Override the user-message template;
                               # placeholders {tgt} and {text} are replaced
  [ --seed N ]                 # RNG seed for reproducibility (default
                               # 0xCAFEBABE; only matters at temperature > 0)
```

Environment variables:
- `HF_HOME` / `HF_TOKEN` — honoured by the auto-fetcher for the standard
  HuggingFace cache locations and gated repos.
- `HY_MT_USE_Q8=1` — switch the CPU STQ1_0 matmul to a per-block int8
  activation kernel (memory traffic ÷ 4). Currently ~10 % slower
  end-to-end on Apple silicon than the f32 path while Rust stable does
  not yet expose `vdotq_s32`; experimental.

`hy-mt inspect --repo <ID>` (or `--model FILE`) auto-detects format and prints
either the GGUF header / metadata or the safetensors tensor index plus the
parsed `config.json` — the same probe used during development to choose the
loader. `hy-mt debug-forward …` is the diagnostic single-step forward used
during development.

Streaming preview is written to **stderr**; the final batch-decoded
translation is on **stdout** (so `hy-mt translate ... | xargs -I{} echo "[{}]"`
just works). The default `--temperature 0` already gives deterministic
output, so the pipe is safe by default.

## Use as a library

The core engine lives in `hy-mt-core`. All loaders implement `ModelSource`,
so the rest of the pipeline doesn't care whether weights came from GGUF or
safetensors:

```rust
use hy_mt_core::{
    device::DeviceCtx,
    generate::Generator,
    hub::{fetch_model, DiscoveredFormat, HubRef},
    gguf::HyGgufFile,
    model::HunyuanDense,
    safetensors_loader::HySafetensors,
    sampling::SamplingParams,
    tokenizer::HyTokenizer,
};

let dev = DeviceCtx::cpu();                            // or .metal(0)?
let format = fetch_model(&HubRef::new("AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF"))?;

// GGUF carries an embedded tokenizer; safetensors always ships an
// external tokenizer.json, so the variants pick different builders.
let (mut model, tokenizer) = match &format {
    DiscoveredFormat::Gguf { gguf, tokenizer } => {
        let f = HyGgufFile::load(gguf)?;
        let tok = match tokenizer {
            Some(path) => HyTokenizer::from_file(path)?,
            None => HyTokenizer::from_gguf(f.content())?,
        };
        (HunyuanDense::load_from(&f, &dev)?, tok)
    }
    DiscoveredFormat::Safetensors { config, shards, tokenizer, .. } => {
        let st = HySafetensors::load(config, shards)?;
        (HunyuanDense::load_from(&st, &dev)?, HyTokenizer::from_file(tokenizer)?)
    }
};

let params = SamplingParams { temperature: 0.0, ..Default::default() };
let mut gen = Generator::new(&mut model, &tokenizer, params, 256);
// build_translate_prompt takes (target_lang, text); source is auto-detected.
let prompt = tokenizer.build_translate_prompt("German", "Hello, world!")?;
let ids = gen.generate(&prompt, |_| true)?;
println!("{}", tokenizer.decode(&ids)?);
```

## Performance

Measured on **Apple M4** with the 440 MB GGUF, greedy decoding,
~160-token English input → ~45-token target-language output:

| Backend                       | Cold-start load | Decode (greedy) |
|-------------------------------|-----------------|-----------------|
| CPU (`cargo build --release`) | ~0.13 s         | ~7 tok/s        |
| Metal (`--features metal`)    | ~1.7 s          | ~2.5 tok/s      |

**On Apple silicon, CPU beats Metal for this model.** STQ1_0 weights stay
packed at 440 MB on CPU and run through a hand-written NEON matvec
(`quant/matmul_neon.rs`); Metal has to dequantize the packed weights to
F16 at load time (~3.6 GB resident) and then runs Candle's generic
batch-of-1 matmul, which is shape-bound at this size. Use CPU for
interactive translation; reach for Metal/CUDA only when you can saturate
the GPU with much larger batches than this CLI ever schedules.

The CPU path is the default for a reason: zero-copy `mmap` of STQ1_0
blocks (no 280 MB `to_vec` at load), NEON-vectorised dot product on
AArch64 (AVX2 + FMA on x86_64), rayon-parallel matvec across output
rows, and direct activation slicing into the kernel without the previous
`Tensor::to_vec1`/`from_vec` round-trip. See
`hy-mt-core/benches/{stq_matmul,quant_linear}.rs` for microbenches.

## Quality notes

The 1.25-bit quantization is aggressive: short translations (greetings,
short sentences, technical one-liners) come out clean across all 33+
languages. Longer prompts in morphologically rich target languages
(e.g. Russian, Ukrainian) occasionally pick a less-natural word form.

The defaults are tuned for translation quality:
- `--temperature 0.0` (greedy) — non-zero values introduce stochastic
  word-form drift in inflected languages without improving meaning.
- `--repeat-penalty 1.1` — greedy decoding can otherwise loop on long
  morphologically-rich text (the same stem repeated indefinitely).
- Keep individual segments under ~150 tokens; for documents, split
  sentence-by-sentence.

The two AngelSlim repos (1.25-bit GGUF and the dequantized-to-BF16
safetensors) hold the **same** post-QAT weights and produce identical
translations after the loader applies the alpha-NTK base rescaling
(`base * alpha^(d/(d-2))`) on both paths. `tencent/HY-MT1.5-1.8B` is a
different checkpoint (no QAT pass) and gives slightly different word
choices, not always strictly better.

## Building blocks

The crate is split into two members:

| Crate | What it contains |
|-------|------------------|
| `hy-mt-core` | STQ1_0 codec, vendored GGUF reader, safetensors loader, hf-hub downloader, Hunyuan model, generator |
| `hy-mt-cli`  | The `hy-mt` binary used in the examples above |

Internally everything is wired through one `ModelSource` trait: GGUF
(`HyGgufFile`) and safetensors (`HySafetensors`) are interchangeable
implementations, and the model loader (`HunyuanDense::load_from`) takes any
`&dyn ModelSource`. Adding a new format is a single new file.

Cargo features (set via `--features X` on either crate):
`metal`, `cuda`, `mkl`, `accelerate` — all gated through Candle.

## Architecture cheat sheet (for the curious)

`HunYuanDenseV1ForCausalLM`, 1.8 B parameters:

- 32 transformer layers, hidden 2048, FFN 6144 (SwiGLU)
- 16 query heads / 4 KV heads (GQA 4:1), head dim 128
- RMSNorm pre-norm + per-head **QK-norm**
- Rotate-half RoPE, effective base ≈ 11 158 840 — both loaders feed the
  cache the post-NTK-rescaling base (`base * alpha^(d/(d-2))`):
  the GGUF metadata stores it directly; the HF safetensors path applies
  the formula on `rope_theta=10000` + `rope_scaling.alpha=1000` at load
- Tied input/output embeddings, vocab 120 818, context 262 144

The 1.25-bit STQ1_0 super-block packs **256 weights into 42 bytes**
(qs[32] + sign[8] + fp16 d), with ternary `{-d, 0, +d}` values and a
3:4 sparsity pattern decoded through a 32-entry codebook.

## License

The vendored `gguf_file.rs` parser keeps its upstream Apache-2.0 license
and attribution. Everything else in this repo is dual-licensed under
**MIT OR Apache-2.0**.

## References

- Model: <https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF>
- Base model (unquantized): <https://huggingface.co/tencent/HY-MT1.5-1.8B>
- llama.cpp PR with STQ1_0: <https://github.com/ggml-org/llama.cpp/pull/22836>
- Sherry paper (ACL 2026): <https://arxiv.org/abs/2601.07892>
- Hunyuan-MT 1.5 tech report: <https://arxiv.org/abs/2512.24092>
