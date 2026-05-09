//! `hy-mt`: command-line driver for Hy-MT 1.5 translation.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

/// Hard cap on stdin input for translation. 16 MiB of UTF-8 is well above
/// any reasonable single-shot translation request and stops a `cat huge.bin`
/// pipe from OOM-ing the process before tokenisation even starts.
const MAX_STDIN_BYTES: u64 = 16 * 1024 * 1024;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use hy_mt_core::device::DeviceCtx;
use hy_mt_core::generate::Generator;
use hy_mt_core::gguf::HyGgufFile;
use hy_mt_core::hub::{detect_local, fetch_model, DiscoveredFormat, HubRef};
use hy_mt_core::model::HunyuanDense;
use hy_mt_core::safetensors_loader::HySafetensors;
use hy_mt_core::sampling::SamplingParams;
use hy_mt_core::tokenizer::HyTokenizer;

const DEFAULT_REPO: &str = "AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF";

#[derive(Parser, Debug)]
#[command(name = "hy-mt", version, about = "Hunyuan-MT 1.5 translation CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Translate a single string (or stdin if `--prompt` is omitted).
    Translate(TranslateArgs),
    /// Print model metadata and tensor descriptors for diagnostics.
    Inspect(InspectArgs),
    /// Run a single forward step on a hand-tokenized prompt and print the
    /// top-K argmax logits — useful for debugging numeric correctness.
    DebugForward(DebugForwardArgs),
}

/// Shared `--repo`/`--model`/`--tokenizer`/`--revision` flags.
///
/// Use either `--repo <ID>` to auto-fetch from HuggingFace Hub or
/// `--model <PATH>` for a local GGUF file or directory containing
/// safetensors. `--tokenizer` overrides the tokenizer.json path;
/// it's auto-resolved from the discovered format otherwise.
#[derive(Args, Debug, Clone)]
struct SourceArgs {
    /// HuggingFace repo id (e.g. AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF).
    /// Defaults to the GGUF AngelSlim repo when neither --repo nor --model
    /// is provided.
    #[arg(long)]
    repo: Option<String>,

    /// Optional repo revision (branch/tag/commit). Defaults to `main`.
    #[arg(long)]
    revision: Option<String>,

    /// Local model path. Either a `.gguf` file or a directory containing
    /// `model.safetensors` / shard layout. Mutually exclusive with --repo.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Optional tokenizer.json path. Required when --model points at a file
    /// without a sibling `tokenizer.json` and --repo is not set.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

impl SourceArgs {
    fn resolve(&self) -> Result<DiscoveredFormat> {
        if let Some(model) = &self.model {
            if self.repo.is_some() {
                anyhow::bail!("pass either --repo or --model, not both");
            }
            return detect_local(model, self.tokenizer.as_deref())
                .with_context(|| format!("inspecting local model {}", model.display()));
        }
        let repo_id = self.repo.clone().unwrap_or_else(|| DEFAULT_REPO.into());
        let mut hub = HubRef::new(repo_id.clone());
        if let Some(rev) = &self.revision {
            hub = hub.with_revision(rev.clone());
        }
        let mut format = fetch_model(&hub).with_context(|| format!("fetching repo `{repo_id}`"))?;
        if let Some(custom_tokenizer) = &self.tokenizer {
            override_tokenizer(&mut format, custom_tokenizer);
        }
        Ok(format)
    }
}

fn override_tokenizer(fmt: &mut DiscoveredFormat, tokenizer: &Path) {
    match fmt {
        DiscoveredFormat::Gguf { tokenizer: t, .. } => *t = Some(tokenizer.to_path_buf()),
        DiscoveredFormat::Safetensors { tokenizer: t, .. } => *t = tokenizer.to_path_buf(),
    }
}

/// Build the tokenizer for a discovered format, preferring an external
/// `tokenizer.json` when one was supplied/discovered, else falling back to
/// the GGUF-embedded vocab.
fn build_tokenizer(fmt: &DiscoveredFormat) -> Result<HyTokenizer> {
    match fmt {
        DiscoveredFormat::Gguf { gguf, tokenizer } => match tokenizer {
            Some(path) => HyTokenizer::from_file(path)
                .with_context(|| format!("loading tokenizer {}", path.display())),
            None => {
                let f = HyGgufFile::load(gguf)
                    .with_context(|| format!("opening {}", gguf.display()))?;
                HyTokenizer::from_gguf(f.content()).context("building tokenizer from GGUF metadata")
            }
        },
        DiscoveredFormat::Safetensors { tokenizer, .. } => HyTokenizer::from_file(tokenizer)
            .with_context(|| format!("loading tokenizer {}", tokenizer.display())),
    }
}

#[derive(Parser, Debug, Clone)]
struct TranslateArgs {
    #[command(flatten)]
    src: SourceArgs,

    /// Target language name (e.g. "Russian", "Spanish", "Japanese").
    #[arg(long = "tgt")]
    tgt: String,

    /// Override the user-message template (use {tgt} and {text} placeholders).
    #[arg(long)]
    instruction: Option<String>,

    /// Compute backend. `auto` (default) picks the best available
    /// (Metal → CUDA → CPU) given the compile-time features.
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Text to translate. If absent, the program reads from stdin.
    #[arg(long)]
    prompt: Option<String>,

    /// Maximum number of tokens to generate. EOS terminates earlier.
    #[arg(long, default_value_t = 1024)]
    max_new_tokens: usize,

    /// Sampling temperature. The default `0.0` is greedy decoding —
    /// the standard for machine-translation quality. Raise it (e.g. 0.7)
    /// only if you want stylistic variety at the cost of accuracy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    /// Discourage immediate repetitions by dividing recently-seen logits.
    /// Defaults to a mild 1.1; lower for poetry/repetition-heavy source,
    /// raise for chatty models.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 64)]
    repeat_window: usize,
    #[arg(long, default_value_t = 0xCAFE_BABE)]
    seed: u64,
}

#[derive(Parser, Debug, Clone)]
struct InspectArgs {
    #[command(flatten)]
    src: SourceArgs,

    /// Show full tensor list (default: only summary by ggml_type and a sample).
    #[arg(long)]
    tensors: bool,

    /// If set, dump the full content of this metadata key (no truncation).
    #[arg(long)]
    key: Option<String>,
}

#[derive(Parser, Debug, Clone)]
struct DebugForwardArgs {
    #[command(flatten)]
    src: SourceArgs,

    /// Raw text to encode (no chat template applied).
    #[arg(long)]
    prompt: String,

    /// If set, wrap `prompt` with the BOS/USER/ASSISTANT chat template and a
    /// "Translate into <tgt>" instruction prefix.
    #[arg(long)]
    chat_translate: Option<String>,

    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 5)]
    decode_steps: usize,
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceArg {
    /// Pick the best available backend at runtime: Metal → CUDA → CPU,
    /// gated by the features the binary was compiled with.
    Auto,
    Cpu,
    Metal,
    Cuda,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Translate(args) => translate(args),
        Command::Inspect(args) => inspect(args),
        Command::DebugForward(args) => debug_forward(args),
    }
}

fn build_device(arg: DeviceArg) -> Result<DeviceCtx> {
    match arg {
        DeviceArg::Cpu => Ok(DeviceCtx::cpu()),
        DeviceArg::Metal => DeviceCtx::metal(0).context("initialising Metal device"),
        DeviceArg::Cuda => DeviceCtx::cuda(0).context("initialising CUDA device"),
        DeviceArg::Auto => {
            // Prefer Metal on macOS, then CUDA, falling back to CPU.
            // `cfg!(feature = "...")` would compile every branch in regardless
            // of the active features; #[cfg] gates the calls so a `cuda`-less
            // build doesn't try to load nvcuda. The caller logs which
            // backend was chosen (`selected=` field on "using compute device"),
            // so this branch only emits debug-level fallback notes.
            #[cfg(feature = "metal")]
            {
                match DeviceCtx::metal(0) {
                    Ok(d) => return Ok(d),
                    Err(e) => tracing::debug!("Metal unavailable, falling back: {e}"),
                }
            }
            #[cfg(feature = "cuda")]
            {
                match DeviceCtx::cuda(0) {
                    Ok(d) => return Ok(d),
                    Err(e) => tracing::debug!("CUDA unavailable, falling back: {e}"),
                }
            }
            Ok(DeviceCtx::cpu())
        }
    }
}

/// Open a `DiscoveredFormat` as a model. Supports GGUF and safetensors.
fn open_model(fmt: &DiscoveredFormat, dev: &DeviceCtx) -> Result<HunyuanDense> {
    match fmt {
        DiscoveredFormat::Gguf { gguf, .. } => {
            let f =
                HyGgufFile::load(gguf).with_context(|| format!("loading {}", gguf.display()))?;
            HunyuanDense::load_from(&f, dev).context("loading model weights")
        }
        DiscoveredFormat::Safetensors { config, shards, .. } => {
            let st = HySafetensors::load(config, shards)
                .with_context(|| format!("loading safetensors model from {}", config.display()))?;
            HunyuanDense::load_from(&st, dev).context("loading model weights")
        }
    }
}

fn translate(args: TranslateArgs) -> Result<()> {
    let dev = build_device(args.device)?;
    tracing::info!(?args.device, selected = ?dev.kind, "using compute device");

    let text = match &args.prompt {
        Some(t) => t.clone(),
        None => {
            let mut buf = Vec::new();
            std::io::stdin()
                .take(MAX_STDIN_BYTES + 1)
                .read_to_end(&mut buf)
                .context("reading stdin")?;
            if buf.len() as u64 > MAX_STDIN_BYTES {
                anyhow::bail!(
                    "stdin input exceeds {} bytes — pass --prompt for shorter texts \
                     or split the source",
                    MAX_STDIN_BYTES
                );
            }
            String::from_utf8(buf)
                .context("stdin is not valid UTF-8")?
                .trim()
                .to_string()
        }
    };
    if text.is_empty() {
        anyhow::bail!("empty input text — pass --prompt or pipe data on stdin");
    }

    let format = args.src.resolve()?;
    let mut model = open_model(&format, &dev)?;
    tracing::info!(layers = model.config.n_layers, "model loaded");

    let tokenizer = build_tokenizer(&format)?;
    let vocab = tokenizer.vocab_size();
    let model_vocab = model.config.vocab_size;
    if vocab != model_vocab {
        // Vocab-size mismatch is common: HF tokenizers often declare a few
        // extra special tokens beyond the embedding rows. We still warn,
        // but the hard check happens against the *encoded prompt ids*
        // below — that's what would actually trigger an out-of-range
        // embedding lookup.
        tracing::warn!(
            tokenizer_vocab = vocab,
            model_vocab,
            "tokenizer vocab size differs from model — proceeding (encoded ids \
             will be validated against model vocab before forward)"
        );
    }

    let sampling = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_window: args.repeat_window,
        seed: args.seed,
    };

    let mut gen = Generator::new(&mut model, &tokenizer, sampling, args.max_new_tokens);
    let prompt = if let Some(tmpl) = &args.instruction {
        let user_msg = tmpl.replace("{tgt}", &args.tgt).replace("{text}", &text);
        tokenizer.apply_chat_template(None, &user_msg)?
    } else {
        tokenizer
            .build_translate_prompt(&args.tgt, &text)
            .context("building prompt")?
    };
    // Hard-check that every encoded id fits in the model's embedding
    // table. This catches the rare case where a tokenizer with extra
    // sentinel tokens actually produces one of those ids — without it
    // we'd panic deep inside `embed.index_select`.
    if let Some(&bad) = prompt.iter().find(|&&id| (id as usize) >= model_vocab) {
        anyhow::bail!(
            "prompt contains token id {bad} ≥ model vocab {model_vocab}; \
             the tokenizer is not compatible with this model"
        );
    }

    // Stream tokens straight to stdout. The Generator already produces
    // UTF-8-safe incremental text in `step.text` (see Generator's
    // incremental decode), so we only need to write and flush.
    let stdout_is_tty = std::io::stdout().is_terminal();
    let mut stdout = std::io::stdout().lock();

    gen.generate(&prompt, |step| {
        if !step.text.is_empty() {
            let _ = stdout.write_all(step.text.as_bytes());
            if stdout_is_tty {
                let _ = stdout.flush();
            }
        }
        true
    })?;

    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();
    Ok(())
}

fn inspect(args: InspectArgs) -> Result<()> {
    let format = args.src.resolve()?;
    match &format {
        DiscoveredFormat::Gguf { gguf, .. } => inspect_gguf(gguf, &args)?,
        DiscoveredFormat::Safetensors { config, shards, .. } => {
            inspect_safetensors(config, shards, &args)?
        }
    }
    Ok(())
}

fn inspect_safetensors(config_path: &Path, shards: &[PathBuf], args: &InspectArgs) -> Result<()> {
    use safetensors::SafeTensors;
    use std::collections::BTreeMap;
    use std::fs::File;

    println!("Format: safetensors ({} shard(s))", shards.len());
    println!();

    // ---- config.json -----------------------------------------------------
    if config_path.exists() {
        let raw = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", config_path.display()))?;
        if let Some(k) = &args.key {
            match json.get(k) {
                None => anyhow::bail!("key {k:?} not found in config.json"),
                Some(v) => {
                    let pretty = serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string());
                    println!("{pretty}");
                }
            }
            return Ok(());
        }
        println!("== config.json ==");
        if let Some(obj) = json.as_object() {
            let mut entries: Vec<(&String, &serde_json::Value)> = obj.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (k, v) in entries {
                let v_str = match v {
                    serde_json::Value::String(s) => {
                        let snippet = if s.len() > 80 {
                            format!("{}…", &s[..80])
                        } else {
                            s.clone()
                        };
                        format!("{snippet:?}")
                    }
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        let s = v.to_string();
                        if s.len() > 80 {
                            format!("{}…", &s[..80])
                        } else {
                            s
                        }
                    }
                    _ => v.to_string(),
                };
                println!("  {k} = {v_str}");
            }
        }
        println!();
    } else if args.key.is_some() {
        anyhow::bail!("config.json missing; cannot resolve --key");
    }

    // ---- safetensors header(s) ------------------------------------------
    // Keep the mmaps alive while parsing so the deserialiser can borrow
    // their bytes; previously the code copied each shard into a `Vec<u8>`
    // (via `mmap.to_vec()`), which materialises the entire 4 GB+ shard in
    // RAM. mmap-then-borrow is O(1) memory.
    let mut by_dtype: BTreeMap<String, usize> = BTreeMap::new();
    let mut all_names: Vec<(String, String, Vec<usize>, usize)> = Vec::new(); // (name, dtype, shape, bytes)

    let mut mmaps: Vec<memmap2::Mmap> = Vec::with_capacity(shards.len());
    for shard in shards {
        let f = File::open(shard).with_context(|| format!("opening {}", shard.display()))?;
        // SAFETY: file is treated as read-only for the duration of this
        // function; no concurrent writer.
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        mmaps.push(mmap);
    }
    for (shard, mmap) in shards.iter().zip(mmaps.iter()) {
        let st = SafeTensors::deserialize(&mmap[..])
            .with_context(|| format!("parsing safetensors {}", shard.display()))?;
        for (name, info) in st.tensors() {
            let dtype = format!("{:?}", info.dtype());
            *by_dtype.entry(dtype.clone()).or_default() += 1;
            all_names.push((
                name.to_string(),
                dtype,
                info.shape().to_vec(),
                info.data().len(),
            ));
        }
    }

    println!("== Tensors ({}) ==", all_names.len());
    for (k, v) in &by_dtype {
        println!("  {v:6} × {k}");
    }

    // Always show a sample of names so the user can see the layout pattern.
    println!();
    println!("First 10 tensor names:");
    let mut sorted = all_names.clone();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (n, d, s, _) in sorted.iter().take(10) {
        println!("  {n:<60} {d:<10} {s:?}");
    }

    // Heuristic packed-format hints — surface anything that looks like a
    // GPTQ/AWQ/STQ-style auxiliary tensor so we can decide how to load.
    let suffix_hints = [
        "qweight", "qs", "sign", "qzeros", "scales", "zeros", "g_idx",
    ];
    let mut hint_groups: BTreeMap<&str, usize> = BTreeMap::new();
    for (n, _, _, _) in &all_names {
        for suffix in suffix_hints {
            if n.ends_with(&format!(".{suffix}")) || n.ends_with(suffix) {
                *hint_groups.entry(suffix).or_default() += 1;
            }
        }
    }
    if !hint_groups.is_empty() {
        println!();
        println!("Packed-format hints:");
        for (k, v) in hint_groups {
            println!("  *{k}: {v}");
        }
    }

    if args.tensors {
        println!();
        for (n, d, s, b) in sorted {
            println!("  {n:<60} {d:<10} {s:?}  {b} bytes");
        }
    }

    Ok(())
}

fn inspect_gguf(path: &Path, args: &InspectArgs) -> Result<()> {
    let gguf = HyGgufFile::load(path).with_context(|| format!("loading {}", path.display()))?;
    let content = gguf.content();

    if let Some(k) = &args.key {
        match content.metadata.get(k) {
            None => anyhow::bail!("key {k:?} not found"),
            Some(hy_mt_core::gguf::vendored::Value::String(s)) => println!("{s}"),
            Some(hy_mt_core::gguf::vendored::Value::Array(a)) => {
                for v in a {
                    println!("{v:?}");
                }
            }
            Some(other) => println!("{other:?}"),
        }
        return Ok(());
    }

    println!("Magic/version: {:?}", content.magic);
    println!("Tensor data offset: {}", content.tensor_data_offset);
    println!();
    println!("== Metadata ({} keys) ==", content.metadata.len());
    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &content.metadata[k];
        let preview = match v {
            hy_mt_core::gguf::vendored::Value::Array(a) => format!(
                "Array(len={}, type={:?})",
                a.len(),
                a.first().map(|e| e.value_type())
            ),
            hy_mt_core::gguf::vendored::Value::String(s) => {
                let snippet = if s.len() > 80 {
                    format!("{}…", &s[..80])
                } else {
                    s.clone()
                };
                format!("String({snippet:?})")
            }
            other => format!("{other:?}"),
        };
        println!("  {k} = {preview}");
    }
    println!();
    println!("== Tensors ({}) ==", content.tensor_infos.len());
    let mut by_dtype: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for info in content.tensor_infos.values() {
        *by_dtype
            .entry(format!("{:?}", info.ggml_dtype))
            .or_default() += 1;
    }
    for (k, v) in by_dtype {
        println!("  {v:6} × {k}");
    }
    if args.tensors {
        println!();
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in names {
            let info = &content.tensor_infos[name];
            println!(
                "  {name:<40} {:?} {:?} @ +{}",
                info.ggml_dtype, info.shape, info.offset
            );
        }
    }
    Ok(())
}

fn debug_forward(args: DebugForwardArgs) -> Result<()> {
    let dev = build_device(args.device)?;
    let format = args.src.resolve()?;
    let mut model = open_model(&format, &dev)?;
    let tokenizer = build_tokenizer(&format)?;

    let prompt_ids = if let Some(tgt) = &args.chat_translate {
        tokenizer.build_translate_prompt(tgt, &args.prompt)?
    } else {
        tokenizer.encode_raw(&args.prompt)?
    };
    println!("prompt = {:?}", args.prompt);
    if let Some(tgt) = &args.chat_translate {
        println!("(chat-template applied; tgt={tgt:?})");
    }
    println!("token ids ({} tokens):", prompt_ids.len());
    for &id in &prompt_ids {
        let s = tokenizer.decode_with_special(&[id]).unwrap_or_default();
        println!("  {id:>6}  {s:?}");
    }
    println!();

    use candle_core::{DType, Tensor};
    model.reset_kv_cache(prompt_ids.len() + args.decode_steps)?;
    let mut history: Vec<u32> = prompt_ids.clone();

    let mut last_logits: Vec<f32> = Vec::new();
    for &tok in &prompt_ids {
        let t = Tensor::from_vec(vec![tok], (1, 1), &dev.device)?;
        let logits = model.forward(&t)?;
        last_logits = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
    }

    println!("=== logits after prompt (step 0) ===");
    print_topk(&last_logits, args.top_k, &tokenizer);

    for step in 1..=args.decode_steps {
        let argmax = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| anyhow::anyhow!("empty logits vector"))?;
        history.push(argmax);
        let txt = tokenizer.decode_with_special(&[argmax]).unwrap_or_default();
        println!("step {step}: pick id={argmax} {txt:?}");

        let t = Tensor::from_vec(vec![argmax], (1, 1), &dev.device)?;
        let logits = model.forward(&t)?;
        last_logits = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        print_topk(&last_logits, args.top_k, &tokenizer);
    }

    let full = tokenizer
        .decode_with_special(&history[prompt_ids.len()..])
        .unwrap_or_default();
    println!();
    println!("Generated: {full:?}");
    Ok(())
}

fn print_topk(logits: &[f32], k: usize, tokenizer: &HyTokenizer) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in idx.iter().take(k) {
        let s = tokenizer
            .decode_with_special(&[i as u32])
            .unwrap_or_default();
        println!("  id={i:<7} {:>10.4}  {s:?}", logits[i]);
    }
}
