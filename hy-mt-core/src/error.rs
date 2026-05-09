use thiserror::Error;

/// All fallible operations exposed by `hy-mt-core` return [`Result`].
pub type Result<T> = std::result::Result<T, Error>;

/// What kind of artefact was being loaded when an [`Error::Loading`] fires.
/// Library consumers can match on this to react differently to a missing
/// tokenizer vs a corrupt shard, without parsing error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoadingKind {
    /// A model weight tensor referenced by name (`hf_name`/`gguf_name`).
    Tensor,
    /// A `config.json` (HuggingFace) describing the architecture.
    Config,
    /// A `.safetensors` shard listed in `model.safetensors.index.json`.
    Shard,
    /// A `tokenizer.json` or GGUF-embedded tokenizer.
    Tokenizer,
}

/// Errors surfaced by the inference stack.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Raw I/O failure (file open, read, mmap).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// GGUF header / tensor-data parsing problem (bad magic, truncated
    /// records, malformed metadata). Reserved for the on-disk GGUF format
    /// itself — non-GGUF problems use other variants.
    #[error("invalid GGUF file: {0}")]
    Gguf(String),

    /// HuggingFace Hub interaction (download, lookup, fallback) failed.
    #[error("HuggingFace Hub: {0}")]
    Hub(String),

    /// A configuration value (HF `config.json`, command-line flag, GGUF
    /// metadata key) is missing, malformed, or out of an expected range.
    #[error("validation failed: {0}")]
    Validation(String),

    /// A `ggml`/`safetensors` dtype id we don't know how to decode.
    #[error("unsupported ggml dtype: {0:?}")]
    UnsupportedDtype(String),

    /// STQ1_0 codec invariant violation (block layout, bounds, etc.).
    #[error("invalid STQ1_0 block: {0}")]
    Stq1_0(String),

    /// A required GGUF metadata key was not present.
    #[error("missing GGUF metadata key: {0}")]
    MissingMeta(String),

    /// Tensor of unexpected rank or per-dim size encountered.
    #[error("tensor `{name}` has unexpected shape: expected {expected:?}, got {actual:?}")]
    BadShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Tokenizer load / encode / decode failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Sampling pipeline failure (e.g. degenerate softmax, distribution
    /// construction).
    #[error("sampling error: {0}")]
    Sampling(String),

    /// A path is suspicious (path traversal, absolute, empty) and must
    /// not be passed downstream.
    #[error("rejected suspicious path `{path}` ({reason})")]
    BadPath { path: String, reason: &'static str },

    /// A length/count read from input or accumulated during inference
    /// exceeds a hard cap. Used both for security-bounded GGUF parsing
    /// and runtime budgets (KV cache, prompt length).
    #[error("input exceeds bound: {what} = {got}, max = {max}")]
    OverLimit {
        what: &'static str,
        got: u64,
        max: u64,
    },

    /// Failure while loading a named artefact from a [`crate::source::ModelSource`].
    /// `kind` indicates whether it was a tensor, config, shard, or
    /// tokenizer; `source` carries the underlying Candle error.
    #[error("loading {kind:?} `{name}`: {source}")]
    Loading {
        kind: LoadingKind,
        name: String,
        #[source]
        source: candle_core::Error,
    },

    /// Pass-through for any Candle error not otherwise classified.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}
