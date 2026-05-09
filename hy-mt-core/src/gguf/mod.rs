//! Top-level GGUF loader for Hy-MT 1.5.
//!
//! Wraps the [`vendored`] reader, mmap-s the file once and exposes raw tensor
//! views typed by their `ggml` dtype. Decoding into [`crate::quant`] blocks
//! or `candle_core::Tensor` happens in `crate::weights`.

pub mod meta;
pub mod tensor_map;
pub mod vendored;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use self::vendored::{Content, GgmlDTypeExt, TensorInfo};
use crate::device::DeviceCtx;
use crate::model::config::HunyuanConfig;
use crate::model::layout::{from_gguf_name, gguf_name, TensorRole};
use crate::source::ModelSource;
use crate::weights::WeightStore;
use crate::{Error, Result};

/// Memory-mapped GGUF file with parsed header.
///
/// The struct holds the [`Mmap`] inside an `Arc` so that views into the file
/// (e.g. STQ1_0 block slices, F16 weight slices) can stay borrow-free, and
/// also caches the parsed [`HunyuanConfig`] so the model loader doesn't have
/// to re-derive it on every access.
pub struct HyGgufFile {
    mmap: Arc<Mmap>,
    content: Content,
    config: HunyuanConfig,
}

/// One tensor from a GGUF file, exposed as a typed slice borrowed from the
/// underlying mmap. The lifetime is tied to the parent [`HyGgufFile`].
///
/// Standard ggml K-quant types (Q4_K, Q6_K, …) flow through the [`Quantized`]
/// variant: they are passed straight to Candle's `qtensor_from_ggml`
/// dequantizer rather than handled here, so we don't have to re-implement
/// every k-quant codec ourselves.
pub enum TensorView<'a> {
    F32 {
        shape: Vec<usize>,
        data: &'a [f32],
    },
    F16 {
        shape: Vec<usize>,
        data: &'a [half::f16],
    },
    Stq1_0 {
        shape: Vec<usize>,
        data: &'a [crate::quant::BlockStq1_0],
    },
    /// Standard ggml-quantized tensor handled by Candle's dequantizer.
    Quantized {
        dtype: vendored::GgmlDTypeExt,
        shape: Vec<usize>,
        bytes: &'a [u8],
    },
}

impl std::fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 { shape, data } => write!(f, "F32 {shape:?} len={}", data.len()),
            Self::F16 { shape, data } => write!(f, "F16 {shape:?} len={}", data.len()),
            Self::Stq1_0 { shape, data } => write!(f, "STQ1_0 {shape:?} blocks={}", data.len()),
            Self::Quantized {
                dtype,
                shape,
                bytes,
            } => {
                write!(f, "{dtype:?} {shape:?} bytes={}", bytes.len())
            }
        }
    }
}

impl HyGgufFile {
    /// Memory-map and parse a GGUF file's header.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("failed to open {}: {e}", path.display()),
            ))
        })?;
        // SAFETY: the file is treated as read-only; the mmap is held alive by
        // `Arc` for the lifetime of all returned views.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let content = Content::read(&mut cursor)?;
        let config = HunyuanConfig::from_gguf(&meta::Meta::new(&content))?;
        Ok(Self {
            mmap,
            content,
            config,
        })
    }

    pub fn content(&self) -> &Content {
        &self.content
    }

    pub fn meta(&self) -> meta::Meta<'_> {
        meta::Meta::new(&self.content)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.content.tensor_infos.keys().map(String::as_str)
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.content
            .tensor_infos
            .get(name)
            .ok_or_else(|| Error::Gguf(format!("missing tensor {name}")))
    }

    /// Return a typed view over a named tensor's data borrowed from the mmap.
    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self.tensor_info(name)?;
        let n_bytes = info.size_in_bytes()?;
        // All offset arithmetic is checked: a malicious GGUF cannot wrap
        // either summand and trick `mmap.get` into accepting an
        // out-of-range slice.
        let abs_offset = (self.content.tensor_data_offset as usize)
            .checked_add(info.offset as usize)
            .ok_or_else(|| {
                Error::Gguf(format!(
                    "tensor `{name}` offset overflows usize ({}+{})",
                    self.content.tensor_data_offset, info.offset
                ))
            })?;
        let end = abs_offset.checked_add(n_bytes).ok_or_else(|| {
            Error::Gguf(format!(
                "tensor `{name}` end offset overflows usize ({abs_offset}+{n_bytes})"
            ))
        })?;
        let bytes = self.mmap.get(abs_offset..end).ok_or_else(|| {
            Error::Gguf(format!(
                "tensor `{name}` extends past mmap end ({abs_offset}+{n_bytes} > {})",
                self.mmap.len()
            ))
        })?;

        match info.ggml_dtype {
            GgmlDTypeExt::F32 => Ok(TensorView::F32 {
                shape: info.shape.clone(),
                // F32 is naturally aligned on disk in GGUF (data offset is a
                // multiple of `general.alignment`, default 32).
                data: bytemuck::try_cast_slice(bytes)
                    .map_err(|e| Error::Gguf(format!("F32 cast for `{name}` failed: {e}")))?,
            }),
            GgmlDTypeExt::F16 => Ok(TensorView::F16 {
                shape: info.shape.clone(),
                data: bytemuck::try_cast_slice(bytes)
                    .map_err(|e| Error::Gguf(format!("F16 cast for `{name}` failed: {e}")))?,
            }),
            GgmlDTypeExt::Stq1_0 => {
                let blocks = crate::quant::BlockStq1_0::from_bytes(bytes)
                    .map_err(|e| Error::Stq1_0(format!("STQ1_0 view for `{name}`: {e}")))?;
                Ok(TensorView::Stq1_0 {
                    shape: info.shape.clone(),
                    data: blocks,
                })
            }
            // Pass standard ggml k-quants and legacy quantizations through
            // to Candle's built-in dequantizer.
            GgmlDTypeExt::Q4_0
            | GgmlDTypeExt::Q4_1
            | GgmlDTypeExt::Q5_0
            | GgmlDTypeExt::Q5_1
            | GgmlDTypeExt::Q8_0
            | GgmlDTypeExt::Q8_1
            | GgmlDTypeExt::Q2K
            | GgmlDTypeExt::Q3K
            | GgmlDTypeExt::Q4K
            | GgmlDTypeExt::Q5K
            | GgmlDTypeExt::Q6K
            | GgmlDTypeExt::Q8K => Ok(TensorView::Quantized {
                dtype: info.ggml_dtype,
                shape: info.shape.clone(),
                bytes,
            }),
        }
    }

    /// Clone of the underlying `Arc<Mmap>` — useful when a caller wants to
    /// keep a view alive beyond the lifetime of the [`HyGgufFile`] handle.
    pub fn mmap(&self) -> Arc<Mmap> {
        self.mmap.clone()
    }
}

impl ModelSource for HyGgufFile {
    fn format(&self) -> &'static str {
        "gguf-v3"
    }

    fn config(&self) -> &HunyuanConfig {
        &self.config
    }

    fn load_role(&self, role: TensorRole, dev: &DeviceCtx) -> Result<WeightStore> {
        let name = gguf_name(role);
        // Fast path: STQ1_0 + CPU = zero-copy `Arc<Mmap>` view straight
        // into the file's tensor data; no `to_vec` of ~280 MB on every load.
        let info = self.tensor_info(&name)?;
        if info.ggml_dtype == GgmlDTypeExt::Stq1_0 && dev.supports_stq1_0_native() {
            let n_bytes = info.size_in_bytes()?;
            let abs_offset = (self.content.tensor_data_offset as usize)
                .checked_add(info.offset as usize)
                .ok_or_else(|| Error::Gguf(format!("tensor `{name}` offset overflows usize")))?;
            // Same range checks `tensor_view` would have run.
            let end = abs_offset.checked_add(n_bytes).ok_or_else(|| {
                Error::Gguf(format!("tensor `{name}` end offset overflows usize"))
            })?;
            if end > self.mmap.len() {
                return Err(Error::Gguf(format!(
                    "tensor `{name}` extends past mmap end ({end} > {})",
                    self.mmap.len()
                )));
            }
            if n_bytes % crate::quant::BLOCK_BYTES != 0 {
                return Err(Error::Stq1_0(format!(
                    "tensor `{name}`: {n_bytes} bytes not a multiple of STQ1_0 block size {}",
                    crate::quant::BLOCK_BYTES
                )));
            }
            let n_blocks = n_bytes / crate::quant::BLOCK_BYTES;
            let (rows, cols) = match info.shape.as_slice() {
                [r, c] => (*r, *c),
                [c] => (1, *c),
                other => {
                    return Err(Error::BadShape {
                        name: name.clone(),
                        expected: vec![1, 1],
                        actual: other.to_vec(),
                    })
                }
            };
            let blocks =
                crate::weights::BlockSource::from_mmap(self.mmap.clone(), abs_offset, n_blocks)?;
            return Ok(WeightStore::Stq1_0 { blocks, rows, cols });
        }
        let view = self.tensor_view(&name)?;
        WeightStore::from_view(&name, view, dev)
    }

    fn available_roles(&self) -> Vec<TensorRole> {
        self.content
            .tensor_infos
            .keys()
            .filter_map(|n| from_gguf_name(n))
            .collect()
    }

    fn metadata_summary(&self) -> Vec<(String, String)> {
        let mut out = vec![
            ("format".into(), "gguf-v3".into()),
            ("magic".into(), format!("{:?}", self.content.magic)),
            (
                "tensor_count".into(),
                self.content.tensor_infos.len().to_string(),
            ),
        ];
        // Surface the architecture-relevant metadata if present.
        let interesting = [
            "general.architecture",
            "general.name",
            "general.file_type",
            "hunyuan-dense.block_count",
            "hunyuan-dense.embedding_length",
            "hunyuan-dense.feed_forward_length",
            "hunyuan-dense.attention.head_count",
            "hunyuan-dense.attention.head_count_kv",
            "hunyuan-dense.attention.key_length",
            "hunyuan-dense.context_length",
            "hunyuan-dense.rope.freq_base",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ];
        for k in interesting {
            if let Some(v) = self.content.metadata.get(k) {
                out.push((k.to_string(), format!("{v:?}")));
            }
        }
        out
    }
}
