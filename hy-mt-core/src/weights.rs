//! Loaded weight storage with two backends (STQ1_0 blocks vs Candle Tensor)
//! and a loader that converts a [`crate::gguf::TensorView`] into the right
//! variant for the chosen device.

use std::sync::Arc;

use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType};
use candle_core::{DType, Tensor};
use half::f16;
use memmap2::Mmap;

use crate::device::DeviceCtx;
use crate::gguf::vendored::GgmlDTypeExt;
use crate::gguf::TensorView;
use crate::quant::{dequantize_row_stq1_0_f32, BlockStq1_0, BLOCK_BYTES};
use crate::util::cast_if;
use crate::{Error, Result};

/// Source for a slice of STQ1_0 blocks. Either a zero-copy view into a
/// memory-mapped GGUF (the production path) or a heap-owned `Arc<[…]>`
/// (synthesised in tests, or copied out of safetensors).
#[derive(Clone)]
pub enum BlockSource {
    /// Borrowed-via-`Arc<Mmap>` view: no allocation, no copy. The mmap
    /// keeps the memory live for as long as any `BlockSource::Mmap` holds
    /// a clone of the `Arc`. Bounds (`offset .. offset + n_blocks * 42`)
    /// are validated up front in [`BlockSource::from_mmap`] so the
    /// hot-path `as_slice` does no checked indexing of its own.
    Mmap {
        mmap: Arc<Mmap>,
        offset: usize,
        n_blocks: usize,
    },
    /// Heap-owned blocks, e.g. from a synthetic test or after dequant
    /// from another quantization format that lacks a packed mmap layout.
    Owned(Arc<[BlockStq1_0]>),
}

impl BlockSource {
    /// Build a zero-copy view into `mmap` covering `n_blocks` consecutive
    /// STQ1_0 blocks starting at byte `offset`.
    ///
    /// Validates the byte range fits inside the mmap so subsequent
    /// `as_slice` calls cannot panic.
    pub fn from_mmap(mmap: Arc<Mmap>, offset: usize, n_blocks: usize) -> Result<Self> {
        let needed = n_blocks
            .checked_mul(BLOCK_BYTES)
            .and_then(|n| offset.checked_add(n))
            .ok_or_else(|| {
                Error::Stq1_0(format!(
                    "BlockSource::from_mmap: offset+len overflow ({offset} + {n_blocks}*{BLOCK_BYTES})"
                ))
            })?;
        if needed > mmap.len() {
            return Err(Error::Stq1_0(format!(
                "BlockSource::from_mmap: range {offset}..{needed} exceeds mmap len {}",
                mmap.len()
            )));
        }
        Ok(Self::Mmap {
            mmap,
            offset,
            n_blocks,
        })
    }

    /// Borrow the blocks as a slice. Hot-path call; no allocation, no
    /// bounds checks (validated at construction).
    #[inline]
    pub fn as_slice(&self) -> &[BlockStq1_0] {
        match self {
            Self::Owned(blocks) => blocks,
            Self::Mmap {
                mmap,
                offset,
                n_blocks,
            } => {
                let bytes = &mmap[*offset..*offset + *n_blocks * BLOCK_BYTES];
                bytemuck::cast_slice(bytes)
            }
        }
    }

    pub fn n_blocks(&self) -> usize {
        match self {
            Self::Owned(blocks) => blocks.len(),
            Self::Mmap { n_blocks, .. } => *n_blocks,
        }
    }
}

impl From<Arc<[BlockStq1_0]>> for BlockSource {
    fn from(blocks: Arc<[BlockStq1_0]>) -> Self {
        Self::Owned(blocks)
    }
}

impl From<Vec<BlockStq1_0>> for BlockSource {
    fn from(blocks: Vec<BlockStq1_0>) -> Self {
        Self::Owned(blocks.into())
    }
}

/// A loaded tensor. Either packed STQ1_0 blocks (CPU only) or a Candle
/// [`Tensor`] in F32/F16 living on the target device.
#[derive(Clone)]
#[non_exhaustive]
pub enum WeightStore {
    /// Packed 1.25-bit weights. The `BlockSource` decides whether the
    /// blocks are an `Arc<Mmap>` view (production GGUF path, zero copy)
    /// or a heap-owned `Arc<[…]>` (tests / non-mmap loaders).
    Stq1_0 {
        blocks: BlockSource,
        rows: usize,
        cols: usize,
    },
    /// Standard tensor weight; `shape` is `[..]`.
    Tensor(Tensor),
}

impl WeightStore {
    /// Convert a tensor view from a GGUF file into a [`WeightStore`] on the
    /// target device. STQ1_0 weights are kept compressed when the device
    /// supports the native matmul (currently CPU); on Metal/CUDA they are
    /// dequantized to F16 and uploaded.
    pub fn from_view(name: &str, view: TensorView<'_>, dev: &DeviceCtx) -> Result<Self> {
        match view {
            TensorView::F32 { shape, data } => {
                let t = Tensor::from_slice(data, shape.as_slice(), &dev.device)?;
                Ok(Self::Tensor(t))
            }
            TensorView::F16 { shape, data } => {
                let t = Tensor::from_slice(data, shape.as_slice(), &dev.device)?;
                Ok(Self::Tensor(t))
            }
            TensorView::Quantized {
                dtype,
                shape,
                bytes,
            } => {
                let candle_dtype = map_dtype_to_candle(dtype).ok_or_else(|| {
                    Error::UnsupportedDtype(format!(
                        "Candle has no built-in decoder for dtype {dtype:?} (tensor `{name}`)"
                    ))
                })?;
                // Candle's `qtensor_from_ggml` always builds a CPU QTensor; we
                // dequantize and move to the target device so the rest of the
                // model code can treat the result as any other Tensor.
                let qt = qtensor_from_ggml(candle_dtype, bytes, shape, &candle_core::Device::Cpu)?;
                let dequantized = qt.dequantize(&candle_core::Device::Cpu)?;
                let on_device = dequantized.to_device(&dev.device)?;
                Ok(Self::Tensor(on_device))
            }
            TensorView::Stq1_0 { shape, data } => {
                let (rows, cols) = match shape.as_slice() {
                    &[r, c] => (r, c),
                    &[c] => (1, c),
                    other => {
                        return Err(Error::BadShape {
                            name: name.to_string(),
                            expected: vec![1, 1],
                            actual: other.to_vec(),
                        })
                    }
                };
                if dev.supports_stq1_0_native() {
                    Ok(Self::Stq1_0 {
                        blocks: BlockSource::Owned(data.to_vec().into()),
                        rows,
                        cols,
                    })
                } else {
                    // Eager dequantization to F16 for Metal/CUDA.
                    let n = rows * cols;
                    let mut buf = vec![f16::ZERO; n];
                    let mut tmp = vec![0.0f32; n];
                    dequantize_row_stq1_0_f32(data, &mut tmp)?;
                    for (s, d) in tmp.iter().zip(buf.iter_mut()) {
                        *d = f16::from_f32(*s);
                    }
                    let t = Tensor::from_vec(buf, (rows, cols), &dev.device)?;
                    Ok(Self::Tensor(t))
                }
            }
        }
    }

    /// Hidden dimension `[rows, cols]` in `[out, in]` order for projection
    /// weights. For 1-D tensors (norms, embeddings as 1-D) the first dim is
    /// 1 and `cols` is the actual length. Tensors of any other rank are
    /// rejected with a [`Error::BadShape`] so a malformed model cannot
    /// silently propagate `(0, 0)` into downstream matmul.
    pub fn shape(&self) -> Result<(usize, usize)> {
        match self {
            Self::Stq1_0 { rows, cols, .. } => Ok((*rows, *cols)),
            Self::Tensor(t) => match t.dims() {
                [r, c] => Ok((*r, *c)),
                [c] => Ok((1, *c)),
                other => Err(Error::BadShape {
                    name: "WeightStore".into(),
                    expected: vec![1, 1],
                    actual: other.to_vec(),
                }),
            },
        }
    }

    /// Borrow as a Candle [`Tensor`] regardless of underlying storage.
    /// For STQ1_0 weights this involves a fresh dequant → CPU F32 tensor.
    pub fn as_tensor(&self) -> Result<Tensor> {
        match self {
            Self::Tensor(t) => Ok(t.clone()),
            Self::Stq1_0 { blocks, rows, cols } => {
                let mut buf = vec![0.0f32; rows * cols];
                dequantize_row_stq1_0_f32(blocks.as_slice(), &mut buf)?;
                Tensor::from_vec(buf, (*rows, *cols), &candle_core::Device::Cpu)
                    .map_err(Error::from)
            }
        }
    }

    /// Cast a [`WeightStore::Tensor`] to a different dtype in-place.
    pub fn to_dtype(self, dtype: DType) -> Result<Self> {
        match self {
            Self::Tensor(t) => Ok(Self::Tensor(cast_if(t, dtype)?)),
            other => Ok(other),
        }
    }
}

/// Map our local [`GgmlDTypeExt`] onto Candle's `GgmlDType` for the standard
/// ggml-quantized formats. Returns `None` for `Stq1_0` (handled separately)
/// or any future variant Candle doesn't support.
fn map_dtype_to_candle(dtype: GgmlDTypeExt) -> Option<GgmlDType> {
    Some(match dtype {
        GgmlDTypeExt::F32 => GgmlDType::F32,
        GgmlDTypeExt::F16 => GgmlDType::F16,
        GgmlDTypeExt::Q4_0 => GgmlDType::Q4_0,
        GgmlDTypeExt::Q4_1 => GgmlDType::Q4_1,
        GgmlDTypeExt::Q5_0 => GgmlDType::Q5_0,
        GgmlDTypeExt::Q5_1 => GgmlDType::Q5_1,
        GgmlDTypeExt::Q8_0 => GgmlDType::Q8_0,
        GgmlDTypeExt::Q8_1 => GgmlDType::Q8_1,
        GgmlDTypeExt::Q2K => GgmlDType::Q2K,
        GgmlDTypeExt::Q3K => GgmlDType::Q3K,
        GgmlDTypeExt::Q4K => GgmlDType::Q4K,
        GgmlDTypeExt::Q5K => GgmlDType::Q5K,
        GgmlDTypeExt::Q6K => GgmlDType::Q6K,
        GgmlDTypeExt::Q8K => GgmlDType::Q8K,
        GgmlDTypeExt::Stq1_0 => return None,
    })
}
