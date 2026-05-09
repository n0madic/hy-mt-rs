//! GGUF v1/v2/v3 reader.
//!
//! Vendored and adapted from
//! `candle-core::quantized::gguf_file` (Apache-2.0, Hugging Face).
//! The upstream reader uses the closed-set `GgmlDType` enum and would reject
//! the new STQ1_0 type (id 42) introduced by `ggml-org/llama.cpp` PR #22836,
//! so this copy substitutes the local [`GgmlDTypeExt`] enum and exposes
//! [`TensorInfo::read_raw`] to hand the on-disk byte slice off to our own
//! decoders (`crate::quant`).
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::{Error, Result};

/// All tensor data starts on a multiple of this many bytes (overridable via
/// the `general.alignment` metadata key).
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Hard caps protecting the parser against pathological / malicious GGUF
/// length fields. Values are deliberately generous for legitimate models
/// but small enough that allocations cannot DoS the host.
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB
const MAX_ARRAY_LEN: u64 = 16 * 1024 * 1024; // 16M elements
const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_METADATA_KV_COUNT: u64 = 100_000;
const MAX_TENSOR_DIMS: u64 = 8;
/// Maximum nesting depth for `Array(Array(...))` metadata values. Without
/// this, a malicious GGUF can stack-overflow the parser by chaining
/// recursively-typed arrays before any allocation cap fires.
const MAX_ARRAY_DEPTH: usize = 8;
/// Cap pre-allocation when reading an array of length `len`. The full
/// `len` is still bounded by `MAX_ARRAY_LEN`, but allocating the entire
/// declared capacity up front would let a 16M-element claim consume
/// hundreds of MiB before any data is read.
const ARRAY_PREALLOC_CAP: usize = 1024;
/// Cap on the byte-size of any single tensor we'll read into a buffer
/// via [`TensorInfo::read_raw`]. Real Hunyuan-MT 1.5 weights top out
/// around 200 MiB per tensor; 4 GiB is a comfortable but firm ceiling.
const MAX_TENSOR_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Number of weights in a single STQ1_0 super-block (must match
/// [`crate::quant::QK_K`]).
pub const STQ1_0_BLOCK: usize = 256;
/// On-disk size of a single STQ1_0 super-block, must match
/// `size_of::<crate::quant::BlockStq1_0>()`.
pub const STQ1_0_TYPE_SIZE: usize = 42;

/// Subset of GGML quantization/storage dtypes we need to recognise when
/// loading the Hy-MT 1.5 model. Extends the upstream Candle enum with
/// `Stq1_0 = 42`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDTypeExt {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    /// Sherry Ternary Quant 1.25 bpw from llama.cpp PR #22836.
    /// Production GGUFs from AngelSlim use type id 40 (early-PR drafts and
    /// some research notes mention 42; we accept either at parse time).
    Stq1_0 = 40,
}

impl GgmlDTypeExt {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            40 | 42 => Self::Stq1_0,
            v => return Err(Error::UnsupportedDtype(format!("ggml dtype id {v}"))),
        })
    }

    /// Number of weights packed into one block of this dtype.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::Stq1_0 => STQ1_0_BLOCK,
        }
    }

    /// Number of bytes occupied by one block of this dtype.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            // K-quants and other formats listed for completeness; the loader
            // will refuse to use them, so the precise sizes aren't critical
            // beyond surfacing a clean error if the GGUF claims them.
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Stq1_0 => STQ1_0_TYPE_SIZE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0x46554747 | 0x47475546 => Ok(Self::Gguf),
            _ => Err(Error::Gguf(format!("unknown magic 0x{value:08x}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        Ok(match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => {
                return Err(Error::Gguf(format!(
                    "unsupported magic/version {magic:?}/{version}"
                )))
            }
        })
    }
}

/// Description of one tensor as it appears in the GGUF header.
///
/// Shape is stored in *row-major*, "natural" order (i.e. matching how
/// PyTorch / HuggingFace describe it); GGUF itself stores dimensions in the
/// reverse order on disk and we un-reverse them here.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDTypeExt,
    pub shape: Vec<usize>,
    /// Offset relative to [`Content::tensor_data_offset`].
    pub offset: u64,
}

impl TensorInfo {
    /// Number of weights described by this tensor. Uses `checked_mul` so a
    /// crafted shape like `[u32::MAX, u32::MAX]` is rejected with a clear
    /// error rather than silently wrapping in release builds.
    pub fn elem_count(&self) -> Result<usize> {
        let mut acc: usize = 1;
        for &d in &self.shape {
            acc = acc.checked_mul(d).ok_or_else(|| {
                Error::Gguf(format!("shape {:?} overflows usize at dim {d}", self.shape))
            })?;
        }
        Ok(acc)
    }

    /// Number of bytes the tensor occupies on disk. All multiplications are
    /// checked against overflow for safety against malicious GGUF inputs.
    pub fn size_in_bytes(&self) -> Result<usize> {
        let elems = self.elem_count()?;
        let block_size = self.ggml_dtype.block_size();
        if elems % block_size != 0 {
            return Err(Error::Gguf(format!(
                "element count {elems} is not divisible by block size {block_size}"
            )));
        }
        let blocks = elems / block_size;
        blocks
            .checked_mul(self.ggml_dtype.type_size())
            .ok_or_else(|| {
                Error::Gguf(format!(
                    "tensor byte size overflows usize: blocks={blocks} type_size={}",
                    self.ggml_dtype.type_size()
                ))
            })
    }

    /// Read the raw on-disk bytes for this tensor.
    pub fn read_raw<R: Read + Seek>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Vec<u8>> {
        let size = self.size_in_bytes()?;
        // Reject absurd per-tensor sizes before allocating — a malicious
        // GGUF could otherwise claim 100 GiB and OOM the host.
        if size as u64 > MAX_TENSOR_BYTES {
            return Err(Error::OverLimit {
                what: "GGUF tensor size",
                got: size as u64,
                max: MAX_TENSOR_BYTES,
            });
        }
        let mut buf = vec![0u8; size];
        reader.seek(SeekFrom::Start(tensor_data_offset + self.offset))?;
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Parsed GGUF header — metadata plus tensor descriptors and the absolute
/// offset where tensor data starts.
#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: Read>(reader: &mut R, magic: &VersionedMagic) -> Result<String> {
    let len_u64 = match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as u64,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64::<LittleEndian>()?,
    };
    if len_u64 > MAX_STRING_BYTES {
        return Err(Error::OverLimit {
            what: "GGUF string length",
            got: len_u64,
            max: MAX_STRING_BYTES,
        });
    }
    let len = len_u64 as usize;
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v)?;
    while let Some(0) = v.last() {
        v.pop();
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    /// Upcasts unsigned integer types to `u64` (mirrors the upstream helper).
    pub fn to_u64(&self) -> Result<u64> {
        Ok(match self {
            Self::U64(v) => *v,
            Self::U32(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::U8(v) => *v as u64,
            Self::Bool(v) => *v as u64,
            v => {
                return Err(Error::Gguf(format!(
                    "value {v:?} is not a u64 (or upcastable to one)"
                )))
            }
        })
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            Self::F64(v) => Ok(*v as f32),
            v => Err(Error::Gguf(format!("value {v:?} is not a f32"))),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => Err(Error::Gguf(format!("value {v:?} is not a bool"))),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => Err(Error::Gguf(format!("value {v:?} is not a string"))),
        }
    }

    pub fn to_array(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => Err(Error::Gguf(format!("value {v:?} is not an array"))),
        }
    }

    fn read<R: Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self> {
        Self::read_at_depth(reader, value_type, magic, 0)
    }

    fn read_at_depth<R: Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
        depth: usize,
    ) -> Result<Self> {
        Ok(match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => return Err(Error::Gguf(format!("unexpected bool value {b}"))),
            },
            ValueType::String => Self::String(read_string(reader, magic)?),
            ValueType::Array => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(Error::OverLimit {
                        what: "GGUF array nesting depth",
                        got: (depth + 1) as u64,
                        max: MAX_ARRAY_DEPTH as u64,
                    });
                }
                let inner = ValueType::from_u32(reader.read_u32::<LittleEndian>()?)?;
                let len_u64 = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as u64,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>()?
                    }
                };
                if len_u64 > MAX_ARRAY_LEN {
                    return Err(Error::OverLimit {
                        what: "GGUF array length",
                        got: len_u64,
                        max: MAX_ARRAY_LEN,
                    });
                }
                let len = len_u64 as usize;
                // Pre-allocate at most ARRAY_PREALLOC_CAP entries; the real
                // length is still validated against MAX_ARRAY_LEN above.
                let mut vs = Vec::with_capacity(len.min(ARRAY_PREALLOC_CAP));
                for _ in 0..len {
                    vs.push(Value::read_at_depth(reader, inner, magic, depth + 1)?);
                }
                Self::Array(vs)
            }
        })
    }
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => return Err(Error::Gguf(format!("unrecognised value-type {v}"))),
        })
    }
}

impl Content {
    /// Parse a GGUF header from `reader`. Tensor data is *not* read.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader)?;
        let tensor_count_u64 = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as u64,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64::<LittleEndian>()?,
        };
        if tensor_count_u64 > MAX_TENSOR_COUNT {
            return Err(Error::OverLimit {
                what: "GGUF tensor_count",
                got: tensor_count_u64,
                max: MAX_TENSOR_COUNT,
            });
        }
        let tensor_count = tensor_count_u64 as usize;
        let metadata_kv_count_u64 = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as u64,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64::<LittleEndian>()?,
        };
        if metadata_kv_count_u64 > MAX_METADATA_KV_COUNT {
            return Err(Error::OverLimit {
                what: "GGUF metadata_kv_count",
                got: metadata_kv_count_u64,
                max: MAX_METADATA_KV_COUNT,
            });
        }
        let metadata_kv_count = metadata_kv_count_u64 as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(reader, &magic)?;
            let value_type = ValueType::from_u32(reader.read_u32::<LittleEndian>()?)?;
            let value = Value::read(reader, value_type, &magic)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = HashMap::new();
        for _ in 0..tensor_count {
            let tensor_name = read_string(reader, &magic)?;
            let n_dimensions_u32 = reader.read_u32::<LittleEndian>()?;
            if n_dimensions_u32 as u64 > MAX_TENSOR_DIMS {
                return Err(Error::OverLimit {
                    what: "GGUF tensor n_dimensions",
                    got: n_dimensions_u32 as u64,
                    max: MAX_TENSOR_DIMS,
                });
            }
            let n_dimensions = n_dimensions_u32 as usize;
            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut buf = vec![0u32; n_dimensions];
                    reader.read_u32_into::<LittleEndian>(&mut buf)?;
                    buf.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut buf = vec![0u64; n_dimensions];
                    reader.read_u64_into::<LittleEndian>(&mut buf)?;
                    buf.into_iter().map(|c| c as usize).collect()
                }
            };
            // GGUF stores dimensions in reverse order (last-axis first).
            dimensions.reverse();

            let ggml_dtype = GgmlDTypeExt::from_u32(reader.read_u32::<LittleEndian>()?)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: dimensions,
                    offset,
                    ggml_dtype,
                },
            );
        }

        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;

        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    /// Read the raw bytes for a named tensor. The dispatch on dtype lives in
    /// the higher-level loader (`crate::gguf::HyGgufFile`).
    pub fn tensor_raw<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<(GgmlDTypeExt, Vec<usize>, Vec<u8>)> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| Error::Gguf(format!("missing tensor {name}")))?;
        let bytes = info.read_raw(reader, self.tensor_data_offset)?;
        Ok((info.ggml_dtype, info.shape.clone(), bytes))
    }
}
