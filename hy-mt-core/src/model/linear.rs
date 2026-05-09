//! Linear (no-bias) projection that dispatches between the custom STQ1_0
//! matmul on CPU and Candle's standard F16/F32 matmul on the active device.
//!
//! The Hunyuan-MT 1.5 architecture uses no biases (`mlp_bias = false`,
//! `attention_bias = false`), so this layer is intentionally bias-free. The
//! input is laid out as `[..., in_features]`; the output replaces the last
//! dimension with `out_features`.

use std::sync::atomic::{AtomicU8, Ordering};

use candle_core::{DType, Device, Storage, Tensor};

use crate::quant::{quantize_row_q8, stq_matmul_f32, stq_matmul_q8, BlockQ8, QK_K};
use crate::util::cast_if;
use crate::weights::WeightStore;
use crate::{Error, Result};

/// Run-time toggle for the int8-activation matmul path. Reads
/// `HY_MT_USE_Q8` once on first call (`0`/empty/`false` → off, anything
/// else → on); cached afterwards as a single relaxed atomic load. The
/// f32 path stays the default until benchmarks show the Q8 path wins
/// across the projection shapes that matter.
fn use_q8_path() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("HY_MT_USE_Q8")
                .ok()
                .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "FALSE"))
                .unwrap_or(false);
            CACHE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

#[derive(Clone)]
pub struct QuantLinear {
    weight: WeightStore,
    out_features: usize,
    in_features: usize,
}

impl QuantLinear {
    pub fn new(weight: WeightStore) -> Result<Self> {
        let (out_features, in_features) = weight.shape()?;
        Ok(Self {
            weight,
            out_features,
            in_features,
        })
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn weight(&self) -> &WeightStore {
        &self.weight
    }

    /// `y = x @ W^T`, where `W` has shape `[out_features, in_features]`.
    ///
    /// Output preserves the dtype of `x` so it can be added to a residual
    /// without an explicit cast.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        match &self.weight {
            WeightStore::Tensor(w) => {
                let x_cast = cast_if(x.clone(), w.dtype())?;
                let y = x_cast.broadcast_matmul(&w.t()?)?;
                cast_if(y, in_dtype)
            }
            WeightStore::Stq1_0 { blocks, rows, cols } => {
                debug_assert_eq!(*rows, self.out_features);
                debug_assert_eq!(*cols, self.in_features);

                let in_device = x.device().clone();
                // Pull onto CPU + F32 (the kernel's native dtypes). Both
                // calls are no-ops on the typical Hy-MT CPU compute path.
                let x_cpu = cast_if(x.to_device(&Device::Cpu)?, DType::F32)?;

                let in_dims = x_cpu.dims().to_vec();
                let last = match in_dims.last() {
                    Some(&v) => v,
                    None => {
                        return Err(Error::BadShape {
                            name: "QuantLinear input".into(),
                            expected: vec![*cols],
                            actual: vec![],
                        })
                    }
                };
                if last != *cols {
                    return Err(Error::BadShape {
                        name: "QuantLinear input".into(),
                        expected: vec![*cols],
                        actual: vec![last],
                    });
                }
                let m: usize = in_dims[..in_dims.len() - 1]
                    .iter()
                    .product::<usize>()
                    .max(1);

                // Zero-copy view into the input tensor's CPU storage. If
                // the layout is non-contiguous (e.g. fresh from a
                // transpose) we fall back to one contiguous() copy — the
                // typical Hy-MT activation flow keeps it contiguous, so
                // the fast path is hit on every projection.
                let x_flat = x_cpu.reshape((m * cols,))?;
                let x_owned;
                let x_view: &Tensor = if x_flat.is_contiguous() {
                    &x_flat
                } else {
                    x_owned = x_flat.contiguous()?;
                    &x_owned
                };
                let (storage, layout) = x_view.storage_and_layout();
                let x_data: &[f32] = match &*storage {
                    Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
                    _ => {
                        return Err(Error::Validation(
                            "QuantLinear: expected CPU storage after to_device(Cpu)".into(),
                        ))
                    }
                };
                let start = layout.start_offset();
                let x_slice = &x_data[start..start + m * cols];

                // Output buffer. `Tensor::from_vec` consumes ownership, so
                // we cannot fold this into a thread-local scratch without
                // also bypassing `from_vec`; the alloc is ~tens of µs and
                // dwarfed by the matmul itself, so we keep the simple shape.
                let mut y_buf = vec![0.0f32; m * rows];
                if use_q8_path() {
                    // Pack the activation into per-block int8 lanes once
                    // and feed it to the integer kernel. Memory traffic
                    // shrinks 4× and the inner loop becomes signed dot
                    // products that map onto NEON / DotProd directly.
                    debug_assert_eq!(cols % QK_K, 0);
                    let bpr = cols / QK_K;
                    let mut q8 = vec![BlockQ8::zeroed(); m * bpr];
                    quantize_row_q8(x_slice, &mut q8)?;
                    stq_matmul_q8(blocks.as_slice(), &q8, &mut y_buf, m, *rows, *cols)?;
                } else {
                    stq_matmul_f32(blocks.as_slice(), x_slice, &mut y_buf, m, *rows, *cols)?;
                }
                drop(storage);

                let mut out_dims = Vec::with_capacity(in_dims.len());
                out_dims.extend_from_slice(&in_dims[..in_dims.len() - 1]);
                out_dims.push(*rows);
                let y = Tensor::from_vec(y_buf, out_dims, &Device::Cpu)?;
                let y = y.to_device(&in_device)?;
                cast_if(y, in_dtype)
            }
        }
    }
}
