//! Device selection and dispatch policy.
//!
//! STQ1_0 weights live on CPU only; Metal/CUDA paths require eager
//! dequantization to F16 at load time. This module captures that policy in
//! one place so the rest of the crate doesn't sprinkle `cfg(feature = "metal")`
//! checks throughout.

use candle_core::{DType, Device};

use crate::{Error, Result};

/// High-level kind of accelerator the model is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda,
}

/// Bundle of [`DeviceKind`] and the underlying Candle [`Device`].
#[derive(Debug, Clone)]
pub struct DeviceCtx {
    pub kind: DeviceKind,
    pub device: Device,
}

impl DeviceCtx {
    pub fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            device: Device::Cpu,
        }
    }

    /// Initialize a Metal device. Available only when the `metal` feature is
    /// enabled at compile time.
    #[cfg(feature = "metal")]
    pub fn metal(ordinal: usize) -> Result<Self> {
        Ok(Self {
            kind: DeviceKind::Metal,
            device: Device::new_metal(ordinal)?,
        })
    }

    #[cfg(not(feature = "metal"))]
    pub fn metal(_ordinal: usize) -> Result<Self> {
        Err(Error::Gguf(
            "this build was compiled without the `metal` feature".into(),
        ))
    }

    /// Initialize a CUDA device. Available only when the `cuda` feature is
    /// enabled at compile time.
    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Ok(Self {
            kind: DeviceKind::Cuda,
            device: Device::new_cuda(ordinal)?,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda(_ordinal: usize) -> Result<Self> {
        Err(Error::Gguf(
            "this build was compiled without the `cuda` feature".into(),
        ))
    }

    /// Whether the device can run the STQ1_0 custom matmul natively.
    /// Currently true only for CPU.
    #[inline]
    pub fn supports_stq1_0_native(&self) -> bool {
        matches!(self.kind, DeviceKind::Cpu)
    }

    /// Single source of truth for the activation dtype on this device:
    /// F32 on CPU (matches the STQ1_0 custom matmul kernel) and F16 on
    /// Metal/CUDA (matches eagerly-dequantized weights). All loaders and
    /// the model must agree, so they call this method instead of branching.
    #[inline]
    pub fn compute_dtype(&self) -> DType {
        match self.kind {
            DeviceKind::Cpu => DType::F32,
            DeviceKind::Metal | DeviceKind::Cuda => DType::F16,
        }
    }
}
