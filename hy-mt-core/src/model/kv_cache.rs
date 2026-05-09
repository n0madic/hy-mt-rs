//! Per-layer key/value cache with a pre-allocated growth buffer.
//!
//! Holds `K`/`V` of shape `[B, H_kv, capacity, head_dim]` and writes new
//! tokens in place via [`Tensor::slice_set`], returning a narrowed view
//! covering the populated prefix `[..., 0..len, ...]`. This replaces the
//! previous `Tensor::cat`-on-every-step approach whose allocation cost
//! grew quadratically with sequence length.

use candle_core::Tensor;

use crate::{Error, Result};

#[derive(Clone, Default)]
pub struct KvCache {
    /// Full backing tensor of shape `[B, H_kv, capacity, head_dim]`.
    /// Allocated lazily on the first `append` call so the cache doesn't
    /// need to know B/H/D at construction time.
    storage_k: Option<Tensor>,
    storage_v: Option<Tensor>,
    /// Number of valid time steps written into `storage_*[..., 0..len, ...]`.
    len: usize,
    /// Maximum number of time steps the cache can hold. `None` means
    /// not yet configured — `append` will return an error in that state.
    /// Production code wires this through [`Self::with_capacity`] or
    /// [`Self::set_capacity`].
    capacity: Option<usize>,
}

impl KvCache {
    /// Build an empty cache; capacity is unset until [`Self::set_capacity`]
    /// or the first call sized through `with_capacity`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a cache pre-configured to hold `capacity` time steps.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: Some(capacity),
            ..Self::default()
        }
    }

    /// Override the capacity. Must be called when the cache is empty — i.e.
    /// either right after construction or after [`Self::reset`]. Calling
    /// while populated would corrupt the storage shape vs. the cap.
    pub fn set_capacity(&mut self, capacity: usize) -> Result<()> {
        if self.storage_k.is_some() || self.len != 0 {
            return Err(Error::Validation(
                "KvCache::set_capacity called on a non-empty cache; \
                 call reset() first"
                    .into(),
            ));
        }
        self.capacity = Some(capacity);
        Ok(())
    }

    /// Drop accumulated state but keep the configured `capacity`. The
    /// backing tensors are released so a subsequent `append` re-allocates
    /// with the new shapes (which may legitimately differ across runs).
    pub fn reset(&mut self) {
        self.storage_k = None;
        self.storage_v = None;
        self.len = 0;
    }

    /// Number of time steps currently populated in the cache.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append `(new_k, new_v)` and return narrowed views covering the
    /// populated prefix. Both tensors must have shape
    /// `[B, H_kv, T_new, head_dim]` and the last dimension is appended
    /// onto along axis 2.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, h, t_new, d) = new_k.dims4()?;
        let (b2, h2, t2, d2) = new_v.dims4()?;
        if (b, h, t_new, d) != (b2, h2, t2, d2) {
            return Err(Error::BadShape {
                name: "KvCache::append (K vs V)".into(),
                expected: vec![b, h, t_new, d],
                actual: vec![b2, h2, t2, d2],
            });
        }
        // If a previous `append` already allocated storage, the new tensor's
        // batch / head / head_dim must match — otherwise `slice_set` would
        // silently mis-write or surface an opaque Candle error several
        // layers deep. Fail loudly here with a clean BadShape.
        if let Some(s) = self.storage_k.as_ref() {
            let (sb, sh, _, sd) = s.dims4()?;
            if (sb, sh, sd) != (b, h, d) {
                return Err(Error::BadShape {
                    name: "KvCache::append (vs existing storage)".into(),
                    expected: vec![sb, sh, t_new, sd],
                    actual: vec![b, h, t_new, d],
                });
            }
        }
        // Capacity must be configured before the first append. The previous
        // 8192 fallback silently allocated hundreds of MiB on real models,
        // hiding the misconfiguration; require explicit setup instead.
        let capacity = self.capacity.ok_or_else(|| {
            Error::Validation(
                "KvCache capacity is not configured — call `with_capacity` or \
                 `set_capacity` before the first append (typically wired \
                 through `HunyuanDense::reset_kv_cache`)"
                    .into(),
            )
        })?;
        let total = self.len.saturating_add(t_new);
        if total > capacity {
            return Err(Error::OverLimit {
                what: "KV cache capacity",
                got: total as u64,
                max: capacity as u64,
            });
        }

        // First call: allocate the full storage so subsequent appends are
        // O(t_new) writes instead of O(len + t_new) re-allocations.
        if self.storage_k.is_none() {
            self.storage_k = Some(Tensor::zeros(
                (b, h, capacity, d),
                new_k.dtype(),
                new_k.device(),
            )?);
            self.storage_v = Some(Tensor::zeros(
                (b, h, capacity, d),
                new_v.dtype(),
                new_v.device(),
            )?);
        }
        let storage_k = self.storage_k.as_ref().expect("just initialised");
        let storage_v = self.storage_v.as_ref().expect("just initialised");

        // `slice_set` requires contiguous sources; defensive `.contiguous()`
        // is cheap on already-contiguous inputs.
        let new_k_c = new_k.contiguous()?;
        let new_v_c = new_v.contiguous()?;
        storage_k.slice_set(&new_k_c, 2, self.len)?;
        storage_v.slice_set(&new_v_c, 2, self.len)?;

        self.len += t_new;
        let k_full = storage_k.narrow(2, 0, self.len)?;
        let v_full = storage_v.narrow(2, 0, self.len)?;
        Ok((k_full, v_full))
    }
}
