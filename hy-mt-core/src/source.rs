//! Format-agnostic interface for everything that can produce Hunyuan-MT 1.5
//! weights — be it a GGUF file, a HuggingFace safetensors model, or a
//! synthetic test fixture.
//!
//! The trait keeps the abstraction at the [`WeightStore`] level, not at the
//! raw-bytes level: each loader takes care of its own dtype/packing details
//! and hands back a ready-to-use `WeightStore`.

use crate::device::DeviceCtx;
use crate::model::config::HunyuanConfig;
use crate::model::layout::TensorRole;
use crate::weights::WeightStore;
use crate::Result;

/// Source of model weights. Implemented by [`crate::gguf::HyGgufFile`] and
/// (after Phase C) by `HySafetensors`.
pub trait ModelSource {
    /// Short identifier for logs and the `inspect` CLI command,
    /// e.g. `"gguf-v3"` or `"safetensors-stq1_0"`.
    fn format(&self) -> &'static str;

    /// Architecture configuration parsed from this source.
    fn config(&self) -> &HunyuanConfig;

    /// Load a single tensor by its semantic [`TensorRole`] into a
    /// [`WeightStore`] living on `dev`.
    fn load_role(&self, role: TensorRole, dev: &DeviceCtx) -> Result<WeightStore>;

    /// All tensor roles this source actually contains. Used to validate the
    /// source against the architecture's expectations.
    fn available_roles(&self) -> Vec<TensorRole>;

    /// `(key, value)` pairs to display in `hy-mt inspect`.
    /// Format-specific; arbitrary content.
    fn metadata_summary(&self) -> Vec<(String, String)>;
}
