//! Safetensors implementation of [`crate::source::ModelSource`].
//!
//! Probing real-world repositories
//! (`AngelSlim/Hy-MT1.5-1.8B-1.25bit`, `tencent/HY-MT1.5-1.8B`) showed that
//! both ship the model in **standard BF16** with the usual HuggingFace
//! tensor naming. There is no custom STQ1_0-packed layout in safetensors —
//! the 1.25-bit format is GGUF-only. So this loader is a thin wrapper
//! around `candle_core::safetensors::MmapedSafetensors` plus the
//! `model/layout.rs` HF-name mapping.

use std::path::{Path, PathBuf};

use candle_core::safetensors::MmapedSafetensors;

use crate::device::DeviceCtx;
use crate::error::LoadingKind;
use crate::model::config::HunyuanConfig;
use crate::model::layout::{from_hf_name, hf_name, TensorRole};
use crate::source::ModelSource;
use crate::util::cast_if;
use crate::weights::WeightStore;
use crate::{Error, Result};

pub struct HySafetensors {
    config: HunyuanConfig,
    mmap: MmapedSafetensors,
}

impl HySafetensors {
    /// Open a single- or multi-shard safetensors model.
    ///
    /// `config_path` points to the accompanying `config.json`. `shards` is
    /// the list of `.safetensors` files (a single entry for unsharded
    /// models, multiple entries for HF-style `model-NNNNN-of-NNNNN.safetensors`).
    pub fn load(config_path: &Path, shards: &[PathBuf]) -> Result<Self> {
        if shards.is_empty() {
            return Err(Error::Validation("no safetensors shards provided".into()));
        }
        let raw = std::fs::read_to_string(config_path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("reading {}: {e}", config_path.display()),
            ))
        })?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::Validation(format!("parsing {}: {e}", config_path.display())))?;
        let config = HunyuanConfig::from_hf_config(&json)?;

        // Candle's `MmapedSafetensors` works with either a single file or
        // a slice of paths.
        let mmap = unsafe {
            if shards.len() == 1 {
                MmapedSafetensors::new(&shards[0])?
            } else {
                MmapedSafetensors::multi(shards)?
            }
        };

        Ok(Self { config, mmap })
    }
}

impl ModelSource for HySafetensors {
    fn format(&self) -> &'static str {
        "safetensors-bf16"
    }

    fn config(&self) -> &HunyuanConfig {
        &self.config
    }

    fn load_role(&self, role: TensorRole, dev: &DeviceCtx) -> Result<WeightStore> {
        let name = hf_name(role);
        let tensor = self
            .mmap
            .load(&name, &dev.device)
            .map_err(|source| Error::Loading {
                kind: LoadingKind::Tensor,
                name: name.clone(),
                source,
            })?;

        // Candle doesn't implement BF16 matmul on CPU, so eagerly cast to
        // the device's compute dtype (F32 on CPU, F16 on Metal/CUDA).
        Ok(WeightStore::Tensor(cast_if(tensor, dev.compute_dtype())?))
    }

    fn available_roles(&self) -> Vec<TensorRole> {
        self.mmap
            .tensors()
            .into_iter()
            .filter_map(|(n, _)| from_hf_name(&n))
            .collect()
    }

    fn metadata_summary(&self) -> Vec<(String, String)> {
        let cfg = &self.config;
        vec![
            ("format".into(), self.format().into()),
            ("n_layers".into(), cfg.n_layers.to_string()),
            ("hidden_size".into(), cfg.hidden_size.to_string()),
            (
                "intermediate_size".into(),
                cfg.intermediate_size.to_string(),
            ),
            ("n_heads".into(), cfg.n_heads.to_string()),
            ("n_kv_heads".into(), cfg.n_kv_heads.to_string()),
            ("head_dim".into(), cfg.head_dim.to_string()),
            ("vocab_size".into(), cfg.vocab_size.to_string()),
            (
                "max_position_embeddings".into(),
                cfg.max_position_embeddings.to_string(),
            ),
            ("rope_theta".into(), cfg.rope_theta.to_string()),
            ("use_qk_norm".into(), cfg.use_qk_norm.to_string()),
            (
                "tie_word_embeddings".into(),
                cfg.tie_word_embeddings.to_string(),
            ),
        ]
    }
}
