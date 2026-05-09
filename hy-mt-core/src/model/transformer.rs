//! Full Hunyuan-MT 1.5 dense decoder model.
//!
//! Loads weights from anything implementing [`ModelSource`] (GGUF or
//! safetensors), applies QK-norm and tied input/output embeddings as
//! configured in [`HunyuanConfig`].

use candle_core::Tensor;

use super::attention::Attention;
use super::config::HunyuanConfig;
use super::ffn::SwiGluFfn;
use super::kv_cache::KvCache;
use super::layer::DecoderLayer;
use super::layout::{BlockSlot, TensorRole};
use super::linear::QuantLinear;
use super::rms_norm::{RmsNorm, RmsNormPerHead};
use crate::device::DeviceCtx;
use crate::gguf::HyGgufFile;
use crate::rope::RopeCache;
use crate::source::ModelSource;
use crate::util::cast_if;
use crate::weights::WeightStore;
use crate::Result;

pub struct HunyuanDense {
    pub config: HunyuanConfig,
    pub device: DeviceCtx,
    embed: Tensor,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    output: Option<QuantLinear>,
    caches: Vec<KvCache>,
}

impl HunyuanDense {
    /// Convenience wrapper for the GGUF path. Equivalent to
    /// [`Self::load_from`] with the file as source.
    pub fn load(gguf: &HyGgufFile, dev: &DeviceCtx) -> Result<Self> {
        Self::load_from(gguf, dev)
    }

    /// Backwards-compat alias used by the synthetic `model_smoke` test.
    pub fn load_with_config(
        gguf: &HyGgufFile,
        dev: &DeviceCtx,
        _config: HunyuanConfig,
    ) -> Result<Self> {
        // The GGUF source already carries the parsed config; the override
        // form is no longer needed in practice. Tests still construct GGUFs
        // whose metadata matches the override they pass in, so just delegate.
        Self::load_from(gguf, dev)
    }

    /// Load the model from any [`ModelSource`].
    pub fn load_from<S: ModelSource + ?Sized>(source: &S, dev: &DeviceCtx) -> Result<Self> {
        let config = *source.config();
        tracing::info!(
            format = source.format(),
            n_layers = config.n_layers,
            hidden = config.hidden_size,
            n_heads = config.n_heads,
            n_kv_heads = config.n_kv_heads,
            head_dim = config.head_dim,
            vocab = config.vocab_size,
            rope_theta = config.rope_theta,
            qk_norm = config.use_qk_norm,
            tied = config.tie_word_embeddings,
            "loading Hunyuan-MT 1.5 model"
        );

        // Single source of truth for the activation dtype on this device
        // (see `DeviceCtx::compute_dtype`).
        let compute_dtype = dev.compute_dtype();
        let cast =
            |t: Tensor| -> Result<Tensor> { cast_if(t.to_device(&dev.device)?, compute_dtype) };

        let load_norm_tensor = |role: TensorRole| -> Result<Tensor> {
            let store = source.load_role(role, dev)?;
            let mut t = cast(store.as_tensor()?)?;
            if t.dims().len() == 2 && t.dim(0)? == 1 {
                t = t.squeeze(0)?;
            }
            Ok(t)
        };

        // Token embedding & final norm.
        let embed = cast(
            source
                .load_role(TensorRole::TokenEmbedding, dev)?
                .as_tensor()?,
        )?;
        let final_norm = RmsNorm::new(
            load_norm_tensor(TensorRole::OutputNorm)?,
            config.rms_norm_eps,
        );

        // Optional output projection (only for non-tied embeddings).
        let output = if config.tie_word_embeddings {
            None
        } else {
            Some(QuantLinear::new(
                source.load_role(TensorRole::Output, dev)?,
            )?)
        };

        let rope = RopeCache::new(config.head_dim, config.rope_theta, &dev.device)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for idx in 0..config.n_layers {
            let load = |slot: BlockSlot| -> Result<WeightStore> {
                source.load_role(TensorRole::Block { idx, slot }, dev)
            };

            let attn_norm = RmsNorm::new(
                load_norm_tensor(TensorRole::Block {
                    idx,
                    slot: BlockSlot::AttnNorm,
                })?,
                config.rms_norm_eps,
            );
            let ffn_norm = RmsNorm::new(
                load_norm_tensor(TensorRole::Block {
                    idx,
                    slot: BlockSlot::FfnNorm,
                })?,
                config.rms_norm_eps,
            );

            let q_proj = QuantLinear::new(load(BlockSlot::AttnQ)?)?;
            let k_proj = QuantLinear::new(load(BlockSlot::AttnK)?)?;
            let v_proj = QuantLinear::new(load(BlockSlot::AttnV)?)?;
            let o_proj = QuantLinear::new(load(BlockSlot::AttnOutput)?)?;

            let (q_norm, k_norm) = if config.use_qk_norm {
                let qn = load_norm_tensor(TensorRole::Block {
                    idx,
                    slot: BlockSlot::AttnQNorm,
                })?;
                let kn = load_norm_tensor(TensorRole::Block {
                    idx,
                    slot: BlockSlot::AttnKNorm,
                })?;
                (
                    Some(RmsNormPerHead::new(qn, config.rms_norm_eps)),
                    Some(RmsNormPerHead::new(kn, config.rms_norm_eps)),
                )
            } else {
                (None, None)
            };

            let attn = Attention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                rope.clone(),
            );

            let gate = QuantLinear::new(load(BlockSlot::FfnGate)?)?;
            let up = QuantLinear::new(load(BlockSlot::FfnUp)?)?;
            let down = QuantLinear::new(load(BlockSlot::FfnDown)?)?;
            let ffn = SwiGluFfn::new(gate, up, down);

            layers.push(DecoderLayer {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        let caches = (0..config.n_layers).map(|_| KvCache::new()).collect();

        Ok(Self {
            config,
            device: dev.clone(),
            embed,
            layers,
            final_norm,
            output,
            caches,
        })
    }

    /// Reset all KV caches and configure the per-layer capacity (in time
    /// steps) for the next prompt. The capacity bounds prompt + decoded
    /// length; oversizing wastes memory, undersizing makes `append` fail.
    pub fn reset_kv_cache(&mut self, capacity: usize) -> Result<()> {
        for c in self.caches.iter_mut() {
            c.reset();
            c.set_capacity(capacity)?;
        }
        Ok(())
    }

    pub fn current_pos(&self) -> usize {
        self.caches.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Run a forward pass on a `[B, T]` token tensor and return logits for
    /// the *last* position only — the typical decode-step output.
    pub fn forward(&mut self, tokens: &Tensor) -> Result<Tensor> {
        let (b, t) = tokens.dims2()?;
        let pos_offset = self.current_pos();

        // Embedding lookup. `embed` has shape [vocab, hidden].
        let tokens_flat = tokens.reshape(b * t)?;
        let h = self.embed.index_select(&tokens_flat, 0)?;
        let h = h.reshape((b, t, self.config.hidden_size))?;

        let mut h = h;
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            h = layer.forward(&h, pos_offset, cache)?;
        }
        let h = self.final_norm.forward(&h)?;

        // Take only the last position to get [B, hidden].
        let last = h.narrow(1, t - 1, 1)?.squeeze(1)?;
        let logits = match &self.output {
            Some(proj) => proj.forward(&last)?,
            None => {
                // Tied embeddings: logits = last @ embed^T.
                let embed_t = self.embed.t()?;
                last.broadcast_matmul(&embed_t)?
            }
        };
        Ok(logits)
    }
}
