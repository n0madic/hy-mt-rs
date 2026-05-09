//! Hyperparameters for the Hunyuan-MT 1.5 dense decoder.
//!
//! When loading from GGUF we read the `hunyuan-dense.*` keys; the [`Self::hy_mt_15_18b`]
//! constructor produces the same values hardcoded for the 1.8B base model
//! (useful for tests and as a sanity reference).

use crate::gguf::meta::Meta;
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct HunyuanConfig {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Dynamic-NTK alpha (only kicks in beyond `max_position_embeddings`).
    pub rope_scaling_alpha: f32,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
}

impl HunyuanConfig {
    /// Reference values for the 1.8B base model — useful for tests where no
    /// GGUF file is available.
    pub fn hy_mt_15_18b() -> Self {
        Self {
            n_layers: 32,
            hidden_size: 2048,
            intermediate_size: 6144,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 128,
            vocab_size: 120818,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling_alpha: 1000.0,
            use_qk_norm: true,
            tie_word_embeddings: true,
            bos_id: 120_000,
            eos_id: 120_020,
            pad_id: 120_002,
        }
    }

    /// Read the configuration from GGUF metadata using the standard
    /// `hunyuan-dense.*` key prefix.
    pub fn from_gguf(meta: &Meta<'_>) -> Result<Self> {
        let arch = meta.architecture()?;
        if arch != "hunyuan-dense" {
            return Err(crate::Error::Gguf(format!(
                "expected architecture `hunyuan-dense`, got `{arch}`"
            )));
        }

        let n_layers = meta.usize(&Meta::arch_key("block_count"))?;
        let hidden_size = meta.usize(&Meta::arch_key("embedding_length"))?;
        let intermediate_size = meta.usize(&Meta::arch_key("feed_forward_length"))?;
        let n_heads = meta.usize(&Meta::arch_key("attention.head_count"))?;
        let n_kv_heads = meta.usize(&Meta::arch_key("attention.head_count_kv"))?;
        let max_position_embeddings = meta.usize(&Meta::arch_key("context_length"))?;
        let rms_norm_eps = meta
            .opt_f32(&Meta::arch_key("attention.layer_norm_rms_epsilon"))?
            .unwrap_or(1e-5);
        let rope_theta = meta
            .opt_f32(&Meta::arch_key("rope.freq_base"))?
            .unwrap_or(10_000.0);
        let rope_scaling_alpha = meta
            .opt_f32(&Meta::arch_key("rope.scaling.alpha"))?
            .unwrap_or(1.0);

        // head_dim may be present explicitly or inferred from hidden_size / n_heads.
        let head_dim = match meta.opt_u32(&Meta::arch_key("attention.key_length"))? {
            Some(v) => v as usize,
            None => hidden_size / n_heads,
        };
        let vocab_size = match meta.usize(&Meta::arch_key("vocab_size")) {
            Ok(v) => v,
            Err(_) => meta
                .opt_array_len("tokenizer.ggml.tokens")?
                .ok_or_else(|| crate::Error::Gguf("cannot determine vocab size".into()))?,
        };

        let use_qk_norm = meta
            .opt_bool(&Meta::arch_key("attention.use_qk_norm"))?
            .unwrap_or(true);
        let tie_word_embeddings = meta
            .opt_bool(&Meta::arch_key("tie_word_embeddings"))?
            .unwrap_or(true);

        let bos_id = meta
            .opt_u32("tokenizer.ggml.bos_token_id")?
            .unwrap_or(120_000);
        let eos_id = meta
            .opt_u32("tokenizer.ggml.eos_token_id")?
            .unwrap_or(120_020);
        let pad_id = meta
            .opt_u32("tokenizer.ggml.padding_token_id")?
            .unwrap_or(120_002);

        Ok(Self {
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            rope_scaling_alpha,
            use_qk_norm,
            tie_word_embeddings,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    /// `n_heads / n_kv_heads`, the GQA repetition factor.
    pub fn kv_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Parse a HuggingFace `config.json` (e.g. as shipped with
    /// `tencent/HY-MT1.5-1.8B` or `AngelSlim/Hy-MT1.5-1.8B-1.25bit`).
    ///
    /// Missing fields fall back to architecture defaults from
    /// [`Self::hy_mt_15_18b`] where reasonable.
    pub fn from_hf_config(json: &serde_json::Value) -> Result<Self> {
        let arch = json
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !arch.contains("HunYuan") {
            return Err(crate::Error::Gguf(format!(
                "expected HunYuan* architecture, got `{arch}`"
            )));
        }

        let defaults = Self::hy_mt_15_18b();
        let usize_or = |key: &str, fallback: usize| -> usize {
            json.get(key)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .unwrap_or(fallback)
        };
        let f32_or = |key: &str, fallback: f32| -> f32 {
            json.get(key)
                .and_then(|v| v.as_f64())
                .map(|x| x as f32)
                .unwrap_or(fallback)
        };
        let u32_or = |key: &str, fallback: u32| -> u32 {
            json.get(key)
                .and_then(|v| v.as_u64())
                .map(|x| x as u32)
                .unwrap_or(fallback)
        };
        let bool_or = |key: &str, fallback: bool| -> bool {
            json.get(key).and_then(|v| v.as_bool()).unwrap_or(fallback)
        };

        let n_heads = usize_or("num_attention_heads", defaults.n_heads);
        let hidden_size = usize_or("hidden_size", defaults.hidden_size);
        let head_dim = json
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                json.get("attention_head_dim")
                    .and_then(|v| v.as_u64())
                    .map(|x| x as usize)
                    .unwrap_or(hidden_size / n_heads.max(1))
            });
        // The HF checkpoint stores the *unscaled* RoPE base (10_000) plus a
        // separate `rope_scaling.alpha` factor. Hunyuan was trained with the
        // alpha-NTK base scaling **always** applied (not only beyond
        // `max_position_embeddings`), so the model expects every position —
        // including the first — to use the rescaled base. The production
        // STQ1_0 GGUF stores the post-scaling base (`11_158_840`) directly,
        // which is why GGUF inference matches the HF reference output and
        // raw-base safetensors inference doesn't. Apply the scaling here so
        // both paths feed `RopeCache::new` an already-effective base.
        //
        //   effective_base = base * alpha^(d / (d - 2))
        //
        // Default alpha = 1.0 (i.e. no scaling) when the field is absent.
        let rope_alpha = json
            .get("rope_scaling")
            .and_then(|v| v.get("alpha"))
            .and_then(|v| v.as_f64())
            .map(|x| x as f32)
            .unwrap_or(1.0);
        let raw_rope_theta = f32_or("rope_theta", defaults.rope_theta);
        let effective_rope_theta =
            if rope_alpha > 0.0 && (rope_alpha - 1.0).abs() > f32::EPSILON && head_dim > 2 {
                let exp = head_dim as f32 / (head_dim as f32 - 2.0);
                raw_rope_theta * rope_alpha.powf(exp)
            } else {
                raw_rope_theta
            };

        Ok(Self {
            n_layers: usize_or("num_hidden_layers", defaults.n_layers),
            hidden_size,
            intermediate_size: usize_or("intermediate_size", defaults.intermediate_size),
            n_heads,
            n_kv_heads: usize_or("num_key_value_heads", defaults.n_kv_heads),
            head_dim,
            vocab_size: usize_or("vocab_size", defaults.vocab_size),
            max_position_embeddings: usize_or(
                "max_position_embeddings",
                defaults.max_position_embeddings,
            ),
            rms_norm_eps: f32_or("rms_norm_eps", defaults.rms_norm_eps),
            rope_theta: effective_rope_theta,
            rope_scaling_alpha: rope_alpha,
            use_qk_norm: bool_or("use_qk_norm", defaults.use_qk_norm),
            tie_word_embeddings: bool_or("tie_word_embeddings", defaults.tie_word_embeddings),
            bos_id: u32_or("bos_token_id", defaults.bos_id),
            eos_id: u32_or("eos_token_id", defaults.eos_id),
            pad_id: u32_or("pad_token_id", defaults.pad_id),
        })
    }
}
