//! HuggingFace Hub integration: download model files into the standard
//! `~/.cache/huggingface/hub` cache and detect the on-disk format.
//!
//! Three repositories are first-class citizens:
//! - `AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF` — the 1.25-bit GGUF
//! - `AngelSlim/Hy-MT1.5-1.8B-1.25bit`      — the 1.25-bit safetensors
//! - `tencent/HY-MT1.5-1.8B`                — the unquantized BF16 base
//!
//! The GGUF AngelSlim repo doesn't ship `tokenizer.json`, so [`fetch_model`]
//! falls back to fetching it from `tencent/HY-MT1.5-1.8B`.

use std::path::{Component, Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};

use crate::{Error, Result};

/// Reject shard names that could escape the model directory or refer to
/// absolute paths. Only plain filename / forward-relative path components
/// are accepted, mirroring what a well-formed `model.safetensors.index.json`
/// is allowed to contain. Defends against path-traversal injected via a
/// malicious local model directory.
///
/// Backslashes are rejected unconditionally: on Unix `Path::components`
/// treats `"C:\\foo"` as a single `Normal` segment, which would silently
/// pass on non-Windows hosts. Better to reject everywhere for cross-
/// platform consistency.
fn validate_shard_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::BadPath {
            path: name.to_string(),
            reason: "empty",
        });
    }
    if name.contains('\\') {
        return Err(Error::BadPath {
            path: name.to_string(),
            reason: "contains backslash (Windows path separator)",
        });
    }
    if name.contains('\0') {
        return Err(Error::BadPath {
            path: name.to_string(),
            reason: "contains NUL byte",
        });
    }
    let p = Path::new(name);
    for c in p.components() {
        match c {
            Component::Normal(_) => {}
            _ => {
                return Err(Error::BadPath {
                    path: name.to_string(),
                    reason: "contains non-normal path component",
                })
            }
        }
    }
    Ok(())
}

/// Parse a HuggingFace `model.safetensors.index.json` and return a
/// deduplicated, sorted list of shard names. Each name is validated against
/// path traversal before being returned.
fn parse_safetensors_index(index_path: &Path) -> Result<Vec<String>> {
    let txt = std::fs::read_to_string(index_path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("reading {}: {e}", index_path.display()),
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&txt)
        .map_err(|e| Error::Gguf(format!("parsing {}: {e}", index_path.display())))?;
    let weight_map = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| Error::Gguf("safetensors index has no weight_map".into()))?;
    let mut names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    names.sort();
    names.dedup();
    for n in &names {
        validate_shard_name(n)?;
    }
    Ok(names)
}

/// HuggingFace repo identifier with optional revision.
#[derive(Debug, Clone)]
pub struct HubRef {
    pub repo_id: String,
    pub revision: Option<String>,
}

impl HubRef {
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: None,
        }
    }

    pub fn with_revision(mut self, rev: impl Into<String>) -> Self {
        self.revision = Some(rev.into());
        self
    }

    fn to_repo(&self) -> Repo {
        let repo_type = RepoType::Model;
        match &self.revision {
            None => Repo::new(self.repo_id.clone(), repo_type),
            Some(r) => Repo::with_revision(self.repo_id.clone(), repo_type, r.clone()),
        }
    }
}

/// On-disk layout for a discovered model — either GGUF or safetensors.
/// Closed enum: adding a third format (ONNX, GGML-legacy, …) would be a
/// breaking change for the CLI's match arms, so `#[non_exhaustive]`
/// wouldn't buy us forward-compat anyway.
#[derive(Debug, Clone)]
pub enum DiscoveredFormat {
    Gguf {
        gguf: PathBuf,
        /// External `tokenizer.json` path. Optional because GGUF carries
        /// `tokenizer.ggml.tokens`/`merges` and `HyTokenizer::from_gguf`
        /// can build the tokenizer entirely from the file's metadata.
        tokenizer: Option<PathBuf>,
    },
    Safetensors {
        config: PathBuf,
        tokenizer: PathBuf,
        shards: Vec<PathBuf>,
    },
}

impl DiscoveredFormat {
    /// External tokenizer path, when one was discovered or supplied.
    /// `None` for GGUF when the embedded vocab will be used.
    pub fn tokenizer(&self) -> Option<&Path> {
        match self {
            Self::Gguf { tokenizer, .. } => tokenizer.as_deref(),
            Self::Safetensors { tokenizer, .. } => Some(tokenizer.as_path()),
        }
    }
}

fn build_api() -> Result<Api> {
    ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| Error::Gguf(format!("hf-hub init failed: {e}")))
}

/// Try to download `path` from `repo`; surface a clear error including the
/// repo id so users can debug 404s without staring at hex hashes.
fn fetch_one(repo: &hf_hub::api::sync::ApiRepo, path: &str, repo_id: &str) -> Result<PathBuf> {
    repo.get(path)
        .map_err(|e| Error::Gguf(format!("failed to fetch `{path}` from `{repo_id}`: {e}")))
}

/// Resolve the tokenizer file for any model.
///
/// Order of preference:
/// 1. `tokenizer.json` in the requested repo
/// 2. fallback to `tencent/HY-MT1.5-1.8B/tokenizer.json` for repos that
///    don't ship one (e.g. `AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF`).
fn fetch_tokenizer(api: &Api, hub: &HubRef) -> Result<PathBuf> {
    let primary = api.repo(hub.to_repo());
    if let Ok(p) = primary.get("tokenizer.json") {
        return Ok(p);
    }
    tracing::info!(
        "tokenizer.json not present in `{}`; falling back to tencent/HY-MT1.5-1.8B",
        hub.repo_id
    );
    let fallback = api.repo(Repo::new("tencent/HY-MT1.5-1.8B".into(), RepoType::Model));
    fetch_one(&fallback, "tokenizer.json", "tencent/HY-MT1.5-1.8B")
}

/// First filename that exists in the repo, or `None` if none of the
/// candidates resolve.
fn first_present(repo: &hf_hub::api::sync::ApiRepo, candidates: &[&str]) -> Option<PathBuf> {
    for cand in candidates {
        if let Ok(p) = repo.get(cand) {
            return Some(p);
        }
    }
    None
}

/// Download the requested model and detect its on-disk format.
///
/// `hf_hub` is content-addressed and idempotent: subsequent calls do not
/// re-download files that already exist in the cache.
pub fn fetch_model(hub: &HubRef) -> Result<DiscoveredFormat> {
    let api = build_api()?;
    let repo = api.repo(hub.to_repo());

    // Try GGUF first, then safetensors.
    let gguf_candidates = ["Hy-MT1.5-1.8B-1.25bit.gguf", "model.gguf"];
    if let Some(gguf) = first_present(&repo, &gguf_candidates) {
        // GGUF embeds the tokenizer in metadata; we don't fetch
        // tokenizer.json by default. The caller can override with
        // `override_tokenizer` if they want a specific external file.
        return Ok(DiscoveredFormat::Gguf {
            gguf,
            tokenizer: None,
        });
    }

    // Safetensors path: prefer single-file `model.safetensors`; if absent,
    // fall back to sharded layout via `model.safetensors.index.json`.
    let config = fetch_one(&repo, "config.json", &hub.repo_id)?;
    let tokenizer = fetch_tokenizer(&api, hub)?;
    if let Ok(single) = repo.get("model.safetensors") {
        return Ok(DiscoveredFormat::Safetensors {
            config,
            tokenizer,
            shards: vec![single],
        });
    }
    // Sharded: read index, fetch each unique shard.
    let index_path = fetch_one(&repo, "model.safetensors.index.json", &hub.repo_id)?;
    let shard_names = parse_safetensors_index(&index_path)?;
    let mut shards = Vec::with_capacity(shard_names.len());
    for shard in &shard_names {
        shards.push(fetch_one(&repo, shard, &hub.repo_id)?);
    }
    Ok(DiscoveredFormat::Safetensors {
        config,
        tokenizer,
        shards,
    })
}

/// Inspect a local path (file or directory) and return the same kind of
/// [`DiscoveredFormat`] we'd hand back from [`fetch_model`]. Useful when the
/// user passes `--model <PATH>` and we want a uniform downstream code path.
///
/// `tokenizer_override` is honoured first; if `None`, the function looks for
/// a sibling `tokenizer.json`.
pub fn detect_local(path: &Path, tokenizer_override: Option<&Path>) -> Result<DiscoveredFormat> {
    if !path.exists() {
        return Err(Error::Gguf(format!(
            "path {} does not exist",
            path.display()
        )));
    }

    // For safetensors, tokenizer.json is mandatory (HF format requires it).
    // For GGUF, we prefer the embedded vocab; an external file is used only
    // when the user explicitly passes one or it sits next to the GGUF.
    let resolve_st_tokenizer = |default_dir: &Path| -> Result<PathBuf> {
        if let Some(p) = tokenizer_override {
            return Ok(p.to_path_buf());
        }
        let candidate = default_dir.join("tokenizer.json");
        if candidate.exists() {
            return Ok(candidate);
        }
        Err(Error::Gguf(format!(
            "tokenizer.json not provided and not found next to {}",
            default_dir.display()
        )))
    };
    let resolve_gguf_tokenizer = |default_dir: &Path| -> Option<PathBuf> {
        if let Some(p) = tokenizer_override {
            return Some(p.to_path_buf());
        }
        let candidate = default_dir.join("tokenizer.json");
        candidate.exists().then_some(candidate)
    };

    if path.is_file() {
        let lower = path
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase);
        let parent = path.parent().unwrap_or(Path::new("."));
        match lower.as_deref() {
            Some("gguf") => Ok(DiscoveredFormat::Gguf {
                gguf: path.to_path_buf(),
                tokenizer: resolve_gguf_tokenizer(parent),
            }),
            Some("safetensors") => Ok(DiscoveredFormat::Safetensors {
                config: parent.join("config.json"),
                tokenizer: resolve_st_tokenizer(parent)?,
                shards: vec![path.to_path_buf()],
            }),
            _ => Err(Error::Gguf(format!(
                "cannot detect format of {}",
                path.display()
            ))),
        }
    } else {
        // Directory: pick safetensors layout if `model.safetensors` is there;
        // otherwise look for any `*.gguf`.
        let single_st = path.join("model.safetensors");
        if single_st.exists() {
            return Ok(DiscoveredFormat::Safetensors {
                config: path.join("config.json"),
                tokenizer: resolve_st_tokenizer(path)?,
                shards: vec![single_st],
            });
        }
        let index = path.join("model.safetensors.index.json");
        if index.exists() {
            let names = parse_safetensors_index(&index)?;
            return Ok(DiscoveredFormat::Safetensors {
                config: path.join("config.json"),
                tokenizer: resolve_st_tokenizer(path)?,
                shards: names.into_iter().map(|n| path.join(n)).collect(),
            });
        }
        // GGUF fallback: pick the first `*.gguf` in the directory. Skip
        // anything that isn't a regular file (symlinks may point at
        // privileged or unrelated files outside the model dir).
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            if !ft.is_file() {
                continue;
            }
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
                return Ok(DiscoveredFormat::Gguf {
                    gguf: p,
                    tokenizer: resolve_gguf_tokenizer(path),
                });
            }
        }
        Err(Error::Gguf(format!(
            "no recognised model files inside {}",
            path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_shard_name_accepts_plain_filenames() {
        validate_shard_name("model.safetensors").unwrap();
        validate_shard_name("model-00001-of-00002.safetensors").unwrap();
        validate_shard_name("subdir/model.safetensors").unwrap();
    }

    #[test]
    fn validate_shard_name_rejects_traversal() {
        for bad in [
            "",
            "../etc/passwd",
            "../../etc/hosts",
            "/etc/passwd",
            "subdir/../escape.bin",
            "./hidden",
            "C:\\Windows\\System32",  // Windows-absolute on Unix
            "shard\\with\\backslash", // backslash anywhere
            "shard\0name",            // NUL byte
        ] {
            let err = validate_shard_name(bad).unwrap_err();
            match err {
                Error::BadPath { path, .. } => assert_eq!(path, bad),
                other => panic!("expected BadPath for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_safetensors_index_rejects_traversal() {
        let dir = std::env::temp_dir().join(format!(
            "hy-mt-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            r#"{"metadata":{},"weight_map":{"a":"../../../etc/passwd"}}"#,
        )
        .unwrap();
        let err = parse_safetensors_index(&index_path)
            .expect_err("traversal-poisoned index must be rejected");
        let _ = std::fs::remove_dir_all(&dir);
        match err {
            Error::BadPath { .. } => {}
            other => panic!("expected BadPath, got {other:?}"),
        }
    }
}
