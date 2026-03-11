use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::error::TranslatorError;

/// Find the first `*.gguf` file in a directory.
fn find_gguf_file(dir: &Path) -> Result<PathBuf, TranslatorError> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(TranslatorError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "gguf"))
        .collect();
    entries.sort();
    entries.into_iter().next().ok_or_else(|| {
        TranslatorError::ModelNotFound(format!(
            "no .gguf file found in {} — run models/download.sh",
            dir.display()
        ))
    })
}

/// Resolve the GGUF model file path.
///
/// Priority:
/// 1. Explicit `model_file` param (from `--model-file` flag)
/// 2. `MODEL_FILE` env var
/// 3. `model-q8_0.gguf` (preferred default — higher precision)
/// 4. `model-q4k.gguf` (fallback if Q8_0 not present)
/// 5. Any `*.gguf` in directory (last resort)
fn resolve_gguf_path(model_dir: &Path, model_file: Option<&str>) -> Result<PathBuf, TranslatorError> {
    // 1. Explicit --model-file flag
    if let Some(name) = model_file {
        let path = model_dir.join(name);
        if !path.exists() {
            return Err(TranslatorError::ModelNotFound(format!(
                "--model-file {name} not found at {}",
                path.display()
            )));
        }
        return Ok(path);
    }

    // 2. MODEL_FILE env var
    if let Ok(name) = std::env::var("MODEL_FILE") {
        let path = model_dir.join(&name);
        if !path.exists() {
            return Err(TranslatorError::ModelNotFound(format!(
                "MODEL_FILE={name} not found at {}",
                path.display()
            )));
        }
        return Ok(path);
    }

    // 3. Q8_0 default (higher precision, comparable throughput under llama.cpp)
    let q8 = model_dir.join("model-q8_0.gguf");
    if q8.exists() {
        return Ok(q8);
    }

    // 4. Q4_K_M fallback
    let q4k = model_dir.join("model-q4k.gguf");
    if q4k.exists() {
        return Ok(q4k);
    }

    // 5. Any *.gguf in directory
    find_gguf_file(model_dir)
}

/// A loaded TranslateGemma 4B model backed by llama.cpp.
///
/// Loaded from a directory containing a `*.gguf` file with quantized weights.
/// The GGUF file embeds the tokenizer, so no separate `tokenizer.json` is needed.
///
/// `LlamaBackend` and `LlamaModel` are stored here; inference contexts
/// (`LlamaContext`) are created on the scheduler thread since they are `!Send`.
pub struct LoadedGemmaModel {
    // Drop order matters: model must drop before backend (declaration order).
    model: LlamaModel,
    backend: LlamaBackend,
    pub(crate) eos_token_id: u32,
}

// SAFETY: LlamaBackend is a process-wide singleton init guard.
// LlamaModel holds read-only weights behind a raw pointer; llama.cpp
// guarantees thread-safe read access to model weights after loading.
unsafe impl Send for LoadedGemmaModel {}
unsafe impl Sync for LoadedGemmaModel {}

impl LoadedGemmaModel {
    /// Load the model from a directory containing a GGUF file.
    ///
    /// `model_file` overrides automatic GGUF file selection (e.g. `"model-q8_0.gguf"`).
    pub fn load(model_dir: &Path, model_file: Option<&str>) -> Result<Self, TranslatorError> {
        let mut backend = LlamaBackend::init()
            .map_err(|e| TranslatorError::Model(format!("llama backend init: {e}")))?;

        // Suppress verbose llama.cpp/ggml logs unless LLAMA_LOG is set.
        if std::env::var("LLAMA_LOG").is_err() {
            backend.void_logs();
        }

        let gguf_path = resolve_gguf_path(model_dir, model_file)?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(999);

        let model = LlamaModel::load_from_file(&backend, &gguf_path, &model_params)
            .map_err(|e| TranslatorError::Model(format!("model load: {e}")))?;

        let eos_token_id = model.token_eos().0 as u32;
        tracing::info!("eos_token_id={eos_token_id}");

        tracing::info!(
            "TranslateGemma model loaded: {} (from {})",
            gguf_path.file_name().unwrap_or_default().to_string_lossy(),
            model_dir.display()
        );

        Ok(Self {
            model,
            backend,
            eos_token_id,
        })
    }

    // ── Public accessors ─────────────────────────────────────────────────────

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Check if a token is an end-of-generation token (covers both `<eos>` and
    /// `<end_of_turn>` for Gemma models).
    pub fn is_eog_token(&self, token_id: u32) -> bool {
        self.model.is_eog_token(LlamaToken(token_id as i32))
    }

    /// Create a new inference context on the current thread.
    ///
    /// `LlamaContext` is `!Send` — must be created and used on the scheduler thread.
    pub fn create_context(
        &self,
        n_ctx: u32,
        n_seq_max: u32,
    ) -> Result<LlamaContext<'_>, TranslatorError> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        // LLAMA_FLASH_ATTN_TYPE_ENABLED = 1 (from llama.h).
        // Explicit ENABLED is needed so flash attention remains active even
        // with quantized KV cache (AUTO would disable it).
        let flash_attn_enabled: i32 = 1;

        let params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(n_ctx).expect("n_ctx must be > 0"),
            ))
            .with_n_batch(n_ctx)
            .with_n_seq_max(n_seq_max)
            .with_flash_attention_policy(flash_attn_enabled)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        tracing::info!(n_threads, "llama context: flash_attn=enabled");

        self.model
            .new_context(&self.backend, params)
            .map_err(|e| TranslatorError::Model(format!("context creation: {e}")))
    }

    /// Tokenize a prompt string to token IDs.
    ///
    /// Does NOT add BOS — the caller includes `<bos>` in the prompt string.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, TranslatorError> {
        self.model
            .str_to_token(text, AddBos::Never)
            .map(|tokens| tokens.into_iter().map(|t| t.0 as u32).collect())
            .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
    }

    /// Decode a sequence of token IDs to a UTF-8 string.
    pub fn decode_output_ids(&self, ids: &[u32]) -> Result<String, TranslatorError> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut result = String::new();
        for &id in ids {
            let piece = self
                .model
                .token_to_piece(LlamaToken(id as i32), &mut decoder, false, None)
                .map_err(|e| TranslatorError::Model(format!("decode token {id}: {e}")))?;
            result.push_str(&piece);
        }
        Ok(result)
    }
}
