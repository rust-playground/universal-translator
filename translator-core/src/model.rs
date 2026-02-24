use std::path::Path;

use candle_core::{Device, Tensor, D};
use candle_transformers::models::quantized_t5 as qt5;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use tokenizers::Tokenizer;

use crate::error::TranslatorError;

const MAX_INPUT_TOKENS: usize = 1024;
const MAX_NEW_TOKENS: usize = 1024;

/// Select the best available inference device in priority order:
///   CUDA (if compiled in and device present) → Metal (macOS) → CPU
fn select_device() -> Result<Device, TranslatorError> {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            tracing::info!("inference device: CUDA");
            return Device::new_cuda(0)
                .map_err(|e| TranslatorError::Model(format!("CUDA init: {e}")));
        }
    }

    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            tracing::info!("inference device: Metal");
            return Device::new_metal(0)
                .map_err(|e| TranslatorError::Model(format!("Metal init: {e}")));
        }
    }

    tracing::info!("inference device: CPU");
    Ok(Device::Cpu)
}

/// A loaded MADLAD-400-3B-MT model with its HuggingFace tokenizer.
///
/// Loaded from a directory containing:
///   model-q4k.gguf  — quantized weights (Q4_K, ~1.65 GB)
///   config.json     — T5 model config
///   tokenizer.json  — HuggingFace fast tokenizer
///
/// Weight tensors are Arc-backed, so cloning the model template is cheap
/// (~100 refcount increments). Each `translate_batch` call works on its own
/// clone, allowing concurrent API requests to run without contention.
pub struct LoadedModel {
    // Arc-backed weights; cloning is cheap and produces an independent KV-cache state.
    model_template: qt5::T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    decoder_start_token_id: u32,
}

// SAFETY: all weight tensors inside T5ForConditionalGeneration are Arc-backed
// (QMatMul wraps Arc<QTensor>; Tensor wraps Arc<Tensor_>). The model is
// treated as a read-only template — mutations happen only on per-call clones.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

impl LoadedModel {
    /// Load the model directory. `_num_threads` is accepted for API compatibility
    /// but Candle manages its own threading via the runtime/device.
    pub fn load(model_dir: &Path, _num_threads: usize) -> Result<Self, TranslatorError> {
        let device = select_device()?;

        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(TranslatorError::Io)?;
        let config: qt5::Config = serde_json::from_str(&config_str)
            .map_err(|e| TranslatorError::Model(format!("config parse: {e}")))?;

        let eos_token_id = config.eos_token_id as u32;
        let decoder_start_token_id = config.decoder_start_token_id.unwrap_or(0) as u32;

        let gguf_path = model_dir.join("model-q4k.gguf");
        if !gguf_path.exists() {
            return Err(TranslatorError::ModelNotFound(format!(
                "{} not found — run models/download.sh",
                gguf_path.display()
            )));
        }

        let vb = QVarBuilder::from_gguf(&gguf_path, &device)
            .map_err(|e| TranslatorError::Model(format!("GGUF load: {e}")))?;

        let model_template = qt5::T5ForConditionalGeneration::load(vb, &config)
            .map_err(|e| TranslatorError::Model(format!("model init: {e}")))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| TranslatorError::Model(format!("tokenizer load: {e}")))?;

        tracing::info!("MADLAD-400 model loaded from {}", model_dir.display());

        Ok(Self {
            model_template,
            tokenizer,
            device,
            eos_token_id,
            decoder_start_token_id,
        })
    }

    /// Translate a batch of strings using true batched inference. Synchronous — always call from `spawn_blocking`.
    ///
    /// All `texts` must be the same tokenized length (guaranteed when called per-chunk
    /// from engine.rs, where every item is `"<2xx> <same_source_text>"`).
    ///
    /// Clones the model template cheaply (Arc refcount increments on shared weights)
    /// to obtain an independent instance with a fresh KV cache.
    pub fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        let b = texts.len();
        if b == 0 {
            return Ok(vec![]);
        }

        // Cheap clone: ~100 Arc refcount increments; all weight tensors are shared.
        let mut model = self.model_template.clone();

        // Tokenize all inputs, truncating to MAX_INPUT_TOKENS each.
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // All items in a chunk share the same source text — identical length.
        // Use the first encoding's length, capped at MAX_INPUT_TOKENS.
        let seq_len = encodings[0].get_ids().len().min(MAX_INPUT_TOKENS);

        if seq_len == 0 {
            return Ok(vec![String::new(); b]);
        }

        // Build [B, seq_len] input tensor by interleaving all token ID rows.
        let all_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().take(MAX_INPUT_TOKENS).copied())
            .collect();

        let input_tensor = Tensor::from_vec(all_ids, (b, seq_len), &self.device)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        model.clear_kv_cache();

        // Encode the full [B, seq_len] source batch once.
        let encoder_output = model
            .encode(&input_tensor)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        // Greedy decode: one [B, 1] step at a time; each call returns [B, vocab_size].
        let step_limit = MAX_NEW_TOKENS.min(seq_len * 3 + 32);

        let mut current_tokens = vec![self.decoder_start_token_id; b];
        let mut output_ids: Vec<Vec<u32>> = vec![vec![]; b];
        let mut finished = vec![false; b];

        for _ in 0..step_limit {
            if finished.iter().all(|&f| f) {
                break;
            }

            // Shape [B, 1] — one token per sequence for this decode step.
            let dec = Tensor::from_vec(current_tokens.clone(), (b, 1), &self.device)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;

            // decode() returns [B, vocab_size] — last position already selected.
            let logits = model
                .decode(&dec, &encoder_output)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;

            // [B, vocab_size] → [B] next token ids.
            let next: Vec<u32> = logits
                .argmax(D::Minus1)
                .map_err(|e| TranslatorError::Model(e.to_string()))?
                .to_vec1()
                .map_err(|e| TranslatorError::Model(e.to_string()))?;

            for (i, &tok) in next.iter().enumerate() {
                if !finished[i] {
                    if tok == self.eos_token_id {
                        finished[i] = true;
                    } else {
                        output_ids[i].push(tok);
                    }
                }
            }
            current_tokens = next;
        }

        output_ids
            .iter()
            .map(|ids| {
                self.tokenizer
                    .decode(ids, true)
                    .map_err(|e| TranslatorError::Model(format!("decode: {e}")))
            })
            .collect()
    }
}
