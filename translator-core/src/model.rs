use std::path::Path;
use std::sync::Arc;

use candle_core::{D, Device, Tensor};
use candle_transformers::models::quantized_t5 as qt5;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use tokenizers::Tokenizer;

use crate::error::TranslatorError;
use crate::scheduler::decoder::CustomT5Decoder;

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
    // Custom decoder with externalized KV cache — for Phase 2 continuous batching.
    custom_decoder: Arc<CustomT5Decoder>,
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

        let config_str =
            std::fs::read_to_string(model_dir.join("config.json")).map_err(TranslatorError::Io)?;
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

        let model_template = qt5::T5ForConditionalGeneration::load(vb.clone(), &config)
            .map_err(|e| TranslatorError::Model(format!("model init: {e}")))?;

        let custom_decoder = Arc::new(CustomT5Decoder::load(vb, &config_str)?);

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| TranslatorError::Model(format!("tokenizer load: {e}")))?;

        tracing::info!("MADLAD-400 model loaded from {}", model_dir.display());

        Ok(Self {
            model_template,
            custom_decoder,
            tokenizer,
            device,
            eos_token_id,
            decoder_start_token_id,
        })
    }

    /// Translate a batch of strings using greedy decoding. Synchronous — always call from `spawn_blocking`.
    pub fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.translate_with_custom_decoder(texts)
    }

    /// Batched greedy decode for CUDA: single `[B, seq_len]` encode + `[B, 1]` decode per step.
    ///
    /// All B inputs are right-padded to the same length and encoded in one forward pass, then
    /// decoded together with [B, 1] tensors at each step. CUDA's higher memory bandwidth
    /// (5–10× Metal) and larger L2 cache (40–80 MB) absorb the cross-attention KV at batch
    /// sizes up to 64. Finished sequences feed `eos_token_id` as their next input to prevent
    /// garbage propagating through the KV cache.
    fn translate_greedy_batched(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        let b = texts.len();

        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_INPUT_TOKENS);
        if seq_len == 0 {
            return Ok(vec![String::new(); b]);
        }

        // Build [B, seq_len] tensor; right-pad shorter sequences with token 0 (T5 pad).
        let all_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| {
                let ids: Vec<u32> = e.get_ids().iter().take(seq_len).copied().collect();
                let pad = seq_len - ids.len();
                ids.into_iter().chain(std::iter::repeat_n(0u32, pad))
            })
            .collect();

        let input_tensor = Tensor::from_vec(all_ids, (b, seq_len), &self.device)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        let mut model = self.model_template.clone();
        model.clear_kv_cache();
        let encoder_output = model
            .encode(&input_tensor)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        let step_limit = MAX_NEW_TOKENS.min(seq_len * 3 + 32);
        let mut current_tokens = vec![self.decoder_start_token_id; b];
        let mut output_ids: Vec<Vec<u32>> = vec![vec![]; b];
        let mut finished = vec![false; b];

        for _ in 0..step_limit {
            if finished.iter().all(|&f| f) {
                break;
            }
            let dec = Tensor::from_vec(current_tokens.clone(), (b, 1), &self.device)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            // decode() returns [B, vocab_size]
            let logits = model
                .decode(&dec, &encoder_output)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            let next: Vec<u32> = logits
                .argmax(D::Minus1)
                .map_err(|e| TranslatorError::Model(e.to_string()))?
                .to_vec1::<u32>()
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
            // Feed EOS as next input for finished sequences to prevent garbage KV cache propagation.
            current_tokens = next
                .into_iter()
                .enumerate()
                .map(|(i, tok)| if finished[i] { self.eos_token_id } else { tok })
                .collect();
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

    /// Encode a batch of texts and return the encoder hidden states.
    ///
    /// Returns `(encoder_output, seq_len)` where `encoder_output` is
    /// `[B, seq_len, d_model]`.  Used by the Phase 2 scheduler to encode
    /// a batch and then dispatch slots to the continuous decode loop.
    pub fn encode_only(&self, texts: &[String]) -> Result<(Tensor, usize), TranslatorError> {
        if texts.is_empty() {
            return Err(TranslatorError::Model("encode_only: empty input".into()));
        }
        let b = texts.len();

        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(t.as_str(), true)
                    .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_INPUT_TOKENS);
        if seq_len == 0 {
            return Err(TranslatorError::Model("encode_only: empty tokens".into()));
        }

        let all_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| {
                let ids: Vec<u32> = e.get_ids().iter().take(seq_len).copied().collect();
                let pad = seq_len - ids.len();
                ids.into_iter().chain(std::iter::repeat_n(0u32, pad))
            })
            .collect();

        let input_tensor = Tensor::from_vec(all_ids, (b, seq_len), &self.device)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        let mut model = self.model_template.clone();
        model.clear_kv_cache();
        let encoder_output = model
            .encode(&input_tensor)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        Ok((encoder_output, seq_len))
    }

    /// Translate a batch using the custom decoder (Phase 2a validation).
    ///
    /// Produces byte-identical output to `translate_greedy_batched` while
    /// exercising the externalized KV-cache path used by the continuous scheduler.
    pub fn translate_with_custom_decoder(
        &self,
        texts: &[String],
    ) -> Result<Vec<String>, TranslatorError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let b = texts.len();

        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(t.as_str(), true)
                    .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_INPUT_TOKENS);
        if seq_len == 0 {
            return Ok(vec![String::new(); b]);
        }

        let all_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| {
                let ids: Vec<u32> = e.get_ids().iter().take(seq_len).copied().collect();
                let pad = seq_len - ids.len();
                ids.into_iter().chain(std::iter::repeat_n(0u32, pad))
            })
            .collect();

        let input_tensor = Tensor::from_vec(all_ids, (b, seq_len), &self.device)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        let mut model = self.model_template.clone();
        model.clear_kv_cache();
        let encoder_output = model
            .encode(&input_tensor)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        let mut cache = self.custom_decoder.new_kv_cache(b)?;
        self.custom_decoder
            .compute_cross_kv(&encoder_output, &mut cache)?;

        let step_limit = MAX_NEW_TOKENS.min(seq_len * 3 + 32);
        let mut current_tokens = vec![self.decoder_start_token_id; b];
        let mut output_ids: Vec<Vec<u32>> = vec![vec![]; b];
        let mut finished = vec![false; b];

        for _ in 0..step_limit {
            if finished.iter().all(|&f| f) {
                break;
            }
            let dec = Tensor::from_vec(current_tokens.clone(), (b, 1), &self.device)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            let logits = self.custom_decoder.decode_step(&dec, &mut cache)?;
            let next: Vec<u32> = logits
                .argmax(D::Minus1)
                .map_err(|e| TranslatorError::Model(e.to_string()))?
                .to_vec1::<u32>()
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
            current_tokens = next
                .into_iter()
                .enumerate()
                .map(|(i, tok)| if finished[i] { self.eos_token_id } else { tok })
                .collect();
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
