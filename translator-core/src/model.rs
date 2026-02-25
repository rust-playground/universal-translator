use std::path::Path;

use candle_core::{Device, Tensor, D};
use candle_transformers::models::quantized_t5 as qt5;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use tokenizers::Tokenizer;

use crate::error::TranslatorError;

const MAX_INPUT_TOKENS: usize = 1024;
const MAX_NEW_TOKENS: usize = 1024;
/// Standard T5 length-normalization exponent for beam scoring.
const BEAM_LENGTH_PENALTY: f32 = 0.6;

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

    /// Translate a batch of strings. Synchronous — always call from `spawn_blocking`.
    ///
    /// Dispatches to:
    ///   - Batched greedy  (beam_width ≤ 1): `[B, seq_len]` encode + `[B, 1]` decode loop
    ///   - Beam search     (beam_width ≥ 2, all devices): per-text with independent KV caches
    pub fn translate_batch(&self, texts: &[String], beam_width: u8) -> Result<Vec<String>, TranslatorError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        if beam_width <= 1 {
            self.translate_batch_greedy(texts)
        } else {
            texts.iter().map(|t| self.translate_beam(t, beam_width as usize)).collect()
        }
    }

    /// Batched greedy decode using true `[B, seq_len]` encoder input and `[B, 1]` decoder steps.
    ///
    /// Inputs may have different tokenized lengths (e.g. cross-request batching). Sequences
    /// shorter than `seq_len` are right-padded with token ID 0 (T5 pad token). Same-length
    /// inputs incur zero padding overhead (preserves existing intra-request chunk behavior).
    fn translate_batch_greedy(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        let b = texts.len();

        let mut model = self.model_template.clone();

        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let seq_len = encodings.iter()
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
                ids.into_iter().chain(std::iter::repeat(0u32).take(pad))
            })
            .collect();

        let input_tensor = Tensor::from_vec(all_ids, (b, seq_len), &self.device)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

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

    /// Beam search decode for a single text. Each beam maintains an independent KV cache
    /// (model clone). Arc-backed weights are shared across all clones for free.
    fn translate_beam(&self, text: &str, beam_width: usize) -> Result<String, TranslatorError> {
        let input_ids: Vec<u32> = self.tokenizer
            .encode(text, true)
            .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))?
            .get_ids()
            .iter()
            .take(MAX_INPUT_TOKENS)
            .copied()
            .collect();
        if input_ids.is_empty() {
            return Ok(String::new());
        }

        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        // Encode once; all beams share the same [1, seq_len, d_model] encoder output.
        let mut seed_model = self.model_template.clone();
        seed_model.clear_kv_cache();
        let encoder_output = seed_model
            .encode(&input_tensor)
            .map_err(|e| TranslatorError::Model(e.to_string()))?;

        struct Beam {
            model: qt5::T5ForConditionalGeneration,
            tokens: Vec<u32>,
            score: f32,
            current_token: u32,
            finished: bool,
        }

        let mut beams: Vec<Beam> = (0..beam_width)
            .map(|_| Beam {
                model: seed_model.clone(),
                tokens: vec![],
                score: 0.0,
                current_token: self.decoder_start_token_id,
                finished: false,
            })
            .collect();

        let mut completed: Vec<(f32, Vec<u32>)> = vec![];

        for _ in 0..MAX_NEW_TOKENS {
            if beams.iter().all(|b| b.finished) {
                break;
            }

            // Each active beam produces beam_width candidates; keep the best beam_width overall.
            let mut candidates: Vec<(f32, u32, usize)> = vec![]; // (score, token, parent_beam_idx)

            for (bi, beam) in beams.iter_mut().enumerate() {
                if beam.finished {
                    continue;
                }

                let dec = Tensor::new(&[beam.current_token], &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| TranslatorError::Model(e.to_string()))?;
                // decode() returns [1, vocab_size]
                let logits = beam
                    .model
                    .decode(&dec, &encoder_output)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?;

                // Bring logits to CPU then compute numerically stable log_softmax in Rust.
                // Avoids reliance on a GPU log_softmax op (not available in candle 0.8).
                let raw: Vec<f32> = logits
                    .squeeze(0)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| TranslatorError::Model(e.to_string()))?;
                let max_l = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let log_sum_exp = max_l + raw.iter().map(|&x| (x - max_l).exp()).sum::<f32>().ln();
                let log_probs: Vec<f32> = raw.iter().map(|&x| x - log_sum_exp).collect();

                // Partial sort: collect top-beam_width tokens from this beam's distribution.
                let mut indexed: Vec<(usize, f32)> = log_probs.into_iter().enumerate().collect();
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (token_id, log_p) in indexed.into_iter().take(beam_width) {
                    candidates.push((beam.score + log_p, token_id as u32, bi));
                }
            }

            // Globally select top-beam_width candidates.
            candidates.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(beam_width);

            // Snapshot parent model states (with their accumulated KV caches) before rebuild.
            // Future decode calls on the clones extend independent KV caches via `cat`.
            let parent_models: Vec<qt5::T5ForConditionalGeneration> =
                candidates.iter().map(|(_, _, p)| beams[*p].model.clone()).collect();
            let parent_tokens: Vec<Vec<u32>> =
                candidates.iter().map(|(_, _, p)| beams[*p].tokens.clone()).collect();

            beams = candidates
                .into_iter()
                .zip(parent_models)
                .zip(parent_tokens)
                .map(|(((score, token, _), model), mut tokens)| {
                    let finished = token == self.eos_token_id;
                    if !finished {
                        tokens.push(token);
                    }
                    Beam { model, tokens, score, current_token: token, finished }
                })
                .collect();

            for beam in &beams {
                if beam.finished {
                    completed.push((beam.score, beam.tokens.clone()));
                }
            }
        }

        // Add unfinished beams as fallback (truncated at max steps).
        for beam in &beams {
            if !beam.finished && !beam.tokens.is_empty() {
                completed.push((beam.score, beam.tokens.clone()));
            }
        }

        // Pick best hypothesis by length-normalized score.
        let best_tokens = completed
            .into_iter()
            .max_by(|(s_a, t_a), (s_b, t_b)| {
                let norm_a = s_a / (t_a.len().max(1) as f32).powf(BEAM_LENGTH_PENALTY);
                let norm_b = s_b / (t_b.len().max(1) as f32).powf(BEAM_LENGTH_PENALTY);
                norm_a.partial_cmp(&norm_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, tokens)| tokens)
            .unwrap_or_default();

        self.tokenizer
            .decode(&best_tokens, true)
            .map_err(|e| TranslatorError::Model(format!("decode: {e}")))
    }
}
