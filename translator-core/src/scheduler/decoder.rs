//! Per-slot Gemma decoder.
//!
//! [`GemmaSlotDecoder`] wraps a per-slot clone of the Gemma model weights.
//! Because all weight tensors inside `quantized_gemma3::ModelWeights` are
//! Arc-backed (via QMatMul/QTensor), cloning is cheap — only KV-cache tensors
//! are unique to each slot and grow incrementally during generation.
//!
//! Usage:
//! 1. Call `prefill(token_ids)` to process the full prompt.  The model
//!    populates its per-layer KV cache and returns logits at the last position.
//! 2. Call `decode_step(token_id)` for each subsequent token.  `index_pos`
//!    is tracked internally and incremented after every call.

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_gemma3;

use crate::error::TranslatorError;

// ── Error helper ──────────────────────────────────────────────────────────────

fn cerr(e: candle_core::Error) -> TranslatorError {
    TranslatorError::Model(e.to_string())
}

// ── GemmaSlotDecoder ──────────────────────────────────────────────────────────

/// A per-slot inference state wrapping one clone of the Gemma model weights.
///
/// The `model` field accumulates KV cache in-place as tokens are processed.
/// Create via [`LoadedGemmaModel::new_slot_decoder`]; drop when the slot retires.
pub struct GemmaSlotDecoder {
    model: quantized_gemma3::ModelWeights,
    /// Number of tokens processed so far (= index_pos for the next forward call).
    step: usize,
    device: Device,
}

// SAFETY: ModelWeights is Arc-backed; per-slot KV caches are owned by this
// decoder and not shared across threads.
unsafe impl Send for GemmaSlotDecoder {}

impl GemmaSlotDecoder {
    pub fn new(model: quantized_gemma3::ModelWeights, device: Device) -> Self {
        Self { model, step: 0, device }
    }

    /// Process the full prompt in one forward pass.
    ///
    /// Populates the KV cache for all `token_ids.len()` prompt tokens and
    /// returns logits `[vocab_size]` at the last prompt position (used to
    /// sample the first output token).
    ///
    /// After this call `self.step == token_ids.len()`.
    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, TranslatorError> {
        if token_ids.is_empty() {
            return Err(TranslatorError::Model("prefill: empty token_ids".into()));
        }
        let seq_len = token_ids.len();
        let input = Tensor::from_slice(token_ids, (1, seq_len), &self.device).map_err(cerr)?;

        // index_pos = 0: this is the start of the sequence.
        let logits = self.model.forward(&input, 0).map_err(cerr)?;

        self.step = seq_len;
        extract_last_logits(logits)
    }

    /// Process a single token and return logits `[vocab_size]` for the next position.
    ///
    /// `token_id` is the token that was just sampled (will be placed at
    /// position `self.step`).  `self.step` is incremented after the call.
    pub fn decode_step(&mut self, token_id: u32) -> Result<Vec<f32>, TranslatorError> {
        let input = Tensor::from_slice(&[token_id], (1usize, 1usize), &self.device)
            .map_err(cerr)?;

        let logits = self.model.forward(&input, self.step).map_err(cerr)?;

        self.step += 1;
        extract_last_logits(logits)
    }
}

/// Extract the logit vector for the last (and only relevant) token position.
///
/// `model.forward` returns shape `[batch, 1, vocab_size]`.  For B=1 this is
/// `[1, 1, vocab_size]`.  We flatten to `[vocab_size]` in f32.
fn extract_last_logits(logits: Tensor) -> Result<Vec<f32>, TranslatorError> {
    // [1, 1, vocab_size] → [vocab_size] in f32
    logits
        .squeeze(0)
        .and_then(|t| t.squeeze(0))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(cerr)
}
