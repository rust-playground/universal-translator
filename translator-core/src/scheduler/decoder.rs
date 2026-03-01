//! Per-slot Gemma decoder.
//!
//! [`GemmaSlotDecoder`] holds the per-slot [`SlotKvCache`] and manages prefill.
//! Token-by-token decode is handled by the scheduler via the batched forward path
//! ([`LoadedGemmaModel::forward_batched`]), so this struct no longer wraps a model
//! clone — only the KV cache and device are stored here.
//!
//! Usage:
//! 1. Call `prefill(model, token_ids)` to process the full prompt.  The KV cache
//!    is populated and logits at the last prompt position are returned (used to
//!    sample the first output token).
//! 2. The scheduler drives all subsequent decode steps via `forward_batched`.

use candle_core::{DType, Device, Tensor};

use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::model_batched::SlotKvCache;

// ── Error helper ──────────────────────────────────────────────────────────────

fn cerr(e: candle_core::Error) -> TranslatorError {
    TranslatorError::Model(e.to_string())
}

// ── GemmaSlotDecoder ──────────────────────────────────────────────────────────

/// Per-slot inference state.
///
/// Holds only the KV cache and device — the model weights are shared and accessed
/// via [`LoadedGemmaModel`] by the scheduler.  Create via
/// [`LoadedGemmaModel::new_slot_decoder`]; drop when the slot retires.
pub struct GemmaSlotDecoder {
    pub kv_cache: SlotKvCache,
    /// Number of tokens processed so far (= `index_pos` for the next call).
    pub step: usize,
    pub(crate) device: Device,
}

// SAFETY: SlotKvCache tensors are owned by this decoder and not shared.
unsafe impl Send for GemmaSlotDecoder {}

impl GemmaSlotDecoder {
    pub fn new(kv_cache: SlotKvCache, device: Device) -> Self {
        Self { kv_cache, step: 0, device }
    }

    /// Process the full prompt in one forward pass.
    ///
    /// Populates the KV cache for all prompt tokens and returns logits
    /// `[vocab_size]` at the last prompt position (used to sample the first
    /// output token).
    ///
    /// After this call `self.step == token_ids.len()` and
    /// `self.kv_cache.seq_len == token_ids.len()`.
    pub fn prefill(
        &mut self,
        model: &LoadedGemmaModel,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, TranslatorError> {
        if token_ids.is_empty() {
            return Err(TranslatorError::Model("prefill: empty token_ids".into()));
        }
        let seq_len = token_ids.len();
        let input =
            Tensor::from_slice(token_ids, (1, seq_len), &self.device).map_err(cerr)?;

        // index_pos = 0: start of a fresh sequence.
        let logits = model.forward_single(&input, 0, &mut self.kv_cache)?;

        self.step = seq_len;
        extract_last_logits(logits)
    }
}

/// Extract the logit vector for the last (and only relevant) token position.
///
/// `forward_single` returns shape `[1, vocab_size]`.  We convert to f32 Vec.
pub(crate) fn extract_last_logits(logits: Tensor) -> Result<Vec<f32>, TranslatorError> {
    logits
        .squeeze(0)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(cerr)
}
