use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::error::TranslatorError;
use crate::model_batched::{ModelWeights, SlotKvCache};
use crate::scheduler::decoder::GemmaSlotDecoder;

fn cerr(e: candle_core::Error) -> TranslatorError {
    TranslatorError::Model(e.to_string())
}

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

/// A loaded TranslateGemma 4B model with its HuggingFace tokenizer.
///
/// Loaded from a directory containing:
///   model-q4k.gguf   — quantized weights (Q4_K, ~2.5 GB)
///   tokenizer.json   — HuggingFace fast tokenizer
///
/// Weights are Arc-shared inside the local `ModelWeights` type.  KV cache is
/// stored externally in per-slot [`SlotKvCache`] structs, so this struct is
/// entirely read-only after loading and can be shared across threads.
pub struct LoadedGemmaModel {
    /// Stateless model weights — KV cache is external.
    pub(crate) model_weights: ModelWeights,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    pub(crate) eos_token_id: u32,
}

// SAFETY: ModelWeights weights are Arc-backed; no mutable state lives here after
// loading. KV mutations happen in per-slot SlotKvCache values owned by the caller.
unsafe impl Send for LoadedGemmaModel {}
unsafe impl Sync for LoadedGemmaModel {}

impl LoadedGemmaModel {
    /// Load the model directory.
    pub fn load(model_dir: &Path) -> Result<Self, TranslatorError> {
        let device = select_device()?;

        let gguf_path = model_dir.join("model-q4k.gguf");
        if !gguf_path.exists() {
            return Err(TranslatorError::ModelNotFound(format!(
                "{} not found — run models/download.sh",
                gguf_path.display()
            )));
        }

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| TranslatorError::Model(format!("tokenizer load: {e}")))?;

        let eos_token_id = tokenizer
            .token_to_id("<end_of_turn>")
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .unwrap_or(1);
        tracing::info!("eos_token_id={eos_token_id}");

        let mut reader = std::fs::File::open(&gguf_path).map_err(TranslatorError::Io)?;
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| TranslatorError::Model(format!("GGUF read: {e}")))?;

        let model_weights = ModelWeights::from_gguf(content, &mut reader, &device)
            .map_err(|e| TranslatorError::Model(format!("model init: {e}")))?;

        tracing::info!("TranslateGemma model loaded from {}", model_dir.display());

        Ok(Self {
            model_weights,
            tokenizer: Arc::new(tokenizer),
            device,
            eos_token_id,
        })
    }

    // ── Public accessors ─────────────────────────────────────────────────────

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn n_layers(&self) -> usize {
        self.model_weights.n_layers()
    }

    /// Create a fresh per-slot decoder backed by an empty [`SlotKvCache`].
    ///
    /// The weights themselves are not cloned — the decoder only holds a KV cache
    /// and a reference to the device.
    pub fn new_slot_decoder(&self) -> GemmaSlotDecoder {
        GemmaSlotDecoder::new(
            SlotKvCache::new(self.model_weights.n_layers()),
            self.device.clone(),
        )
    }

    /// Single-slot forward pass.
    ///
    /// Wraps [`ModelWeights::forward`] with `TranslatorError` mapping.
    pub fn forward_single(
        &self,
        x: &Tensor,
        index_pos: usize,
        kv: &mut SlotKvCache,
    ) -> Result<Tensor, TranslatorError> {
        self.model_weights.forward(x, index_pos, kv).map_err(cerr)
    }

    /// N-slot batched decode step.
    ///
    /// Wraps [`ModelWeights::forward_batched`] with `TranslatorError` mapping.
    pub fn forward_batched(
        &self,
        tokens: &Tensor,
        kv_caches: &mut [SlotKvCache],
    ) -> Result<Tensor, TranslatorError> {
        self.model_weights.forward_batched(tokens, kv_caches).map_err(cerr)
    }

    /// Tokenize a prompt string to token IDs.  Does NOT add special tokens —
    /// the caller is responsible for including `<bos>` and chat template tokens
    /// in the prompt string itself.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, TranslatorError> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a sequence of token IDs to a UTF-8 string, skipping special tokens.
    pub fn decode_output_ids(&self, ids: &[u32]) -> Result<String, TranslatorError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| TranslatorError::Model(format!("decode: {e}")))
    }
}
