use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_gemma3;
use tokenizers::Tokenizer;

use crate::error::TranslatorError;
use crate::scheduler::decoder::GemmaSlotDecoder;

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
/// The model template is Clone-cheap (Arc-backed weights). Each decode slot
/// gets its own fresh clone via `new_slot_decoder()`.
pub struct LoadedGemmaModel {
    /// Template model — weights are Arc-shared. Never used for inference directly.
    /// Each slot clones this to get its own KV-cache state.
    model_template: quantized_gemma3::ModelWeights,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    pub(crate) eos_token_id: u32,
    pub(crate) bos_token_id: u32,
}

// SAFETY: Gemma weight tensors (QMatMul, RmsNorm, Embedding) are Arc-backed;
// the model template is treated as read-only — KV cache mutations happen only on
// per-slot clones. Same reasoning as the previous LoadedModel unsafe impls.
unsafe impl Send for LoadedGemmaModel {}
unsafe impl Sync for LoadedGemmaModel {}

impl LoadedGemmaModel {
    /// Load the model directory.
    pub fn load(model_dir: &Path, _num_threads: usize) -> Result<Self, TranslatorError> {
        let device = select_device()?;

        let gguf_path = model_dir.join("model-q4k.gguf");
        if !gguf_path.exists() {
            return Err(TranslatorError::ModelNotFound(format!(
                "{} not found — run models/download.sh",
                gguf_path.display()
            )));
        }

        // Load tokenizer first so we can look up EOS/BOS token IDs.
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| TranslatorError::Model(format!("tokenizer load: {e}")))?;

        // Determine EOS token: prefer <end_of_turn> (Gemma instruct), fall back to <eos>.
        let eos_token_id = tokenizer
            .token_to_id("<end_of_turn>")
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .unwrap_or(1);
        let bos_token_id = tokenizer.token_to_id("<bos>").unwrap_or(2);

        tracing::info!("eos_token_id={eos_token_id}, bos_token_id={bos_token_id}");

        // Load GGUF weights.
        let mut reader = std::fs::File::open(&gguf_path).map_err(TranslatorError::Io)?;
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| TranslatorError::Model(format!("GGUF read: {e}")))?;

        let model_template =
            quantized_gemma3::ModelWeights::from_gguf(content, &mut reader, &device)
                .map_err(|e| TranslatorError::Model(format!("model init: {e}")))?;

        tracing::info!("TranslateGemma model loaded from {}", model_dir.display());

        Ok(Self {
            model_template,
            tokenizer: Arc::new(tokenizer),
            device,
            eos_token_id,
            bos_token_id,
        })
    }

    // ── Public accessors ─────────────────────────────────────────────────────

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Create a fresh per-slot decoder. Weights are Arc-shared with the template;
    /// only per-slot KV cache memory is allocated on first forward pass.
    pub fn new_slot_decoder(&self) -> GemmaSlotDecoder {
        GemmaSlotDecoder::new(self.model_template.clone(), self.device.clone())
    }

    /// Tokenize a prompt string to token IDs. Does NOT add special tokens —
    /// the caller is responsible for including <bos> and chat template tokens
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
