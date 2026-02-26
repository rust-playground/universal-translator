use std::path::Path;

use ct2rs::sys::Translator as SysTranslator;
use ct2rs::tokenizers::sentencepiece::Tokenizer as SpmTokenizer;
use ct2rs::{BatchType, ComputeType, Config, Device, Tokenizer, TranslationOptions};

use crate::error::TranslatorError;

/// Select the inference device and compute type.
///
/// When the `cuda` feature is compiled in and a CUDA device is present at runtime,
/// uses CUDA with FLOAT16 (the standard high-performance CUDA compute type).
/// Falls back to the best CPU compute type for the current architecture.
fn select_device_and_compute() -> (Device, ComputeType) {
    #[cfg(feature = "cuda")]
    if ct2rs::sys::get_device_count(Device::CUDA) > 0 {
        return (Device::CUDA, ComputeType::FLOAT16);
    }

    #[cfg(target_arch = "aarch64")]
    return (Device::CPU, ComputeType::FLOAT32); // ARM NEON: FLOAT32 > INT8 for small-batch inference

    #[cfg(target_arch = "x86_64")]
    return (
        Device::CPU,
        if std::is_x86_feature_detected!("avx2") {
            ComputeType::INT8 // AVX2 VNNI makes INT8 2-3× faster; CTranslate2 quantizes at load time
        } else {
            ComputeType::FLOAT32
        },
    );

    #[allow(unreachable_code)]
    (Device::CPU, ComputeType::FLOAT32) // safe default for other arches
}

/// A loaded CTranslate2 model with its SentencePiece tokenizer.
/// The tokenizer auto-loads `source.spm` and `target.spm` from `model_dir`.
pub struct LoadedModel {
    sys_translator: SysTranslator,
    tokenizer: SpmTokenizer,
}

impl LoadedModel {
    /// Load a model directory containing `model.bin`, `source.spm`, and `target.spm`.
    pub fn load(model_dir: &Path, num_threads: usize) -> Result<Self, TranslatorError> {
        let tokenizer = SpmTokenizer::new(model_dir)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        let (device, compute_type) = select_device_and_compute();
        let config = Config {
            compute_type,
            device,
            num_threads_per_replica: num_threads,
            ..Config::default()
        };
        let sys_translator = SysTranslator::new(model_dir, &config)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        Ok(Self {
            sys_translator,
            tokenizer,
        })
    }

    fn run_batch(&self, tokenized: Vec<Vec<String>>, beam_size: usize) -> Result<Vec<String>, TranslatorError> {
        let options = TranslationOptions::<String, String> {
            beam_size,
            no_repeat_ngram_size: 3,
            replace_unknowns: true,
            max_input_length: 1024,   // MADLAD-400 supports 1024; was 512 for opus-mt
            max_decoding_length: 1024, // match input — MADLAD supports 1024 on both sides
            max_batch_size: 4096,
            batch_type: BatchType::Tokens,
            ..Default::default()
        };
        let results = self
            .sys_translator
            .translate_batch(&tokenized, &options, None)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        results
            .into_iter()
            .map(|r| {
                let tokens = r
                    .hypotheses
                    .into_iter()
                    .next()
                    .ok_or_else(|| TranslatorError::Ct2("no hypothesis returned".to_string()))?;
                self.tokenizer
                    .decode(tokens)
                    .map_err(|e| TranslatorError::Ct2(e.to_string()))
            })
            .collect()
    }

    /// Translate a batch of strings. Synchronous — always call from `spawn_blocking`.
    pub fn translate_batch(&self, texts: &[String], beam_size: usize) -> Result<Vec<String>, TranslatorError> {
        let tokenized = texts
            .iter()
            .map(|t| self.tokenizer.encode(t))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        self.run_batch(tokenized, beam_size)
    }

}
