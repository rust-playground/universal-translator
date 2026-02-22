use std::path::Path;

use ct2rs::sys::Translator as SysTranslator;
use ct2rs::tokenizers::sentencepiece::Tokenizer as SpmTokenizer;
use ct2rs::{BatchType, ComputeType, Config, Tokenizer, TranslationOptions};

use crate::error::TranslatorError;

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
        #[cfg(target_arch = "aarch64")]
        let compute_type = ComputeType::FLOAT32; // ARM NEON FLOAT32 > INT8 for small-batch inference

        #[cfg(target_arch = "x86_64")]
        let compute_type = if std::is_x86_feature_detected!("avx2") {
            ComputeType::INT8 // AVX2 VNNI makes INT8 2-3× faster; CTranslate2 quantizes at load time
        } else {
            ComputeType::FLOAT32
        };

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let compute_type = ComputeType::FLOAT32; // safe default for other arches

        let config = Config {
            compute_type,
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

    fn run_batch(&self, tokenized: Vec<Vec<String>>) -> Result<Vec<String>, TranslatorError> {
        let options = TranslationOptions::<String, String> {
            beam_size: 4,
            no_repeat_ngram_size: 3,
            replace_unknowns: true,
            max_input_length: 512,
            max_decoding_length: 512,
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
    pub fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        let tokenized = texts
            .iter()
            .map(|t| self.tokenizer.encode(t))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        self.run_batch(tokenized)
    }

    /// Translate a batch where each text has its own prefix token.
    /// Used by the engine to combine all en-mul target languages into one inference call.
    pub fn translate_batch_with_per_text_prefix(
        &self,
        texts: &[String],
        prefixes: &[&str],
    ) -> Result<Vec<String>, TranslatorError> {
        debug_assert_eq!(texts.len(), prefixes.len());
        let tokenized = texts.iter().zip(prefixes)
            .map(|(t, &prefix)| {
                self.tokenizer.encode(t)
                    .map_err(|e| TranslatorError::Ct2(e.to_string()))
                    .map(|mut tokens| { tokens.insert(0, prefix.to_string()); tokens })
            })
            .collect::<Result<Vec<_>, TranslatorError>>()?;
        self.run_batch(tokenized)
    }

    /// Translate using a source-side language prefix token (e.g. `">>jpn<<"` for Japanese).
    /// Helsinki-NLP opus-mt multilingual models expect the token prepended to the encoder input,
    /// not used as a decoder target prefix. The prefix is inserted directly as a token piece
    /// (bypassing re-tokenization) so SentencePiece's character-level fallback cannot split it.
    pub fn translate_batch_with_prefix(
        &self,
        texts: &[String],
        prefix_token: &str,
    ) -> Result<Vec<String>, TranslatorError> {
        let tokenized = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(t)
                    .map_err(|e| TranslatorError::Ct2(e.to_string()))
                    .map(|mut tokens| {
                        tokens.insert(0, prefix_token.to_string());
                        tokens
                    })
            })
            .collect::<Result<Vec<_>, TranslatorError>>()?;
        self.run_batch(tokenized)
    }
}
