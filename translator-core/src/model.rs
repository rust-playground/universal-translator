use std::path::Path;

use ct2rs::tokenizers::sentencepiece::Tokenizer as SpmTokenizer;
use ct2rs::{ComputeType, Config, TranslationOptions, Translator};

use crate::error::TranslatorError;

/// A loaded CTranslate2 model with its SentencePiece tokenizer.
/// The tokenizer auto-loads `source.spm` and `target.spm` from `model_dir`.
pub struct LoadedModel {
    translator: Translator<SpmTokenizer>,
}

impl LoadedModel {
    /// Load a model directory containing `model.bin`, `source.spm`, and `target.spm`.
    pub fn load(model_dir: &Path) -> Result<Self, TranslatorError> {
        let tokenizer = SpmTokenizer::new(model_dir)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        let config = Config {
            compute_type: ComputeType::INT8_FLOAT32,
            ..Config::default()
        };
        let translator = Translator::with_tokenizer(model_dir, tokenizer, &config)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        Ok(Self { translator })
    }

    /// Translate a batch of strings. Synchronous — always call from `spawn_blocking`.
    pub fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        let options = TranslationOptions::<String, String> {
            beam_size: 4,
            replace_unknowns: true,
            ..Default::default()
        };
        let results = self
            .translator
            .translate_batch(texts, &options, None)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;

        Ok(results.into_iter().map(|(text, _score)| text).collect())
    }

    /// Translate using a mandatory target-language prefix token (e.g. `">>tha<<"` for Thai,
    /// `">>jpn<<"` for Japanese). The prefix is stripped from output automatically by ct2rs.
    pub fn translate_batch_with_prefix(
        &self,
        texts: &[String],
        prefix_token: &str,
    ) -> Result<Vec<String>, TranslatorError> {
        let prefixes: Vec<Vec<String>> = texts
            .iter()
            .map(|_| vec![prefix_token.to_string()])
            .collect();
        let options = TranslationOptions::<String, String> {
            beam_size: 4,
            replace_unknowns: true,
            ..Default::default()
        };
        let results = self
            .translator
            .translate_batch_with_target_prefix(texts, &prefixes, &options, None)
            .map_err(|e| TranslatorError::Ct2(e.to_string()))?;
        Ok(results.into_iter().map(|(text, _score)| text).collect())
    }
}
