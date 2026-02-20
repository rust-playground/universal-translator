use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};

use crate::error::TranslatorError;

pub struct Detector {
    inner: LanguageDetector,
}

impl Detector {
    pub fn new() -> Self {
        let mut builder = LanguageDetectorBuilder::from_all_spoken_languages();
        builder.with_preloaded_language_models();
        let inner = builder.build();
        Self { inner }
    }

    /// Returns a lowercase ISO 639-1 code, e.g. `"en"`.
    pub fn detect(&self, text: &str) -> Result<String, TranslatorError> {
        self.inner
            .detect_language_of(text)
            .map(|lang| language_to_iso639_1(&lang))
            .ok_or_else(|| {
                TranslatorError::DetectionFailed(format!(
                    "Could not detect language for text: {text:?}"
                ))
            })
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

fn language_to_iso639_1(language: &Language) -> String {
    language.iso_code_639_1().to_string().to_lowercase()
}
