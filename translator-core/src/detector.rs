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
        // Lingua first — covers 75 languages including all but 1 of our supported set.
        if let Some(lang) = self.inner.detect_language_of(text) {
            return Ok(language_to_iso639_1(&lang));
        }
        // Lingua returned None. Fall back to script-based detection for languages
        // outside lingua's coverage (currently: Malayalam).
        if let Some(code) = detect_script(text) {
            return Ok(code.to_string());
        }
        Err(TranslatorError::DetectionFailed(format!(
            "Could not detect language for text: {text:?}"
        )))
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

/// Script-based language detection fallback for languages not covered by lingua.
///
/// Malayalam uses a unique Unicode block (U+0D00–U+0D7F). No other language uses
/// these codepoints, so a single character scan is unambiguous.
fn detect_script(text: &str) -> Option<&'static str> {
    if text.chars().any(|c| ('\u{0D00}'..='\u{0D7F}').contains(&c)) {
        return Some("ml");
    }
    None
}

fn language_to_iso639_1(language: &Language) -> String {
    language.iso_code_639_1().to_string().to_lowercase()
}
