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

    /// Returns `(iso_code, language_name, confidence)`.
    ///
    /// **Confidence semantics:** `top / (top + second)` — the fraction of probability
    /// mass assigned to the top two Lingua candidates that belongs to the winner.
    /// This is a *relative* score: it answers "how clearly does the top language beat
    /// its nearest competitor?" rather than "how certain are we in absolute terms?"
    ///
    /// Because Lingua's raw scores sum to 1.0 across ~75 languages, even an obvious
    /// detection like English "Hello, how are you?" would score only ~17% in absolute
    /// terms. The relative formula gives ~73% for that same input, and 95%+ for longer
    /// or script-distinctive text.
    ///
    /// Practical interpretation:
    /// - > 0.90  — strong, unambiguous signal
    /// - 0.70–0.90 — confident detection; short or common phrases may land here
    /// - 0.50–0.70 — moderate; treat as a best guess
    /// - < 0.50  — weak; text is very short or genuinely ambiguous
    ///
    /// Script fallback (Malayalam): confidence = 1.0 (unambiguous Unicode-range match).
    pub fn detect_with_confidence(&self, text: &str) -> Result<(String, String, f64), TranslatorError> {
        let values = self.inner.compute_language_confidence_values(text.to_string());
        let mut iter = values.into_iter();
        if let Some((lang, top)) = iter.next() {
            let second = iter.next().map(|(_, s)| s).unwrap_or(0.0);
            let confidence = if top + second > 0.0 { top / (top + second) } else { 1.0 };
            let code = language_to_iso639_1(&lang);
            let name = format!("{lang:?}");
            return Ok((code, name, confidence));
        }
        // Lingua returned empty — try script fallback.
        if let Some(code) = detect_script(text) {
            return Ok((code.to_string(), "Malayalam".to_string(), 1.0));
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
