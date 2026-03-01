use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Batch request: many texts, one shared set of target languages.
#[derive(Debug, Deserialize)]
pub struct TranslationBatch {
    pub texts: Vec<String>,
    /// ISO 639-1 codes, e.g. ["fr", "de"]
    pub target_languages: Vec<String>,
    /// ISO 639-1 code for all texts in this batch. When set, skips auto-detection.
    pub source_language: Option<String>,
}

/// Translation result for a single source text.
#[derive(Debug, Serialize)]
pub struct TranslationResult {
    pub source_text: String,
    pub detected_language: String,
    /// "fr" → "Bonjour"
    pub translations: HashMap<String, String>,
    /// Per-language errors; omitted from JSON when empty.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub errors: HashMap<String, String>,
}

/// Top-level batch response — one result per input text.
#[derive(Debug, Serialize)]
pub struct TranslationResultSet {
    pub results: Vec<TranslationResult>,
}

/// Result of a standalone language detection request.
#[derive(Debug, Serialize)]
pub struct LanguageDetectionResult {
    pub language_code: String,
    pub language: String,
    pub confidence: f64,
    pub translation_supported: bool,
}
