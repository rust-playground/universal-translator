use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::language::Language;

/// JSON deserialization target — accepts raw strings including the `"all"` sentinel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub texts: Vec<String>,
    /// ISO 639-1 codes, e.g. `["fr", "de"]`, or `["all"]`.
    pub target_languages: Vec<String>,
    /// ISO 639-1 code for all texts in this batch. When set, skips auto-detection.
    pub source_language: Option<String>,
}

/// Engine-internal batch — fully typed, no `"all"` sentinel.
#[derive(Debug)]
pub struct TranslationBatch {
    pub texts: Vec<String>,
    pub target_languages: Vec<Language>,
    /// Parsed at the API/CLI boundary; invalid codes rejected early.
    pub source_language: Option<Language>,
}

/// Translation result for a single source text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub source_text: String,
    pub detected_language: String,
    /// `"fr"` → `"Bonjour"`
    pub translations: HashMap<String, String>,
    /// Per-language errors; omitted from JSON when empty.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub errors: HashMap<String, String>,
}

/// Top-level batch response — one result per input text.
#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationResultSet {
    pub results: Vec<TranslationResult>,
}

/// Result of a standalone language detection request.
#[derive(Debug, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    pub language_code: String,
    pub language: String,
    pub confidence: f64,
    pub translation_supported: bool,
}
