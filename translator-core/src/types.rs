use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::TranslationItemError;
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
    pub detected_language: Option<Language>,
    pub translations: HashMap<Language, String>,
    /// Per-language errors; omitted from JSON when empty.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub errors: HashMap<Language, TranslationItemError>,
}

/// Top-level batch response — one result per input text.
#[derive(Debug, Serialize, Deserialize)]
pub struct TranslationResultSet {
    pub results: Vec<TranslationResult>,
}

/// Result of a standalone language detection request.
///
/// Two language fields are present because the detector's output universe
/// is broader than the translate-side `Language` enum:
///
/// - `language` is the **raw BCP 47 code** the detector identified. May
///   include script subtags (`zh-CN`, `sr-Cyrl`), heuristic regional tags
///   (`pt-BR`, `en-US`), or lingua-only codes outside the translate set
///   (`cy` Welsh, `ka` Georgian, `nb` Norwegian Bokmål, `tl` Tagalog).
///   This is the most precise signal we have about the input.
///
/// - `translate_language` is the **translate-side equivalent** — the same
///   code parsed into the `Language` enum, applying the standard
///   normalization aliases (`nb`/`nn` → `No`, `tl` → `Fil`, `iw` → `He`,
///   `zh-Hans` → `zh_CN`, etc.). `None` for lingua-only languages the
///   engine can't translate from. Use this when you need a value
///   `/translate` accepts as `source_language` / `target_languages`.
///
/// In most cases the two are equal. They differ when:
/// 1. The detector returns an alias of a translate-supported code
///    (e.g. `language = "nb"` → `translate_language = Some(No)` because
///    Bokmål and Nynorsk both collapse to the macrolanguage in our enum,
///    or `language = "tl"` → `translate_language = Some(Fil)`).
/// 2. The detector returns a script-subtag form whose region-form
///    counterpart is in the enum (e.g. `language = "zh-Hant"` →
///    `translate_language = Some(zh_TW)`). After Chinese unification the
///    detector emits the region form directly, so this case is rarer.
/// 3. The detector returns a lingua-only language; `translate_language`
///    is `None`.
///
/// `translate_language.is_some()` indicates whether `/translate` will
/// accept the result as a `source_language`.
#[derive(Debug, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    /// Raw BCP 47 code from the detector. May include script/region
    /// refinements (`zh-CN`, `pt-BR`, `sr-Cyrl`) and may sit outside the
    /// translate-side `Language` enum (`cy`, `nb`, `tl`).
    pub language: String,

    /// Translate-side enum equivalent of `language`. `Some` when `language`
    /// parses (directly or via FromStr alias) into the `Language` enum;
    /// `None` for lingua-only codes the engine can't translate from.
    pub translate_language: Option<Language>,

    /// Relative confidence in `[0, 1]` — see `Detector::detect_with_confidence`
    /// for semantics.
    pub confidence: f64,
}
