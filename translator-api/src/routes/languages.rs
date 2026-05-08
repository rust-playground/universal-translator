use axum::extract::Query;
use axum::Json;
use serde::{Deserialize, Serialize};
use translator_core::detector::detect_supported_codes;
use translator_core::types::LanguageEntry;
use translator_core::Language;

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum For {
    #[default]
    Translate,
    Detect,
}

#[derive(Default, Deserialize)]
pub struct LanguagesQuery {
    #[serde(default, rename = "for")]
    pub r#for: For,
}

#[derive(Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<LanguageEntry>,
}

/// `GET /languages` — defaults to `?for=translate`.
///
/// `?for=detect` returns the broader detect-side coverage (lingua + script +
/// heuristic refinements). Codes from the detect list may not round-trip into
/// translation; check against `?for=translate` to know what translate accepts.
pub async fn languages(Query(query): Query<LanguagesQuery>) -> Json<LanguagesResponse> {
    let languages = match query.r#for {
        For::Translate => Language::all()
            .iter()
            .map(|lang| LanguageEntry {
                code: lang.code().to_string(),
                name: lang.full_name().to_string(),
            })
            .collect(),
        For::Detect => detect_supported_codes()
            .iter()
            .map(|(code, name)| LanguageEntry {
                code: code.to_string(),
                name: name.to_string(),
            })
            .collect(),
    };
    Json(LanguagesResponse { languages })
}
