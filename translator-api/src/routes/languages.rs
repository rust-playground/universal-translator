use axum::extract::Query;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use translator_core::detector::detect_supported_codes;
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

/// Translate-side response — typed `Vec<Language>` serializes as an array of
/// BCP 47 codes (`Language` serializes as its `.code()`). Clients can
/// deserialize each entry back into `Language` and use `.full_name()` for the
/// English label without a wire round-trip.
#[derive(Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<Language>,
}

/// Detect-side response — broader than the translate enum (includes
/// lingua-only codes like `cy`, `eu`, `ka`). Codes are strings since not all
/// of them map to `Language`; parse with `Language::from_str` if you need
/// the enum form, and check for `None` for translate-unsupported codes.
#[derive(Serialize)]
pub struct DetectLanguagesResponse {
    pub languages: Vec<&'static str>,
}

/// `GET /languages` — defaults to `?for=translate`.
///
/// `?for=detect` returns the broader detect-side coverage (lingua + script +
/// heuristic refinements). Codes from the detect list may not round-trip into
/// translation; check against `?for=translate` to know what translate accepts.
pub async fn languages(Query(query): Query<LanguagesQuery>) -> Response {
    match query.r#for {
        For::Translate => Json(LanguagesResponse {
            languages: Language::all().to_vec(),
        })
        .into_response(),
        For::Detect => Json(DetectLanguagesResponse {
            languages: detect_supported_codes()
                .iter()
                .map(|(code, _name)| *code)
                .collect(),
        })
        .into_response(),
    }
}
