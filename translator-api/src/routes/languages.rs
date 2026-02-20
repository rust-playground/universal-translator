use axum::Json;
use serde::Serialize;
use translator_core::engine::supported_target_languages;

#[derive(Serialize)]
pub struct LanguagesResponse {
    pub languages: &'static [&'static str],
}

pub async fn languages() -> Json<LanguagesResponse> {
    Json(LanguagesResponse {
        languages: supported_target_languages(),
    })
}
