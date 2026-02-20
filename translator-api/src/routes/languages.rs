use axum::Json;
use serde::Serialize;
use translator_core::engine::supported_languages;

#[derive(Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<&'static str>,
}

pub async fn languages() -> Json<LanguagesResponse> {
    Json(LanguagesResponse {
        languages: supported_languages(),
    })
}
