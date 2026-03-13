use axum::Json;
use serde::Serialize;
use translator_core::Language;

#[derive(Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<Language>,
}

pub async fn languages() -> Json<LanguagesResponse> {
    Json(LanguagesResponse {
        languages: Language::all().to_vec(),
    })
}
