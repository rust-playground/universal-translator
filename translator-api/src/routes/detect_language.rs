use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use translator_core::types::LanguageDetectionResult;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct DetectLanguageRequest {
    pub text: String,
}

/// POST /detect-language
///
/// Detects the language of the supplied text.
///
/// **`confidence` semantics:** relative score `top / (top + second)`, where `top`
/// and `second` are Lingua's raw probability scores for the first- and second-ranked
/// candidate languages. This reflects how clearly the top language beats its nearest
/// competitor, not an absolute probability. Short common phrases (e.g. "Hello, how
/// are you?") may score ~70–80% even when correctly identified; longer or
/// script-distinctive text typically scores 95%+.
pub async fn detect_language(
    State(state): State<AppState>,
    Json(req): Json<DetectLanguageRequest>,
) -> Result<Json<LanguageDetectionResult>, ApiError> {
    let engine = state.engine.clone();
    let text = req.text.clone();
    let result = tokio::task::spawn_blocking(move || engine.detect_language_full(&text))
        .await
        .map_err(|_| ApiError(translator_core::error::TranslatorError::TranslationFailed("detect panicked".into())))??;
    Ok(Json(result))
}
