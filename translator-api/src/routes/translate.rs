use axum::{extract::State, Json};
use translator_core::{
    error::TranslatorError,
    types::{TranslationBatch, TranslationResultSet},
};

use crate::{error::ApiError, state::AppState};

pub async fn translate(
    State(state): State<AppState>,
    Json(batch): Json<TranslationBatch>,
) -> Result<Json<TranslationResultSet>, ApiError> {
    if batch.texts.is_empty() {
        return Err(ApiError(TranslatorError::UnsupportedLanguage(
            "texts cannot be empty".to_string(),
        )));
    }

    let engine = state.engine.clone();
    let result = tokio::task::spawn_blocking(move || engine.translate_batch_chunked(batch))
        .await
        .map_err(|e| {
            tracing::error!(panic = %e, "translator-scheduler panicked");
            ApiError(TranslatorError::TranslationFailed("scheduler panicked".into()))
        })?
        .map_err(|e| {
            record_error(&state, &e);
            ApiError(e)
        })?;

    Ok(Json(result))
}

#[cfg_attr(not(feature = "opentelemetry"), allow(unused_variables))]
fn record_error(state: &AppState, e: &TranslatorError) {
    #[cfg(feature = "opentelemetry")]
    {
        use opentelemetry::KeyValue;
        let kind = match e {
            TranslatorError::ModelNotFound(_) => "model_not_found",
            TranslatorError::DetectionFailed(_) => "detection_failed",
            TranslatorError::UnsupportedLanguage(_) => "unsupported_language",
            TranslatorError::TranslationFailed(_) => "translation_failed",
            TranslatorError::Io(_) => "io",
            TranslatorError::Model(_) => "model",
            TranslatorError::ServiceUnavailable(_) => "service_unavailable",
            TranslatorError::InputTooLong(_) => "input_too_long",
        };
        state.error_ctr.add(1, &[KeyValue::new("error_type", kind)]);
    }
}
