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

    let result = state
        .engine
        .translate_batch(batch)
        .await
        .map_err(|e| {
            #[cfg(feature = "opentelemetry")]
            {
                use opentelemetry::KeyValue;
                let kind = match &e {
                    TranslatorError::ModelNotFound(_) => "model_not_found",
                    TranslatorError::DetectionFailed(_) => "detection_failed",
                    TranslatorError::UnsupportedLanguage(_) => "unsupported_language",
                    TranslatorError::TranslationFailed(_) => "translation_failed",
                    TranslatorError::Io(_) => "io",
                    TranslatorError::Model(_) => "model",
                };
                state.error_ctr.add(1, &[KeyValue::new("error_type", kind)]);
            }
            ApiError(e)
        })?;

    Ok(Json(result))
}
