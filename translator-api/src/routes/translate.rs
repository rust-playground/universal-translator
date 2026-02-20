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

    let result = state.engine.translate_batch(batch).await?;
    Ok(Json(result))
}
