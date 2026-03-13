use std::convert::Infallible;

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use tokio_stream::wrappers::ReceiverStream;
use translator_core::{
    Language,
    error::TranslatorError,
    types::{TranslationBatch, TranslationRequest, TranslationResultSet},
};

use crate::{error::ApiError, state::AppState};

/// Expand `"all"` sentinel and parse string codes into typed `Language` values.
fn resolve_batch(req: TranslationRequest) -> Result<TranslationBatch, ApiError> {
    let target_languages = if req.target_languages == ["all"] {
        Language::all().to_vec()
    } else {
        req.target_languages
            .iter()
            .map(|s| s.parse::<Language>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::from)?
    };

    let source_language = req
        .source_language
        .as_deref()
        .map(|s| s.parse::<Language>())
        .transpose()
        .map_err(ApiError::from)?;

    Ok(TranslationBatch {
        texts: req.texts,
        target_languages,
        source_language,
    })
}

/// Validate batch size limits. Returns an error if the batch exceeds configured limits.
pub fn validate_request(req: &TranslationRequest, state: &AppState) -> Result<(), ApiError> {
    if req.texts.is_empty() {
        return Err(ApiError(TranslatorError::UnsupportedLanguage(
            "texts cannot be empty".to_string(),
        )));
    }

    if req.texts.len() > state.max_texts_per_request {
        return Err(ApiError(TranslatorError::InputTooLong(format!(
            "too many texts: {} (max {})",
            req.texts.len(),
            state.max_texts_per_request
        ))));
    }

    let n_langs = if req.target_languages == ["all"] {
        Language::all().len()
    } else {
        req.target_languages.len()
    };
    let work_items = req.texts.len() * n_langs;
    if work_items > state.max_work_items_per_request {
        return Err(ApiError(TranslatorError::InputTooLong(format!(
            "too many work items: {} texts × {} languages = {} (max {})",
            req.texts.len(),
            n_langs,
            work_items,
            state.max_work_items_per_request
        ))));
    }

    Ok(())
}

pub async fn translate(
    State(state): State<AppState>,
    Json(req): Json<TranslationRequest>,
) -> Result<Json<TranslationResultSet>, ApiError> {
    validate_request(&req, &state)?;
    let batch = resolve_batch(req)?;

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

/// SSE streaming endpoint: translates each text independently and streams
/// results as they complete. Each text produces one `translation` event.
/// A final `done` event signals completion.
pub async fn translate_stream(
    State(state): State<AppState>,
    Json(req): Json<TranslationRequest>,
) -> Result<Response, ApiError> {
    validate_request(&req, &state)?;

    let target_languages: Vec<Language> = if req.target_languages == ["all"] {
        Language::all().to_vec()
    } else {
        req.target_languages
            .iter()
            .map(|s| s.parse::<Language>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::from)?
    };
    let source_language = req
        .source_language
        .as_deref()
        .map(|s| s.parse::<Language>())
        .transpose()
        .map_err(ApiError::from)?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(req.texts.len() + 1);
    let engine = state.engine.clone();
    let texts = req.texts;

    tokio::spawn(async move {
        for text in texts {
            let sub_batch = TranslationBatch {
                texts: vec![text],
                target_languages: target_languages.clone(),
                source_language,
            };
            let engine = engine.clone();
            let result =
                tokio::task::spawn_blocking(move || engine.translate_batch_chunked(sub_batch))
                    .await;

            let event = match result {
                Ok(Ok(result_set)) => {
                    match serde_json::to_string(&result_set.results[0]) {
                        Ok(json) => Event::default().event("translation").data(json),
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize translation result");
                            break;
                        }
                    }
                }
                Ok(Err(e)) => {
                    tracing::error!(error = %e, "translation failed in stream");
                    let err_json = serde_json::json!({ "error": e.to_string() });
                    Event::default()
                        .event("error")
                        .data(err_json.to_string())
                }
                Err(e) => {
                    tracing::error!(panic = %e, "translator-scheduler panicked in stream");
                    break;
                }
            };

            // If send fails, the client disconnected — exit cleanly.
            if tx.send(Ok(event)).await.is_err() {
                return;
            }
        }

        let done = Event::default().event("done").data("[DONE]");
        let _ = tx.send(Ok(done)).await;
    });

    let stream = ReceiverStream::new(rx);
    Ok(IntoResponse::into_response(Sse::new(stream).keep_alive(KeepAlive::default())))
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
