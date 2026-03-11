use axum::{extract::State, Json};
use unicode_segmentation::UnicodeSegmentation;
use translator_core::{
    error::TranslatorError,
    types::{TranslationBatch, TranslationResultSet},
};

use crate::{error::ApiError, state::AppState};

/// Split text at sentence boundaries if it exceeds `max_chars`.
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for sentence in text.unicode_sentences() {
        if !current.is_empty() && current.len() + sentence.len() > max_chars {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(sentence);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    // Fallback: if unicode_sentences produced nothing useful (e.g. no sentence
    // boundaries detected), return the original text as a single chunk so the
    // scheduler's InputTooLong guard can handle it gracefully.
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

pub async fn translate(
    State(state): State<AppState>,
    Json(batch): Json<TranslationBatch>,
) -> Result<Json<TranslationResultSet>, ApiError> {
    if batch.texts.is_empty() {
        return Err(ApiError(TranslatorError::UnsupportedLanguage(
            "texts cannot be empty".to_string(),
        )));
    }

    // Chunking threshold: leave ~100 tokens for prompt template overhead,
    // multiply by ~4 chars/token to get a character limit.
    let max_chars = (state.engine.kv_budget_per_slot().saturating_sub(100) * 4) as usize;

    // Check if any text needs chunking.
    let needs_chunking = batch.texts.iter().any(|t| t.len() > max_chars);

    if !needs_chunking {
        // Fast path: no chunking needed.
        let engine = state.engine.clone();
        let result = tokio::task::spawn_blocking(move || engine.translate_batch(batch))
            .await
            .map_err(|e| {
                tracing::error!(panic = %e, "translator-scheduler panicked");
                ApiError(TranslatorError::TranslationFailed("scheduler panicked".into()))
            })?
            .map_err(|e| {
                record_error(&state, &e);
                ApiError(e)
            })?;
        return Ok(Json(result));
    }

    // Slow path: chunk long texts, translate, reassemble.
    // Build a mapping from original text index → chunk indices.
    let mut chunked_texts: Vec<String> = Vec::new();
    // (original_idx, start_chunk_idx, chunk_count)
    let mut chunk_map: Vec<(usize, usize, usize)> = Vec::new();

    for (i, text) in batch.texts.iter().enumerate() {
        let chunks = chunk_text(text, max_chars);
        let start = chunked_texts.len();
        let count = chunks.len();
        chunked_texts.extend(chunks);
        chunk_map.push((i, start, count));
    }

    let chunked_batch = TranslationBatch {
        texts: chunked_texts,
        target_languages: batch.target_languages.clone(),
        source_language: batch.source_language.clone(),
    };

    let engine = state.engine.clone();
    let chunked_result = tokio::task::spawn_blocking(move || engine.translate_batch(chunked_batch))
        .await
        .map_err(|e| {
            tracing::error!(panic = %e, "translator-scheduler panicked");
            ApiError(TranslatorError::TranslationFailed("scheduler panicked".into()))
        })?
        .map_err(|e| {
            record_error(&state, &e);
            ApiError(e)
        })?;

    // Reassemble: concatenate chunk translations per language for each original text.
    let mut results = Vec::with_capacity(chunk_map.len());
    for (orig_idx, start, count) in &chunk_map {
        if *count == 1 {
            // No chunking happened for this text — use result directly.
            results.push(chunked_result.results[*start].clone());
            // Restore original source text.
            results.last_mut().unwrap().source_text = batch.texts[*orig_idx].clone();
        } else {
            let first = &chunked_result.results[*start];
            let mut merged_translations = first.translations.clone();
            let mut merged_errors = first.errors.clone();

            for chunk_idx in (*start + 1)..(*start + *count) {
                let chunk_result = &chunked_result.results[chunk_idx];
                for (lang, translation) in &chunk_result.translations {
                    merged_translations
                        .entry(lang.clone())
                        .and_modify(|existing| existing.push_str(translation))
                        .or_insert_with(|| translation.clone());
                }
                for (lang, err) in &chunk_result.errors {
                    merged_errors.entry(lang.clone()).or_insert_with(|| err.clone());
                }
            }

            results.push(translator_core::types::TranslationResult {
                source_text: batch.texts[*orig_idx].clone(),
                detected_language: first.detected_language.clone(),
                translations: merged_translations,
                errors: merged_errors,
            });
        }
    }

    Ok(Json(TranslationResultSet { results }))
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
