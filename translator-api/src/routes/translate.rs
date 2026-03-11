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

#[cfg(test)]
mod tests {
    use super::chunk_text;

    #[test]
    fn short_text_no_chunking() {
        let result = chunk_text("Hello world.", 100);
        assert_eq!(result, vec!["Hello world."]);
    }

    #[test]
    fn exact_boundary() {
        let text = "Hello world.";
        let result = chunk_text(text, text.len());
        assert_eq!(result, vec!["Hello world."]);
    }

    #[test]
    fn two_sentences_split() {
        let text = "First sentence. Second sentence.";
        // max_chars just big enough for the first sentence but not both
        let result = chunk_text(text, 16);
        assert_eq!(result.len(), 2);
        assert!(result[0].contains("First"));
        assert!(result[1].contains("Second"));
    }

    #[test]
    fn single_long_sentence_no_boundary() {
        // One sentence with no internal sentence boundaries — greedy push keeps it as one chunk
        let text = "This is one very long sentence without any terminating punctuation that goes on and on";
        let result = chunk_text(text, 20);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], text);
    }

    #[test]
    fn multiple_sentences_greedy_packing() {
        let text = "One. Two. Three. Four. Five. Six.";
        // Each sentence is ~5 chars; set max_chars so ~2-3 sentences fit per chunk
        let result = chunk_text(text, 12);
        // Should produce fewer chunks than sentences (greedy packing)
        assert!(result.len() < 6, "expected greedy packing, got {} chunks", result.len());
        // Reassembled text should equal original
        let reassembled: String = result.concat();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn empty_string() {
        let result = chunk_text("", 100);
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn unicode_sentences() {
        // Japanese text with sentence-ending periods (。)
        let text = "これは文です。もう一つの文です。";
        // Small enough to force a split if boundaries are detected
        let result = chunk_text(text, 24);
        // Should produce at least 2 chunks at the 。 boundary
        assert!(result.len() >= 2, "expected split at Japanese sentence boundary, got {} chunks", result.len());
        let reassembled: String = result.concat();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn labels_without_sentence_boundaries() {
        // Short field labels — colons must not cause splitting
        let r = chunk_text("Title:", 10);
        assert_eq!(r, vec!["Title:"]);

        let r = chunk_text("First Name:", 20);
        assert_eq!(r, vec!["First Name:"]);

        let r = chunk_text("Last Name:", 20);
        assert_eq!(r, vec!["Last Name:"]);
    }

    #[test]
    fn no_sentence_boundaries() {
        // Long text with no sentence-ending punctuation
        let text = "word ".repeat(50);
        let text = text.trim();
        let result = chunk_text(text, 20);
        // unicode_sentences finds nothing useful → fallback returns original
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], text);
    }
}
