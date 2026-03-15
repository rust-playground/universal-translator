use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use crate::detector::Detector;
use crate::error::{TranslationItemError, TranslatorError};
use crate::language::{self, Language};
use crate::model::LoadedGemmaModel;
use crate::scheduler::{ContinuousScheduler, InferRequest, SLOT_CAPACITY};
use crate::types::{LanguageDetectionResult, TranslationBatch, TranslationResult, TranslationResultSet};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the translation engine.
///
/// All fields are optional — sensible defaults are used when `None`.
/// Env-var reading belongs in the CLI/API layer (via clap `env = "..."`),
/// not here.
pub struct EngineConfig {
    pub model_path: PathBuf,
    /// Number of concurrent decode slots.
    pub n_slots: Option<usize>,
    /// Maximum tokens per translation (maps to KV budget per slot).
    pub max_tokens: Option<u32>,
    /// Bounded queue capacity for pending translation requests.
    pub queue_capacity: Option<usize>,
    /// Prefill accumulation delay in milliseconds.
    pub prefill_delay_ms: Option<u64>,
    /// Hard ceiling (in characters) for text chunks sent to the model.
    /// Defaults to `(kv_budget_per_slot - 100) * 4`.
    pub max_chunk_chars: Option<usize>,
    /// Target size (in characters) for paragraph-level chunk packing.
    /// Shorter chunks improve translation quality. Defaults to ~60% of `max_chunk_chars`.
    pub paragraph_target_chars: Option<usize>,
    /// Timeout (in seconds) when sending work items to the scheduler queue.
    /// Defaults to 30. Allows requests to wait for capacity instead of failing instantly.
    pub queue_send_timeout_secs: Option<u64>,
}

// ── Prompt builder ───────────────────────────────────────────────────────────

/// Build a full Gemma instruct-format translation prompt.
fn translate_gemma_prompt(src_lang: Language, tgt_lang: Language, text: &str) -> String {
    format!(
        "<bos><start_of_turn>system\n\
         You are a translation engine. Output only the translated text. \
         Do not add explanations, alternatives, notes, or any other text.<end_of_turn>\n\
         <start_of_turn>user\n\
         Translate from {} to {}:\n{}<end_of_turn>\n\
         <start_of_turn>model\n",
        src_lang.full_name(),
        tgt_lang.full_name(),
        text
    )
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// Joins the scheduler thread on drop, ensuring LlamaContext/LlamaModel are
/// fully freed before process exit runs Metal/CUDA static destructors.
struct SchedulerGuard(Option<std::thread::JoinHandle<()>>);

impl Drop for SchedulerGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}

/// The central translation engine. Cheap to clone — all heavy state is reference-counted.
///
/// **Drop order matters:** `worker_tx` is declared before `_scheduler_guard` so
/// that when the last clone drops, the channel closes first (signaling the
/// scheduler to exit), then the `Arc<SchedulerGuard>` refcount hits zero and
/// joins the thread — ensuring full cleanup before static destructors run.
#[derive(Clone)]
pub struct TranslationEngine {
    worker_tx: crossbeam_channel::Sender<InferRequest>,
    _scheduler_guard: Arc<SchedulerGuard>,
    detector: Arc<Detector>,
    /// Bounded channel capacity.
    queue_capacity: usize,
    /// Timeout for sending work items to the queue.
    queue_send_timeout: std::time::Duration,
    /// Resolved KV budget per slot (tokens).
    kv_budget_per_slot: u32,
    /// Hard ceiling (in characters) for text chunks.
    max_chunk_chars: usize,
    /// Paragraph-level packing target (in characters).
    paragraph_target_chars: usize,
    #[cfg(feature = "opentelemetry")]
    requests: opentelemetry::metrics::Counter<u64>,
    #[cfg(feature = "opentelemetry")]
    batch_size: opentelemetry::metrics::Histogram<u64>,
    #[cfg(feature = "opentelemetry")]
    duration_ms: opentelemetry::metrics::Histogram<f64>,
}

/// Default slot counts per backend.
/// llama.cpp manages KV cache memory internally — no need for GPU memory queries.
#[cfg(feature = "metal")]
const DEFAULT_N_SLOTS_METAL: usize = 32;
#[cfg(feature = "cuda")]
const DEFAULT_N_SLOTS_CUDA: usize = 64;
const DEFAULT_N_SLOTS_CPU: usize = 4;

/// Compile-time default slot count per backend.
fn auto_n_slots() -> usize {
    #[cfg(feature = "metal")]
    {
        return DEFAULT_N_SLOTS_METAL;
    }

    #[cfg(feature = "cuda")]
    {
        return DEFAULT_N_SLOTS_CUDA;
    }

    #[allow(unreachable_code)]
    DEFAULT_N_SLOTS_CPU
}

/// Default KV budget per slot (tokens).
pub(crate) const DEFAULT_KV_BUDGET_PER_SLOT: u32 = 1024;

/// Default prefill accumulation delay (ms).
pub(crate) const DEFAULT_PREFILL_ACCUMULATION_MS: u64 = 10;

/// Default queue send timeout (seconds).
pub(crate) const DEFAULT_QUEUE_SEND_TIMEOUT_SECS: u64 = 30;


impl TranslationEngine {
    pub fn from_config(config: EngineConfig) -> Result<Self, TranslatorError> {
        tracing::info!(model_path = %config.model_path.display(), "Loading TranslateGemma model");
        let model = Arc::new(LoadedGemmaModel::load(&config.model_path)?);

        let n_slots = config.n_slots.unwrap_or_else(auto_n_slots);
        let kv_budget_per_slot = config.max_tokens.unwrap_or(DEFAULT_KV_BUDGET_PER_SLOT);
        let queue_capacity = config.queue_capacity.unwrap_or_else(|| (n_slots * 8).max(128));
        let queue_send_timeout_secs = config.queue_send_timeout_secs.unwrap_or(DEFAULT_QUEUE_SEND_TIMEOUT_SECS);
        let queue_send_timeout = std::time::Duration::from_secs(queue_send_timeout_secs);
        let prefill_delay_ms = config.prefill_delay_ms.unwrap_or(DEFAULT_PREFILL_ACCUMULATION_MS);
        let max_chunk_chars = config.max_chunk_chars
            .unwrap_or_else(|| (kv_budget_per_slot.saturating_sub(100) * 4) as usize);
        let paragraph_target_chars = config.paragraph_target_chars
            .unwrap_or_else(|| max_chunk_chars * 3 / 5);
        tracing::info!(n_slots, kv_budget_per_slot, queue_capacity, queue_send_timeout_secs, prefill_delay_ms, max_chunk_chars, paragraph_target_chars, "engine config resolved");

        let (tx, rx) = crossbeam_channel::bounded(queue_capacity);
        let handle = std::thread::Builder::new()
            .name("translator-scheduler".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    ContinuousScheduler::new(model, rx, n_slots, kv_budget_per_slot, prefill_delay_ms).run()
                }));
                if let Err(panic) = result {
                    let msg = panic
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| panic.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("unknown panic");
                    tracing::error!(
                        panic = msg,
                        "translator-scheduler thread panicked — service is down until restart"
                    );
                }
            })
            .expect("failed to spawn translator-scheduler thread");

        #[cfg(feature = "opentelemetry")]
        let meter = opentelemetry::global::meter("translator");
        Ok(Self {
            worker_tx: tx,
            _scheduler_guard: Arc::new(SchedulerGuard(Some(handle))),
            detector: Arc::new(Detector::new()),
            queue_capacity,
            queue_send_timeout,
            kv_budget_per_slot,
            max_chunk_chars,
            paragraph_target_chars,
            #[cfg(feature = "opentelemetry")]
            requests: meter.u64_counter("translator.translation.requests").build(),
            #[cfg(feature = "opentelemetry")]
            batch_size: meter.u64_histogram("translator.translation.batch_size").build(),
            #[cfg(feature = "opentelemetry")]
            duration_ms: meter
                .f64_histogram("translator.translation.duration_ms")
                .with_boundaries(vec![
                    100., 250., 500., 1000., 2000., 5000., 10000., 30000., 60000., 120000.,
                ])
                .build(),
        })
    }

    /// Convenience constructor taking a direct model path.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, TranslatorError> {
        Self::from_config(EngineConfig {
            model_path: model_path.as_ref().to_path_buf(),
            n_slots: None,
            max_tokens: None,
            queue_capacity: None,
            prefill_delay_ms: None,
            max_chunk_chars: None,
            paragraph_target_chars: None,
            queue_send_timeout_secs: None,
        })
    }

    /// Resolved KV budget per slot (tokens) — for chunking thresholds.
    pub fn kv_budget_per_slot(&self) -> u32 {
        self.kv_budget_per_slot
    }

    /// Hard ceiling (in characters) for text chunks sent to the model.
    pub fn max_chunk_chars(&self) -> usize {
        self.max_chunk_chars
    }

    /// Paragraph-level packing target (in characters).
    pub fn paragraph_target_chars(&self) -> usize {
        self.paragraph_target_chars
    }

    /// Translate a batch, automatically chunking any texts that exceed
    /// `max_chunk_chars`. Delegates to `translate_batch()` on the fast path.
    pub fn translate_batch_chunked(
        &self,
        batch: TranslationBatch,
    ) -> Result<TranslationResultSet, TranslatorError> {
        use crate::chunking::chunk_text;

        let max_chars = self.max_chunk_chars;
        let paragraph_target = self.paragraph_target_chars;

        // Fast path: no chunking needed.
        if !batch.texts.iter().any(|t| t.len() > max_chars) {
            return self.translate_batch(batch);
        }

        // Slow path: chunk long texts, translate, reassemble.
        let mut chunked_texts: Vec<String> = Vec::new();
        let mut chunk_separators: Vec<&'static str> = Vec::new();
        // (original_idx, start_chunk_idx, chunk_count)
        let mut chunk_map: Vec<(usize, usize, usize)> = Vec::new();

        for (i, text) in batch.texts.iter().enumerate() {
            let chunks = chunk_text(text, paragraph_target, max_chars);
            let start = chunked_texts.len();
            let count = chunks.len();
            for chunk in chunks {
                chunked_texts.push(chunk.text);
                chunk_separators.push(chunk.join_separator);
            }
            chunk_map.push((i, start, count));
        }

        let chunked_batch = TranslationBatch {
            texts: chunked_texts,
            target_languages: batch.target_languages.clone(),
            source_language: batch.source_language,
        };

        let chunked_result = self.translate_batch(chunked_batch)?;

        // Reassemble: concatenate chunk translations per language for each original text.
        let mut results = Vec::with_capacity(chunk_map.len());
        for (orig_idx, start, count) in &chunk_map {
            if *count == 1 {
                let mut result = chunked_result.results[*start].clone();
                result.source_text = batch.texts[*orig_idx].clone();
                results.push(result);
            } else {
                let first = &chunked_result.results[*start];
                let mut merged_translations = first.translations.clone();
                let mut merged_errors = first.errors.clone();

                let rest = (*start + 1)..(*start + *count);
                for (sep, chunk_result) in chunk_separators[rest.clone()].iter()
                    .zip(&chunked_result.results[rest])
                {
                    for (lang, translation) in &chunk_result.translations {
                        merged_translations
                            .entry(*lang)
                            .and_modify(|existing| {
                                existing.push_str(sep);
                                existing.push_str(translation);
                            })
                            .or_insert_with(|| translation.clone());
                    }
                    for (lang, err) in &chunk_result.errors {
                        merged_errors.entry(*lang).or_insert_with(|| err.clone());
                    }
                }

                results.push(TranslationResult {
                    source_text: batch.texts[*orig_idx].clone(),
                    detected_language: first.detected_language,
                    translations: merged_translations,
                    errors: merged_errors,
                });
            }
        }

        Ok(TranslationResultSet { results })
    }

    /// Detect the language of `text`, returning a lowercase ISO 639-1 code.
    pub fn detect_language(&self, text: &str) -> Result<String, TranslatorError> {
        self.detector.detect(text)
    }

    /// Detect the language of `text`, returning full metadata including Lingua confidence.
    pub fn detect_language_full(
        &self,
        text: &str,
    ) -> Result<LanguageDetectionResult, TranslatorError> {
        let (code, _language_name, confidence) = self.detector.detect_with_confidence(text)?;
        let lang = code.parse::<Language>().ok();
        Ok(LanguageDetectionResult {
            language: lang,
            confidence,
            translation_supported: lang.is_some(),
        })
    }

    /// Translate a batch of texts into all requested target languages.
    #[tracing::instrument(skip(self, batch), fields(n_texts = batch.texts.len(), n_targets = batch.target_languages.len()))]
    pub fn translate_batch(
        &self,
        batch: TranslationBatch,
    ) -> Result<TranslationResultSet, TranslatorError> {
        if batch.texts.is_empty() {
            return Ok(TranslationResultSet { results: vec![] });
        }

        let n = batch.texts.len();

        #[cfg(feature = "opentelemetry")]
        let _start = std::time::Instant::now();
        #[cfg(feature = "opentelemetry")]
        {
            self.requests.add(1, &[]);
            self.batch_size.record(n as u64, &[]);
        }

        // Phase 1 — resolve source languages: use caller hint or detect in parallel via rayon.
        let t0 = std::time::Instant::now();
        let source_langs: Vec<Result<Language, TranslatorError>> = if let Some(src) = batch.source_language {
            (0..n).map(|_| Ok(src)).collect()
        } else {
            batch
                .texts
                .par_iter()
                .map(|text| {
                    let code = self.detector.detect(text)?;
                    code.parse::<Language>().map_err(|_| {
                        TranslatorError::DetectionFailed(format!(
                            "detected language '{code}' is not in the supported set"
                        ))
                    })
                })
                .collect()
        };
        tracing::debug!(detection_ms = t0.elapsed().as_millis(), "phase 1 done");

        let mut all_translations: Vec<HashMap<Language, String>> =
            (0..n).map(|_| HashMap::new()).collect();
        let mut all_errors: Vec<HashMap<Language, TranslationItemError>> =
            (0..n).map(|_| HashMap::new()).collect();

        // Phase 2 — build flat list of work items for all texts × target languages.
        let mut work_texts: Vec<String> = vec![];
        let mut work_expected_lens: Vec<usize> = vec![];
        let mut work_indices: Vec<(usize, Language)> = vec![];

        for i in 0..n {
            let src = match &source_langs[i] {
                Ok(lang) => *lang,
                Err(e) => {
                    // Per-text detection failure: populate errors for all targets, skip inference.
                    let item_err = TranslationItemError::from(e);
                    for tgt in &batch.target_languages {
                        all_errors[i].insert(*tgt, item_err.clone());
                    }
                    continue;
                }
            };

            // Empty text shortcut — return as-is without inference.
            if batch.texts[i].trim().is_empty() {
                for tgt in &batch.target_languages {
                    all_translations[i].insert(*tgt, batch.texts[i].clone());
                }
                continue;
            }

            for &tgt in &batch.target_languages {
                // Same-language shortcut — return original text unchanged.
                if tgt == src {
                    all_translations[i].insert(tgt, batch.texts[i].clone());
                    continue;
                }

                let text = &batch.texts[i];
                // chars / 3 ≈ 1.5 tokens (UTF-8 bytes / 3 ≈ tokens), +15 slack.
                // Scaled by language pair expansion ratio for better EOS bias timing.
                let expected_output_len =
                    ((text.len() as f32 / 3.0 + 15.0) * language::expansion_ratio(src, tgt)) as usize;
                let expected_output_len = expected_output_len.clamp(15, SLOT_CAPACITY);
                let prompt = translate_gemma_prompt(src, tgt, text);
                work_texts.push(prompt);
                work_expected_lens.push(expected_output_len);
                work_indices.push((i, tgt));
            }
        }

        // Phase 3 — dispatch work items to the continuous scheduler.
        if !work_texts.is_empty() {
            let work_item_count = work_texts.len();
            let tx = &self.worker_tx;

            tracing::debug!(
                work_items = work_item_count,
                queue_len = tx.len(),
                queue_capacity = self.queue_capacity,
                "dispatching to scheduler"
            );

            // Single shared reply channel — all N slots reply into one receiver.
            // Each reply carries its index so results can be placed in order.
            let (reply_tx, reply_rx) =
                std::sync::mpsc::channel::<(usize, Result<String, TranslatorError>)>();
            let mut enqueued = 0usize;
            let send_deadline = self.queue_send_timeout;
            for (idx, (text, expected_output_len)) in
                work_texts.into_iter().zip(work_expected_lens).enumerate()
            {
                let req = InferRequest {
                    text,
                    expected_output_len,
                    index: idx,
                    reply_tx: reply_tx.clone(),
                };
                match tx.try_send(req) {
                    Ok(()) => {}
                    Err(crossbeam_channel::TrySendError::Full(req)) => {
                        tx.send_timeout(req, send_deadline).map_err(|_| {
                            TranslatorError::ServiceUnavailable(
                                "translation queue full — timed out waiting for capacity".into(),
                            )
                        })?;
                    }
                    Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                        return Err(TranslatorError::TranslationFailed(
                            "scheduler stopped".into(),
                        ));
                    }
                }
                enqueued += 1;
            }
            drop(reply_tx); // close our copy so channel ends after N replies

            let t2 = std::time::Instant::now();
            let mut first_reply_ms: Option<u128> = None;
            let mut translated: Vec<Option<String>> = vec![None; work_item_count];
            for n_recv in 0..enqueued {
                let (idx, result) = reply_rx
                    .recv()
                    .map_err(|_| TranslatorError::TranslationFailed("scheduler dropped reply".into()))?;
                if n_recv == 0 {
                    first_reply_ms = Some(t2.elapsed().as_millis());
                }
                match result {
                    Ok(text) => translated[idx] = Some(text),
                    Err(e) => {
                        let (text_idx, target_lang) = work_indices[idx];
                        all_errors[text_idx].insert(target_lang, TranslationItemError::from(&e));
                    }
                }
            }
            tracing::debug!(
                first_reply_ms = first_reply_ms.unwrap_or(0),
                all_replies_ms = t2.elapsed().as_millis(),
                work_items = enqueued,
                "phase 4 done (replies received)"
            );

            for (idx, &(text_idx, target_lang)) in work_indices.iter().enumerate() {
                if let Some(translation) = translated[idx].take() {
                    all_translations[text_idx].insert(target_lang, translation);
                }
            }
        }

        // Assemble results, preserving original order.
        let results = (0..n)
            .map(|i| {
                let detected = source_langs[i].as_ref().ok().copied();
                let mut translations = std::mem::take(&mut all_translations[i]);
                if let Some(src) = detected {
                    translations
                        .entry(src)
                        .or_insert_with(|| batch.texts[i].clone());
                }
                TranslationResult {
                    source_text: batch.texts[i].clone(),
                    detected_language: detected,
                    translations,
                    errors: std::mem::take(&mut all_errors[i]),
                }
            })
            .collect();

        #[cfg(feature = "opentelemetry")]
        self.duration_ms.record(_start.elapsed().as_millis() as f64, &[]);

        Ok(TranslationResultSet { results })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_format() {
        let prompt = translate_gemma_prompt(Language::En, Language::Fr, "Hello");
        assert!(prompt.contains("Translate from English to French:"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.starts_with("<bos>"));
    }
}
