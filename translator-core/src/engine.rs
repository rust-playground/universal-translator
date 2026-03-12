use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use crate::detector::Detector;
use crate::error::TranslatorError;
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
    pub models_dir: PathBuf,
    pub model_file: Option<String>,
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

// ── Language helpers ─────────────────────────────────────────────────────────

/// Map ISO 639-1 code → full English language name used in the Gemma prompt.
fn lang_full_name(code: &str) -> &str {
    match code {
        "af" => "Afrikaans",
        "am" => "Amharic",
        "ar" => "Arabic",
        "bg" => "Bulgarian",
        "bn" => "Bengali",
        "ca" => "Catalan",
        "cs" => "Czech",
        "da" => "Danish",
        "de" => "German",
        "el" => "Greek",
        "en" => "English",
        "es" => "Spanish",
        "et" => "Estonian",
        "fa" => "Persian",
        "fi" => "Finnish",
        "fr" => "French",
        "gu" => "Gujarati",
        "ha" => "Hausa",
        "hi" => "Hindi",
        "hr" => "Croatian",
        "hu" => "Hungarian",
        "id" => "Indonesian",
        "it" => "Italian",
        "ja" => "Japanese",
        "kn" => "Kannada",
        "ko" => "Korean",
        "lt" => "Lithuanian",
        "lv" => "Latvian",
        "ml" => "Malayalam",
        "mr" => "Marathi",
        "ms" => "Malay",
        "mt" => "Maltese",
        "ne" => "Nepali",
        "nl" => "Dutch",
        "no" => "Norwegian",
        "pa" => "Punjabi",
        "pl" => "Polish",
        "pt" => "Portuguese",
        "ro" => "Romanian",
        "ru" => "Russian",
        "si" => "Sinhala",
        "sk" => "Slovak",
        "sl" => "Slovenian",
        "sr" => "Serbian",
        "sv" => "Swedish",
        "sw" => "Swahili",
        "ta" => "Tamil",
        "te" => "Telugu",
        "th" => "Thai",
        "tr" => "Turkish",
        "uk" => "Ukrainian",
        "ur" => "Urdu",
        "vi" => "Vietnamese",
        "yi" => "Yiddish",
        "zh" => "Chinese",
        other => other, // unknown code — pass through as-is
    }
}

/// Build a full Gemma instruct-format translation prompt.
fn translate_gemma_prompt(src_lang: &str, tgt_lang: &str, text: &str) -> String {
    format!(
        "<bos><start_of_turn>system\n\
         You are a translation engine. Output only the translated text. \
         Do not add explanations, alternatives, notes, or any other text.<end_of_turn>\n\
         <start_of_turn>user\n\
         Translate from {} to {}:\n{}<end_of_turn>\n\
         <start_of_turn>model\n",
        lang_full_name(src_lang),
        lang_full_name(tgt_lang),
        text
    )
}

/// Map regional locale codes to their base ISO 639-1 code.
fn normalize_lang_code(code: &str) -> &str {
    match code {
        "zh-hk" | "zh-cn" | "zh-tw" => "zh",
        "fr-ca" => "fr",
        "es-mx" => "es",
        "pt-br" | "pt-pt" => "pt",
        "nb" | "nn" => "no", // Norwegian Bokmål / Nynorsk
        other => other,
    }
}

/// All target language codes supported by this engine.
/// 55 languages officially supported by TranslateGemma 4B (https://huggingface.co/google/translategemma-4b-it).
pub fn supported_target_languages() -> &'static [&'static str] {
    &[
        "af", "am", "ar", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa",
        "fi", "fr", "gu", "ha", "hi", "hr", "hu", "id", "it", "ja", "kn", "ko", "lt", "lv",
        "ml", "mr", "ms", "mt", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "si", "sk",
        "sl", "sr", "sv", "sw", "ta", "te", "th", "tr", "uk", "ur", "vi", "yi", "zh",
    ]
}

/// Language codes that are both detectable (by Lingua or script fallback) and
/// translatable as a target.
pub fn supported_languages() -> Vec<&'static str> {
    use lingua::Language;
    let detectable: std::collections::HashSet<String> = Language::all()
        .into_iter()
        .map(|l| format!("{:?}", l.iso_code_639_1()).to_lowercase())
        .collect();
    let mut langs: Vec<&'static str> = supported_target_languages()
        .iter()
        .copied()
        .filter(|&code| detectable.contains(code) || code == "ml")
        .collect();
    langs.sort_unstable();
    langs
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
        let model_dir = config.models_dir.join("translategemma-4b");
        tracing::info!(?model_dir, "Loading TranslateGemma model");
        let model = Arc::new(LoadedGemmaModel::load(&model_dir, config.model_file.as_deref())?);

        let n_slots = config.n_slots.unwrap_or_else(auto_n_slots);
        let kv_budget_per_slot = config.max_tokens.unwrap_or(DEFAULT_KV_BUDGET_PER_SLOT);
        let queue_capacity = config.queue_capacity.unwrap_or_else(|| (n_slots * 64).max(2048));
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

    /// Convenience constructor matching the old `new(models_dir, model_file)` signature.
    pub fn new(models_dir: impl AsRef<Path>, model_file: Option<&str>) -> Result<Self, TranslatorError> {
        Self::from_config(EngineConfig {
            models_dir: models_dir.as_ref().to_path_buf(),
            model_file: model_file.map(String::from),
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
            source_language: batch.source_language.clone(),
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
                            .entry(lang.clone())
                            .and_modify(|existing| {
                                existing.push_str(sep);
                                existing.push_str(translation);
                            })
                            .or_insert_with(|| translation.clone());
                    }
                    for (lang, err) in &chunk_result.errors {
                        merged_errors.entry(lang.clone()).or_insert_with(|| err.clone());
                    }
                }

                results.push(TranslationResult {
                    source_text: batch.texts[*orig_idx].clone(),
                    detected_language: first.detected_language.clone(),
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
        let (code, language_name, confidence) = self.detector.detect_with_confidence(text)?;
        let supported = supported_target_languages().contains(&code.as_str());
        Ok(LanguageDetectionResult {
            language_code: code,
            language: language_name,
            confidence,
            translation_supported: supported,
        })
    }

    /// Translate a batch of texts into all requested target languages.
    #[tracing::instrument(skip(self, batch), fields(n_texts = batch.texts.len(), n_targets = batch.target_languages.len()))]
    pub fn translate_batch(
        &self,
        mut batch: TranslationBatch,
    ) -> Result<TranslationResultSet, TranslatorError> {
        if batch.texts.is_empty() {
            return Ok(TranslationResultSet { results: vec![] });
        }
        if batch.target_languages == ["all"] {
            batch.target_languages = supported_target_languages()
                .iter()
                .map(|s| s.to_string())
                .collect();
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
        let source_langs: Vec<String> = if let Some(ref src) = batch.source_language {
            let normalized = normalize_lang_code(src).to_string();
            vec![normalized; n]
        } else {
            batch
                .texts
                .par_iter()
                .map(|text| self.detector.detect(text))
                .collect::<Result<Vec<String>, TranslatorError>>()?
        };
        tracing::debug!(detection_ms = t0.elapsed().as_millis(), "phase 1 done");

        let mut all_translations: Vec<HashMap<String, String>> =
            (0..n).map(|_| HashMap::new()).collect();
        let mut all_errors: Vec<HashMap<String, String>> = (0..n).map(|_| HashMap::new()).collect();

        // Phase 2 — build flat list of work items for all texts × target languages.
        let mut work_texts: Vec<String> = vec![];
        let mut work_expected_lens: Vec<usize> = vec![];
        let mut work_indices: Vec<(usize, String)> = vec![];

        for i in 0..n {
            let src = source_langs[i].as_str();

            // Empty text shortcut — return as-is without inference.
            if batch.texts[i].trim().is_empty() {
                for target_lang in &batch.target_languages {
                    all_translations[i].insert(target_lang.clone(), batch.texts[i].clone());
                }
                continue;
            }

            for target_lang in &batch.target_languages {
                let norm_lang = normalize_lang_code(target_lang);

                // Same-language shortcut — return original text unchanged.
                if norm_lang == src || target_lang.as_str() == src {
                    all_translations[i].insert(target_lang.clone(), batch.texts[i].clone());
                    continue;
                }

                // Validate that the target language is in our supported set.
                if lang_full_name(norm_lang) == norm_lang
                    && !supported_target_languages().contains(&norm_lang)
                {
                    all_errors[i].insert(
                        target_lang.clone(),
                        format!("Unsupported target language: {norm_lang}"),
                    );
                    continue;
                }

                let text = &batch.texts[i];
                // chars / 3 ≈ 1.5 tokens (UTF-8 bytes / 3 ≈ tokens), +15 slack.
                let expected_output_len = (text.len() / 3 + 15).clamp(15, SLOT_CAPACITY);
                let prompt = translate_gemma_prompt(src, norm_lang, text);
                work_texts.push(prompt);
                work_expected_lens.push(expected_output_len);
                work_indices.push((i, target_lang.clone()));
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
                tx.send_timeout(InferRequest {
                    text,
                    expected_output_len,
                    index: idx,
                    reply_tx: reply_tx.clone(),
                }, send_deadline)
                .map_err(|_| TranslatorError::ServiceUnavailable(
                    "translation queue full — timed out waiting for capacity".into()
                ))?;
                enqueued += 1;
            }
            drop(reply_tx); // close our copy so channel ends after N replies

            let t2 = std::time::Instant::now();
            let mut first_reply_ms: Option<u128> = None;
            let mut translated: Vec<Option<Result<String, TranslatorError>>> =
                (0..work_item_count).map(|_| None).collect();
            for n_recv in 0..enqueued {
                let (idx, result) = reply_rx
                    .recv()
                    .map_err(|_| TranslatorError::TranslationFailed("scheduler dropped reply".into()))?;
                if n_recv == 0 {
                    first_reply_ms = Some(t2.elapsed().as_millis());
                }
                translated[idx] = Some(result);
            }
            tracing::debug!(
                first_reply_ms = first_reply_ms.unwrap_or(0),
                all_replies_ms = t2.elapsed().as_millis(),
                work_items = enqueued,
                "phase 4 done (replies received)"
            );

            for ((text_idx, target_lang), result) in
                work_indices.iter().zip(translated.into_iter().map(|o| o.unwrap()))
            {
                match result {
                    Ok(translation) => {
                        all_translations[*text_idx].insert(target_lang.clone(), translation);
                    }
                    Err(e) => {
                        all_errors[*text_idx].insert(target_lang.clone(), e.to_string());
                    }
                }
            }
        }

        // Assemble results, preserving original order.
        let results = (0..n)
            .map(|i| {
                let mut translations = std::mem::take(&mut all_translations[i]);
                translations
                    .entry(source_langs[i].clone())
                    .or_insert_with(|| batch.texts[i].clone());
                TranslationResult {
                    source_text: batch.texts[i].clone(),
                    detected_language: source_langs[i].clone(),
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
