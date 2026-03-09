use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use crate::detector::Detector;
use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::scheduler::{ContinuousScheduler, InferRequest, SLOT_CAPACITY};
use crate::types::{LanguageDetectionResult, TranslationBatch, TranslationResult, TranslationResultSet};

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

/// Build a Gemma instruct-format translation prompt.
///
/// Uses the Gemma 3 chat template with a system turn that constrains output
/// to a single translation (prevents multi-option "helpful assistant" mode):
///
///   <bos>
///   <start_of_turn>system
///   You are a translation engine. Output only the translated text. Do not add explanations, alternatives, notes, or any other text.<end_of_turn>
///   <start_of_turn>user
///   Translate from {src} to {tgt}:
///   {text}<end_of_turn>
///   <start_of_turn>model
///
/// `<bos>` is included so we tokenize without `add_special_tokens`.
/// The model generates the translation and ends with `<end_of_turn>`.
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

/// The central translation engine. Cheap to clone — all heavy state is reference-counted.
#[derive(Clone)]
pub struct TranslationEngine {
    worker_tx: crossbeam_channel::Sender<InferRequest>,
    detector: Arc<Detector>,
    /// Bounded channel capacity — respects QUEUE_CAPACITY env var.
    queue_capacity: usize,
    #[cfg(feature = "opentelemetry")]
    requests: opentelemetry::metrics::Counter<u64>,
    #[cfg(feature = "opentelemetry")]
    batch_size: opentelemetry::metrics::Histogram<u64>,
    #[cfg(feature = "opentelemetry")]
    duration_ms: opentelemetry::metrics::Histogram<f64>,
}

/// Metal: 32 slots — Metal GPU scales sub-linearly above ~10 active slots.
#[allow(dead_code)]
const DEFAULT_MAX_DECODE_SLOTS_METAL: usize = 32;

/// CUDA: 128 slots — NVIDIA GPUs have higher compute throughput and memory bandwidth.
/// With flash attention, scaling remains efficient well beyond 32 active slots.
#[allow(dead_code)]
const DEFAULT_MAX_DECODE_SLOTS_CUDA: usize = 128;

fn auto_n_slots(model: &LoadedGemmaModel) -> usize {
    // Env var always wins — allows manual tuning.
    if let Some(n) = std::env::var("MAX_DECODE_SLOTS").ok().and_then(|v| v.parse().ok()) {
        tracing::info!(n_slots = n, "n_slots from MAX_DECODE_SLOTS env var");
        return n;
    }

    // KV cache bytes per element: F16 on CUDA (flash attention), F32 on Metal/CPU.
    // 200 tokens is a generous average for translation outputs.
    // These are only consumed by the GPU feature branches below; allow unused on CPU builds.
    #[allow(unused_variables)]
    let kv_per_slot: usize = {
        let n_layers = model.n_layers();
        let n_kv_heads = model.n_kv_heads();
        let head_dim = model.head_dim();
        const AVG_OUTPUT_TOKENS: usize = 200;

        #[cfg(feature = "cuda")]
        const BYTES_PER_ELEMENT: usize = 2; // F16 KV cache
        #[cfg(not(feature = "cuda"))]
        const BYTES_PER_ELEMENT: usize = 4; // F32 KV cache

        n_layers * 2 * AVG_OUTPUT_TOKENS * n_kv_heads * head_dim * BYTES_PER_ELEMENT
    };

    #[cfg(feature = "metal")]
    {
        if let Ok(metal_dev) = model.device().as_metal_device() {
            let raw = metal_dev.metal_device();
            let budget = raw.recommended_max_working_set_size();
            let used = raw.current_allocated_size();
            let free = budget.saturating_sub(used);
            // Reserve 15% for forward-pass activation tensors and Metal driver overhead.
            let usable = free * 17 / 20;
            let n = (usable / kv_per_slot.max(1)).clamp(4, DEFAULT_MAX_DECODE_SLOTS_METAL);
            tracing::info!(
                budget_mb = budget / 1_048_576,
                used_mb = used / 1_048_576,
                free_mb = free / 1_048_576,
                kv_slot_mb = kv_per_slot / 1_048_576,
                n_slots = n,
                "auto n_slots (Metal)"
            );
            return n;
        }
    }

    #[cfg(feature = "cuda")]
    {
        if model.device().is_cuda() {
            if let Ok((free, _total)) = cudarc::driver::result::mem_get_info() {
                // Reserve 15% headroom for activations and CUDA driver overhead.
                let usable = free * 17 / 20;
                let n = (usable / kv_per_slot.max(1)).clamp(4, DEFAULT_MAX_DECODE_SLOTS_CUDA);
                tracing::info!(
                    free_mb = free / 1_048_576,
                    kv_slot_mb = kv_per_slot / 1_048_576,
                    n_slots = n,
                    "auto n_slots (CUDA)"
                );
                return n;
            }
        }
    }

    // CPU fallback: decode throughput is memory-bandwidth-bound not slot-count-bound;
    // 4 slots is enough to overlap prefill of new requests with ongoing decode steps.
    tracing::info!(n_slots = 4, "auto n_slots (CPU)");
    4
}

impl TranslationEngine {
    pub fn new(models_dir: impl AsRef<Path>) -> Result<Self, TranslatorError> {
        let model_dir = models_dir.as_ref().join("translategemma-4b");
        tracing::info!(?model_dir, "Loading TranslateGemma model");
        let model = Arc::new(LoadedGemmaModel::load(&model_dir)?);

        let n_slots = auto_n_slots(&model);
        // Default: max(n_slots*4, 512) so that "all" language batches
        // (e.g. 9 texts × 55 languages = 495 items) don't instant-reject.
        let queue_capacity: usize = std::env::var("QUEUE_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (n_slots * 4).max(512));

        let (tx, rx) = crossbeam_channel::bounded(queue_capacity);
        std::thread::Builder::new()
            .name("translator-scheduler".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    ContinuousScheduler::new(model, rx, n_slots).run()
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
            detector: Arc::new(Detector::new()),
            queue_capacity,
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
        tracing::info!(detection_ms = t0.elapsed().as_millis(), "phase 1 done");

        let mut all_translations: Vec<HashMap<String, String>> =
            (0..n).map(|_| HashMap::new()).collect();
        let mut all_errors: Vec<HashMap<String, String>> = (0..n).map(|_| HashMap::new()).collect();

        // Phase 2 — build flat list of work items for all texts × target languages.
        let mut work_texts: Vec<String> = vec![];
        let mut work_expected_lens: Vec<usize> = vec![];
        let mut work_indices: Vec<(usize, String)> = vec![];

        for i in 0..n {
            let src = source_langs[i].as_str();

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

            // Backpressure check: reject entire batch if insufficient queue capacity.
            let available = self.queue_capacity - tx.len();
            if available < work_item_count {
                return Err(TranslatorError::ServiceUnavailable(format!(
                    "translation queue full: {work_item_count} items needed, {available} available"
                )));
            }

            tracing::info!(
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
            let t1 = std::time::Instant::now();
            for (idx, (text, expected_output_len)) in
                work_texts.into_iter().zip(work_expected_lens).enumerate()
            {
                tx.try_send(InferRequest {
                    text,
                    expected_output_len,
                    index: idx,
                    reply_tx: reply_tx.clone(),
                })
                .map_err(|_| TranslatorError::ServiceUnavailable("translation queue full".into()))?;
                enqueued += 1;
            }
            drop(reply_tx); // close our copy so channel ends after N replies
            tracing::info!(enqueue_ms = t1.elapsed().as_millis(), "phase 3 done");

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
                translated[idx] = Some(result?);
            }
            tracing::info!(
                first_reply_ms = first_reply_ms.unwrap_or(0),
                all_replies_ms = t2.elapsed().as_millis(),
                work_items = enqueued,
                "phase 4 done (replies received)"
            );

            for ((text_idx, target_lang), result) in
                work_indices.iter().zip(translated.into_iter().map(|o| o.unwrap()))
            {
                all_translations[*text_idx].insert(target_lang.clone(), result);
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
