use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

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
    models_dir: PathBuf,
    /// Cached model reference — populated on first use.
    model_cache: Arc<OnceLock<Arc<LoadedGemmaModel>>>,
    detector: Arc<Detector>,
    /// Scheduler channel sender — initialised on first use.
    worker: Arc<OnceLock<crossbeam_channel::Sender<InferRequest>>>,
    /// Number of parallel decode slots.
    n_slots: usize,
    /// Bounded channel capacity — respects QUEUE_CAPACITY env var.
    queue_capacity: usize,
    #[cfg(feature = "opentelemetry")]
    requests: opentelemetry::metrics::Counter<u64>,
    #[cfg(feature = "opentelemetry")]
    batch_size: opentelemetry::metrics::Histogram<u64>,
    #[cfg(feature = "opentelemetry")]
    duration_ms: opentelemetry::metrics::Histogram<f64>,
}

impl TranslationEngine {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        let n_slots: usize = std::env::var("MAX_DECODE_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        // Default: max(n_slots*4, 512) so that "all" language batches
        // (e.g. 9 texts × 55 languages = 495 items) don't instant-reject.
        let queue_capacity: usize = std::env::var("QUEUE_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (n_slots * 4).max(512));

        #[cfg(feature = "opentelemetry")]
        let meter = opentelemetry::global::meter("translator");
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            model_cache: Arc::new(OnceLock::new()),
            detector: Arc::new(Detector::new()),
            worker: Arc::new(OnceLock::new()),
            n_slots,
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
        }
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
            let tx = self.get_or_start_worker();

            // Backpressure check: reject entire batch if insufficient queue capacity.
            let available = self.queue_capacity - tx.len();
            if available < work_item_count {
                return Err(TranslatorError::ServiceUnavailable(format!(
                    "translation queue full: {work_item_count} items needed, {available} available"
                )));
            }

            let mut reply_rxs = Vec::with_capacity(work_item_count);
            for (text, expected_output_len) in work_texts.into_iter().zip(work_expected_lens) {
                let (reply_tx, reply_rx) = std::sync::mpsc::channel();
                tx.try_send(InferRequest { text, expected_output_len, reply_tx })
                    .map_err(|_| TranslatorError::ServiceUnavailable(
                        "translation queue full".into()
                    ))?;
                reply_rxs.push(reply_rx);
            }

            let mut translated = Vec::with_capacity(work_item_count);
            for rx in reply_rxs {
                translated.push(
                    rx.recv()
                        .map_err(|_| TranslatorError::TranslationFailed("scheduler dropped reply".into()))
                        .and_then(|r| r)?
                );
            }

            for ((text_idx, target_lang), result) in work_indices.iter().zip(translated) {
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

    /// Returns the scheduler channel sender, starting the scheduler on first call.
    fn get_or_start_worker(&self) -> &crossbeam_channel::Sender<InferRequest> {
        let n_slots = self.n_slots;
        let queue_capacity = self.queue_capacity;
        self.worker.get_or_init(|| {
            let (tx, rx) = crossbeam_channel::bounded(queue_capacity);
            let model_cache = self.model_cache.clone();
            let model_dir = self.models_dir.join("translategemma-4b");
            std::thread::Builder::new()
                .name("translator-scheduler".into())
                .spawn(move || {
                    tracing::info!("Continuous scheduler starting, loading model…");
                    let model = match LoadedGemmaModel::load(&model_dir) {
                        Ok(m) => Arc::new(m),
                        Err(e) => {
                            tracing::error!("Scheduler failed to load model: {e}");
                            return;
                        }
                    };
                    let _ = model_cache.set(model.clone());
                    ContinuousScheduler::new(model, rx, n_slots).run();
                })
                .expect("failed to spawn translator-scheduler thread");
            tx
        })
    }
}
