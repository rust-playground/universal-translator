use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::detector::Detector;
use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::scheduler::{ContinuousScheduler, InferRequest, SLOT_CAPACITY};
use crate::types::{TranslationBatch, TranslationResult, TranslationResultSet};
use futures::future::try_join_all;
use tokio::task;

// ── Language helpers ─────────────────────────────────────────────────────────

/// Map ISO 639-1 code → full English language name used in the Gemma prompt.
fn lang_full_name(code: &str) -> &str {
    match code {
        "af" => "Afrikaans",
        "ar" => "Arabic",
        "az" => "Azerbaijani",
        "be" => "Belarusian",
        "bg" => "Bulgarian",
        "bn" => "Bengali",
        "ca" => "Catalan",
        "cs" => "Czech",
        "cy" => "Welsh",
        "da" => "Danish",
        "de" => "German",
        "el" => "Greek",
        "en" => "English",
        "es" => "Spanish",
        "et" => "Estonian",
        "eu" => "Basque",
        "fa" => "Persian",
        "fi" => "Finnish",
        "fr" => "French",
        "gu" => "Gujarati",
        "he" => "Hebrew",
        "hi" => "Hindi",
        "hr" => "Croatian",
        "hu" => "Hungarian",
        "hy" => "Armenian",
        "id" => "Indonesian",
        "is" => "Icelandic",
        "it" => "Italian",
        "ja" => "Japanese",
        "kk" => "Kazakh",
        "ko" => "Korean",
        "lt" => "Lithuanian",
        "lv" => "Latvian",
        "mk" => "Macedonian",
        "ml" => "Malayalam",
        "mn" => "Mongolian",
        "mr" => "Marathi",
        "ms" => "Malay",
        "nl" => "Dutch",
        "no" => "Norwegian",
        "pa" => "Punjabi",
        "pl" => "Polish",
        "pt" => "Portuguese",
        "ro" => "Romanian",
        "ru" => "Russian",
        "sk" => "Slovak",
        "sl" => "Slovenian",
        "so" => "Somali",
        "sq" => "Albanian",
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
        "xh" => "Xhosa",
        "yo" => "Yoruba",
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
/// Same 62 languages as the previous MADLAD implementation; re-validated
/// against TranslateGemma output quality is deferred to post-deployment.
pub fn supported_target_languages() -> &'static [&'static str] {
    &[
        "af", "ar", "az", "be", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et",
        "eu", "fa", "fi", "fr", "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "kk",
        "ko", "lt", "lv", "mk", "ml", "mn", "mr", "ms", "nl", "no", "pa", "pl", "pt", "ro", "ru",
        "sk", "sl", "so", "sq", "sr", "sv", "sw", "ta", "te", "th", "tr", "uk", "ur", "vi", "xh",
        "yo", "zh",
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
    /// Sender half of the continuous-batching scheduler channel.  Initialised lazily.
    work_tx: Arc<OnceLock<std::sync::mpsc::Sender<InferRequest>>>,
}

impl TranslationEngine {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            model_cache: Arc::new(OnceLock::new()),
            detector: Arc::new(Detector::new()),
            work_tx: Arc::new(OnceLock::new()),
        }
    }

    /// Detect the language of `text`, returning a lowercase ISO 639-1 code.
    pub async fn detect_language(&self, text: &str) -> Result<String, TranslatorError> {
        let text_owned = text.to_string();
        let detector = self.detector.clone();
        task::spawn_blocking(move || detector.detect(&text_owned))
            .await
            .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))?
    }

    /// Translate a batch of texts into all requested target languages.
    pub async fn translate_batch(
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

        // Phase 1 — resolve source languages: use caller hint or detect in parallel.
        let source_langs: Vec<String> = if let Some(ref src) = batch.source_language {
            let normalized = normalize_lang_code(src).to_string();
            vec![normalized; n]
        } else {
            let detect_handles: Vec<_> = batch
                .texts
                .iter()
                .map(|text| {
                    let engine = self.clone();
                    let text = text.clone();
                    task::spawn(async move { engine.detect_language(&text).await })
                })
                .collect();
            let mut langs = Vec::with_capacity(n);
            for handle in detect_handles {
                langs.push(
                    handle
                        .await
                        .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))??,
                );
            }
            langs
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

        // Phase 3 — dispatch each work item individually to the continuous scheduler.
        if !work_texts.is_empty() {
            let tx = self.get_or_start_worker();
            let mut reply_rxs = Vec::with_capacity(work_texts.len());
            for (text, expected_output_len) in work_texts.into_iter().zip(work_expected_lens) {
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                tx.send(InferRequest {
                    text,
                    expected_output_len,
                    reply_tx,
                })
                .map_err(|_| TranslatorError::TranslationFailed("scheduler stopped".into()))?;
                reply_rxs.push(reply_rx);
            }
            let translated: Vec<String> =
                try_join_all(reply_rxs.into_iter().map(|rx| async move {
                    rx.await
                        .map_err(|_| {
                            TranslatorError::TranslationFailed("scheduler dropped reply".into())
                        })
                        .and_then(|r| r)
                }))
                .await?;
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

        Ok(TranslationResultSet { results })
    }

    /// Returns the scheduler channel sender, starting it on first call.
    fn get_or_start_worker(&self) -> &std::sync::mpsc::Sender<InferRequest> {
        self.work_tx.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            let model_cache = self.model_cache.clone();
            let model_dir = self.models_dir.join("translategemma-4b");
            tokio::spawn(async move {
                tracing::info!("Continuous scheduler starting, loading model…");
                let model =
                    match task::spawn_blocking(move || LoadedGemmaModel::load(&model_dir)).await {
                        Ok(Ok(m)) => Arc::new(m),
                        Ok(Err(e)) => {
                            tracing::error!("Scheduler failed to load model: {e}");
                            return;
                        }
                        Err(e) => {
                            tracing::error!("Scheduler model-load panicked: {e}");
                            return;
                        }
                    };
                let _ = model_cache.set(model.clone());
                ContinuousScheduler::new(model, rx).run().await;
            });
            tx
        })
    }
}
