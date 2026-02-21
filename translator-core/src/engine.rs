use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::task;

use crate::detector::Detector;
use crate::error::TranslatorError;
use crate::model::LoadedModel;
use crate::types::{TranslationBatch, TranslationResult, TranslationResultSet};

/// Fix character-encoding corruption produced by the en-is (Icelandic) model.
///
/// The Helsinki-NLP opus-mt-en-is model consistently substitutes three Icelandic
/// characters with visually similar Latin Extended characters:
///   ó (U+00F3) → ķ (U+0137)
///   ð (U+00F0) → đ (U+0111)
///   þ (U+00FE) → ū (U+016B)
/// and their uppercase counterparts. None of these substitutes are valid Icelandic
/// characters, so the reversal is unambiguous.
fn fix_icelandic_chars(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'ķ' => 'ó',
            'Ķ' => 'Ó',
            'đ' => 'ð',
            'Đ' => 'Ð',
            'ū' => 'þ',
            'Ū' => 'Þ',
            _ => c,
        })
        .collect()
}

/// All en-mul prefix tokens for languages reachable via en-mul.
fn all_mul_tokens(lang: &str) -> Option<&'static str> {
    match lang {
        "af" => Some(">>afr<<"),
        "ar" => Some(">>ara<<"),
        "bg" => Some(">>bul<<"),
        "ca" => Some(">>cat<<"),
        "cs" => Some(">>ces<<"),
        "cy" => Some(">>cym<<"),
        "da" => Some(">>dan<<"),
        "de" => Some(">>deu<<"),
        "el" => Some(">>ell<<"),
        "eo" => Some(">>epo<<"),
        "es" => Some(">>spa<<"),
        "et" => Some(">>est<<"),
        "eu" => Some(">>eus<<"),
        "fi" => Some(">>fin<<"),
        "fr" => Some(">>fra<<"),
        "he" => Some(">>heb<<"),
        "hi" => Some(">>hin<<"),
        "hu" => Some(">>hun<<"),
        "hy" => Some(">>hye<<"),
        "id" => Some(">>ind<<"),
        "is" => Some(">>isl<<"),
        "it" => Some(">>ita<<"),
        "ja" => Some(">>jpn<<"),
        "lt" => Some(">>lit<<"),
        "lv" => Some(">>lav<<"),
        "mk" => Some(">>mkd<<"),
        "ml" => Some(">>mal<<"),
        "mr" => Some(">>mar<<"),
        "nl" => Some(">>nld<<"),
        "pt" => Some(">>por<<"),
        "ro" => Some(">>ron<<"),
        "ru" => Some(">>rus<<"),
        "sk" => Some(">>slk<<"),
        "sq" => Some(">>sqi<<"),
        "sv" => Some(">>swe<<"),
        "sw" => Some(">>swh<<"),
        "tl" => Some(">>tgl<<"),
        "tr" => Some(">>tur<<"),
        "uk" => Some(">>ukr<<"),
        "ur" => Some(">>urd<<"),
        "vi" => Some(">>vie<<"),
        "zh" => Some(">>zho<<"),
        _ => None,
    }
}

/// Languages where the dedicated en-X model produces clearly inferior output
/// and en-mul should be preferred instead.
///
/// Determined by side-by-side comparison of all 42 dual-model languages.
fn prefer_mul(lang: &str) -> bool {
    matches!(lang, "cy" | "mr" | "ja")
}

/// en-mul target token for Tier-1 languages that have a dedicated en-X model on disk,
/// used as fallback when the dedicated model directory is missing.
fn mul_target_token(lang: &str) -> Option<&'static str> {
    match lang {
        "cy" => Some(">>cym<<"),  // Welsh (fallback if en-cy missing)
        "eo" => Some(">>epo<<"),  // Esperanto (fallback if en-eo missing)
        "eu" => Some(">>eus<<"),  // Basque (fallback if en-eu missing)
        "hy" => Some(">>hye<<"),  // Armenian (fallback if en-hy missing)
        "is" => Some(">>isl<<"),  // Icelandic (fallback if en-is missing)
        "lv" => Some(">>lav<<"),  // Latvian (fallback if en-lv missing)
        "mk" => Some(">>mkd<<"),  // Macedonian (fallback if en-mk missing)
        _ => None,
    }
}

/// Map regional locale codes to their base ISO 639-1 code for model directory lookup.
///
/// The engine stores models as `en-zh`, `en-fr`, etc. Callers may pass BCP-47 regional
/// variants like `zh-cn` or `fr-ca`; this function normalises them before the lookup so
/// the right model directory is found.
fn normalize_lang_code(code: &str) -> &str {
    match code {
        "zh-hk" | "zh-cn" | "zh-tw" => "zh",
        "fr-ca" => "fr",
        "es-mx" => "es",
        "pt-br" | "pt-pt" => "pt",
        "nb" | "nn" => "no",  // Norwegian Bokmål / Nynorsk
        other => other,
    }
}

// Not supported — source detection impossible without adding a new dependency:
//   gl (Galician): Latin script, statistically indistinct from Portuguese/Spanish;
//                  all speakers are fluent in the already-supported `es`.
//   mt (Maltese):  Latin script, ~520K speakers with near-universal `en` proficiency;
//                  lingua excludes it due to unreliable detection accuracy.
// Malayalam (ml) is supported via script-based fallback detection in detector.rs.

/// All target language codes supported by this engine.
/// Only languages verified to produce correct output via smoke testing.
pub fn supported_target_languages() -> &'static [&'static str] {
    &[
        "af", "ar", "bg", "ca", "cs", "cy", "da", "de", "el", "en",
        "eo", "es", "et", "eu", "fi", "fr", "he", "hi", "hu",
        "hy", "id", "is", "it", "ja", "lt", "lv", "mk", "ml", "mr",
        "nl", "pt", "ro", "ru", "sk", "sq", "sv", "sw", "tl",
        "tr", "uk", "ur", "vi", "zh",
    ]
}

/// Language codes that are fully supported: detectable as source AND translatable as target.
/// 42 languages are detected by lingua; Malayalam (ml) is detected via Unicode script analysis.
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

/// Resolved plan for one target language — model key + prefix only, no loaded model yet.
struct WorkPlan {
    target_lang: String,
    to_translate: Vec<usize>,
    texts_batch: Vec<String>,
    model_key: String,
    prefix: Option<String>,
}

struct WorkItem {
    target_lang: String,
    to_translate: Vec<usize>,
    texts_batch: Vec<String>,
    model: Arc<LoadedModel>,
    prefix: Option<String>,
}

/// The central translation engine. Cheap to clone — all heavy state is reference-counted.
#[derive(Clone)]
pub struct TranslationEngine {
    models_dir: PathBuf,
    /// Key: "en-fr", "en-de", etc.
    cache: Arc<DashMap<String, Arc<LoadedModel>>>,
    detector: Arc<Detector>,
    /// 0 = dynamic (compute per-request from work item count), n > 0 = fixed.
    threads_per_model: usize,
}

impl TranslationEngine {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            cache: Arc::new(DashMap::new()),
            detector: Arc::new(Detector::new()),
            threads_per_model: 0,
        }
    }

    /// Override the thread count used when loading models.
    /// Use for API/operator deployments; `0` (the default) means dynamic.
    pub fn with_threads_per_model(mut self, n: usize) -> Self {
        self.threads_per_model = n;
        self
    }

    fn model_exists(&self, pair: &str) -> bool {
        self.models_dir.join(pair).exists()
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
        let parallelism = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        // Phase 1 — detect all languages in parallel.
        let detect_handles: Vec<_> = batch.texts.iter()
            .map(|text| {
                let engine = self.clone();
                let text = text.clone();
                task::spawn(async move { engine.detect_language(&text).await })
            })
            .collect();
        let mut source_langs = Vec::with_capacity(n);
        for handle in detect_handles {
            source_langs.push(
                handle.await.map_err(|e| TranslatorError::TranslationFailed(e.to_string()))??
            );
        }

        // Phase 2 — pivot non-English texts to English, grouped by source language.
        // Phase 2 loads one model at a time so give it full parallelism.
        let mut english_texts: Vec<String> = batch.texts.clone();
        let mut pivot_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, lang) in source_langs.iter().enumerate() {
            if lang != "en" {
                pivot_groups.entry(lang.clone()).or_default().push(i);
            }
        }
        for (src_lang, indices) in &pivot_groups {
            let texts_for_model: Vec<String> = indices.iter().map(|&i| batch.texts[i].clone()).collect();
            let model = match self.get_or_load_model(&format!("{src_lang}-en"), parallelism).await {
                Ok(m) => m,
                Err(TranslatorError::ModelNotFound(_)) => {
                    self.get_or_load_model("mul-en", parallelism).await
                        .map_err(|_| TranslatorError::ModelNotFound(format!("{src_lang}-en")))?
                }
                Err(e) => return Err(e),
            };
            let model_ref = model.clone();
            let translated = task::spawn_blocking(move || model_ref.translate_batch(&texts_for_model))
                .await
                .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))??;
            for (&orig_idx, en_text) in indices.iter().zip(translated) {
                english_texts[orig_idx] = en_text;
            }
        }

        // Phase 3 — translate to each target language, batching all texts per language.
        let mut all_translations: Vec<HashMap<String, String>> = (0..n).map(|_| HashMap::new()).collect();
        let mut all_errors: Vec<HashMap<String, String>> = (0..n).map(|_| HashMap::new()).collect();

        // Phase 3a-plan — determine model key and prefix for each target language (sync, no I/O).
        // Passthrough and error cases are handled immediately; translatable items are
        // collected into WorkPlans for loading and dispatch in subsequent phases.
        let mut work_plans: Vec<WorkPlan> = vec![];

        for target_lang in &batch.target_languages {
            // Normalise regional variants (e.g. zh-cn → zh) for model directory lookup.
            // We keep `target_lang` as the key in the result maps so callers see their
            // original code back; only the model path uses `norm_lang`.
            let norm_lang = normalize_lang_code(target_lang);

            let mut to_translate: Vec<usize> = vec![];
            let mut texts_batch: Vec<String> = vec![];

            for i in 0..n {
                let src = source_langs[i].as_str();
                if norm_lang == src || target_lang.as_str() == src {
                    // Same language (or regional variant of source) — return original text unchanged.
                    all_translations[i].insert(target_lang.clone(), batch.texts[i].clone());
                } else if norm_lang == "en" {
                    // English target — already have it from the pivot step.
                    all_translations[i].insert(target_lang.clone(), english_texts[i].clone());
                } else {
                    to_translate.push(i);
                    texts_batch.push(english_texts[i].clone());
                }
            }

            if to_translate.is_empty() {
                continue;
            }

            // Languages where en-mul beats the dedicated model — route directly to en-mul.
            if prefer_mul(norm_lang) {
                match all_mul_tokens(norm_lang) {
                    None => {
                        let msg = format!("No mul token for en-{norm_lang}");
                        for &i in &to_translate {
                            all_errors[i].insert(target_lang.clone(), msg.clone());
                        }
                    }
                    Some(token) => work_plans.push(WorkPlan {
                        target_lang: target_lang.clone(),
                        to_translate,
                        texts_batch,
                        model_key: "en-mul".into(),
                        prefix: Some(token.to_string()),
                    }),
                }
                continue;
            }

            // Dedicated model path with mul fallback.
            if self.model_exists(&format!("en-{norm_lang}")) {
                work_plans.push(WorkPlan {
                    target_lang: target_lang.clone(),
                    to_translate,
                    texts_batch,
                    model_key: format!("en-{norm_lang}"),
                    prefix: None,
                });
            } else {
                match mul_target_token(norm_lang) {
                    None => {
                        let msg = format!("No model available for en-{norm_lang}");
                        for &i in &to_translate {
                            all_errors[i].insert(target_lang.clone(), msg.clone());
                        }
                    }
                    Some(token) => work_plans.push(WorkPlan {
                        target_lang: target_lang.clone(),
                        to_translate,
                        texts_batch,
                        model_key: "en-mul".into(),
                        prefix: Some(token.to_string()),
                    }),
                }
            }
        }

        // Phase 3a-load — compute thread count, load each unique model, convert WorkPlan → WorkItem.
        let threads_per_model = if self.threads_per_model > 0 {
            self.threads_per_model
        } else {
            let concurrent = work_plans.len().min(parallelism).max(1);
            (parallelism / concurrent).max(1)
        };

        let mut work_items: Vec<WorkItem> = Vec::with_capacity(work_plans.len());
        for plan in work_plans {
            match self.get_or_load_model(&plan.model_key, threads_per_model).await {
                Ok(model) => work_items.push(WorkItem {
                    target_lang: plan.target_lang,
                    to_translate: plan.to_translate,
                    texts_batch: plan.texts_batch,
                    model,
                    prefix: plan.prefix,
                }),
                Err(e) => {
                    for &i in &plan.to_translate {
                        all_errors[i].insert(plan.target_lang.clone(), e.to_string());
                    }
                }
            }
        }

        // Phase 3b — dispatch work items in concurrent rounds of available_parallelism().
        // Each model uses threads_per_model intra-op threads; round size ensures total active
        // threads stays within parallelism. Processing in rounds prevents thread explosion.
        let mut work_iter = work_items.into_iter().peekable();
        while work_iter.peek().is_some() {
            let handles: Vec<_> = work_iter.by_ref().take(parallelism)
                .map(|item| {
                    task::spawn_blocking(move || {
                        let result = match item.prefix {
                            Some(ref p) => item.model.translate_batch_with_prefix(&item.texts_batch, p),
                            None => item.model.translate_batch(&item.texts_batch),
                        };
                        (item.target_lang, item.to_translate, result)
                    })
                })
                .collect();
            for handle in handles {
                let (target_lang, to_translate, result) = handle
                    .await
                    .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))?;
                match result {
                    Ok(out) => {
                        for (&i, t) in to_translate.iter().zip(out) {
                            all_translations[i].insert(target_lang.clone(), t);
                        }
                    }
                    Err(e) => {
                        for &i in &to_translate {
                            all_errors[i].insert(target_lang.clone(), e.to_string());
                        }
                    }
                }
            }
        }

        // Apply per-language post-processing fixes for known model output bugs.
        for translations in &mut all_translations {
            if let Some(t) = translations.get_mut("is") {
                *t = fix_icelandic_chars(t);
            }
        }

        // Assemble results, preserving original order.
        let results = (0..n)
            .map(|i| {
                let mut translations = std::mem::take(&mut all_translations[i]);
                // Always include the detected source language for caller convenience.
                translations.entry(source_langs[i].clone()).or_insert_with(|| batch.texts[i].clone());
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

    /// Returns a cached model, loading it on first access with `num_threads` intra-op threads.
    /// On a cache hit the cached model is returned as-is regardless of `num_threads`.
    async fn get_or_load_model(&self, pair: &str, num_threads: usize) -> Result<Arc<LoadedModel>, TranslatorError> {
        // Fast path: already cached.
        if let Some(model) = self.cache.get(pair) {
            return Ok(model.clone());
        }

        let model_dir = self.models_dir.join(pair);
        if !model_dir.exists() {
            return Err(TranslatorError::ModelNotFound(pair.to_string()));
        }

        // Slow path: load from disk on a blocking thread.
        let model = task::spawn_blocking(move || LoadedModel::load(&model_dir, num_threads))
            .await
            .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))??;

        let model = Arc::new(model);
        // Benign race: if another thread inserted concurrently, we drop our copy.
        self.cache.entry(pair.to_string()).or_insert_with(|| model.clone());

        Ok(model)
    }
}
