use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::task;

use crate::detector::Detector;
use crate::error::TranslatorError;
use crate::model::LoadedModel;
use crate::types::{TranslationBatch, TranslationResult, TranslationResultSet};

/// en-mul target token for Tier-1 languages that have a dedicated en-X model on disk,
/// used as fallback when the dedicated model directory is missing.
fn mul_target_token(lang: &str) -> Option<&'static str> {
    match lang {
        "cy" => Some(">>cym<<"),  // Welsh (fallback if en-cy missing)
        "eo" => Some(">>epo<<"),  // Esperanto (fallback if en-eo missing)
        "eu" => Some(">>eus<<"),  // Basque (fallback if en-eu missing)
        "gl" => Some(">>glg<<"),  // Galician (fallback if en-gl missing)
        "hy" => Some(">>hye<<"),  // Armenian (fallback if en-hy missing)
        "is" => Some(">>isl<<"),  // Icelandic (fallback if en-is missing)
        "lv" => Some(">>lav<<"),  // Latvian (fallback if en-lv missing)
        "mk" => Some(">>mkd<<"),  // Macedonian (fallback if en-mk missing)
        "mt" => Some(">>mlt<<"),  // Maltese (fallback if en-mt missing)
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

/// All target language codes supported by this engine.
/// Only languages verified to produce correct output via smoke testing.
pub fn supported_target_languages() -> &'static [&'static str] {
    &[
        "af", "ar", "bg", "ca", "cs", "cy", "da", "de", "el", "eo",
        "es", "et", "eu", "fi", "fr", "gl", "he", "hi", "hu", "hy",
        "id", "is", "it", "lt", "lv", "mk", "ml", "mr", "mt", "nl",
        "pt", "ro", "ru", "sk", "sq", "sv", "sw", "tl", "tr", "uk",
        "ur", "vi", "zh",
    ]
}

/// Language codes that are fully supported: detectable as source AND translatable as target.
/// This is the intersection of lingua's detectable set and supported_target_languages().
pub fn supported_languages() -> Vec<&'static str> {
    use lingua::Language;
    let detectable: std::collections::HashSet<String> = Language::all()
        .into_iter()
        .map(|l| format!("{:?}", l.iso_code_639_1()).to_lowercase())
        .collect();
    let mut langs: Vec<&'static str> = supported_target_languages()
        .iter()
        .copied()
        .filter(|&code| detectable.contains(code))
        .collect();
    langs.sort_unstable();
    langs
}

/// The central translation engine. Cheap to clone — all heavy state is reference-counted.
#[derive(Clone)]
pub struct TranslationEngine {
    models_dir: PathBuf,
    /// Key: "en-fr", "en-de", etc.
    cache: Arc<DashMap<String, Arc<LoadedModel>>>,
    detector: Arc<Detector>,
}

impl TranslationEngine {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            cache: Arc::new(DashMap::new()),
            detector: Arc::new(Detector::new()),
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

        let mut results = Vec::with_capacity(batch.texts.len());
        for text in &batch.texts {
            let result = self.translate_text(text, &batch.target_languages).await?;
            results.push(result);
        }
        Ok(TranslationResultSet { results })
    }

    async fn translate_text(
        &self,
        text: &str,
        target_languages: &[String],
    ) -> Result<TranslationResult, TranslatorError> {
        // Detect source language.
        let source_lang = self.detect_language(text).await?;

        // Pivot: if the source isn't English, translate to English first.
        let english_text = if source_lang != "en" {
            let model = match self.get_or_load_model(&format!("{source_lang}-en")).await {
                Ok(m) => m,
                Err(TranslatorError::ModelNotFound(_)) => {
                    // Fallback to multilingual→English for languages without a dedicated model.
                    self.get_or_load_model("mul-en").await
                        .map_err(|_| TranslatorError::ModelNotFound(
                            format!("{source_lang}-en")
                        ))?
                }
                Err(e) => return Err(e),
            };
            let input = vec![text.to_string()];
            let model_ref = model.clone();
            task::spawn_blocking(move || model_ref.translate_batch(&input))
                .await
                .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))??
                .into_iter()
                .next()
                .unwrap_or_default()
        } else {
            text.to_string()
        };

        let mut translations: HashMap<String, String> = HashMap::new();
        let mut errors: HashMap<String, String> = HashMap::new();

        for target_lang in target_languages {
            // Normalise regional variants (e.g. zh-cn → zh) for model directory lookup.
            // We keep `target_lang` as the key in the result maps so callers see their
            // original code back; only the model path uses `norm_lang`.
            let norm_lang = normalize_lang_code(target_lang);

            // Same language (or regional variant of source) — return original text unchanged.
            if norm_lang == source_lang || target_lang == &source_lang {
                translations.insert(target_lang.clone(), text.to_string());
                continue;
            }

            // English target — already have it from the pivot step.
            if norm_lang == "en" {
                translations.insert(target_lang.clone(), english_text.clone());
                continue;
            }

            // Dedicated model path.
            match self.get_or_load_model(&format!("en-{norm_lang}")).await {
                Err(TranslatorError::ModelNotFound(_)) => {
                    if let Some(token) = mul_target_token(norm_lang) {
                        match self.get_or_load_model("en-mul").await {
                            Ok(model) => {
                                let input = vec![english_text.clone()];
                                let token = token.to_string();
                                let model_ref = model.clone();
                                match task::spawn_blocking(move || {
                                    model_ref.translate_batch_with_prefix(&input, &token)
                                })
                                .await
                                .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))
                                {
                                    Ok(Ok(mut out)) => {
                                        translations.insert(target_lang.clone(), out.pop().unwrap_or_default());
                                    }
                                    Ok(Err(e)) => { errors.insert(target_lang.clone(), e.to_string()); }
                                    Err(e) => { errors.insert(target_lang.clone(), e.to_string()); }
                                }
                            }
                            Err(e) => { errors.insert(target_lang.clone(), e.to_string()); }
                        }
                    } else {
                        errors.insert(
                            target_lang.clone(),
                            format!("No model available for en-{norm_lang}"),
                        );
                    }
                }
                Err(e) => {
                    errors.insert(target_lang.clone(), e.to_string());
                }
                Ok(model) => {
                    let input = vec![english_text.clone()];
                    let model_ref = model.clone();
                    match task::spawn_blocking(move || model_ref.translate_batch(&input)).await {
                        Err(e) => {
                            errors.insert(
                                target_lang.clone(),
                                TranslatorError::TranslationFailed(e.to_string()).to_string(),
                            );
                        }
                        Ok(Err(e)) => {
                            errors.insert(target_lang.clone(), e.to_string());
                        }
                        Ok(Ok(mut translated)) => {
                            if let Some(result) = translated.pop() {
                                translations.insert(target_lang.clone(), result);
                            }
                        }
                    }
                }
            }
        }

        // Always include the detected source language in translations for caller convenience.
        translations.entry(source_lang.clone()).or_insert_with(|| text.to_string());

        Ok(TranslationResult {
            source_text: text.to_string(),
            detected_language: source_lang,
            translations,
            errors,
        })
    }

    /// Returns a cached model, loading it on first access.
    async fn get_or_load_model(&self, pair: &str) -> Result<Arc<LoadedModel>, TranslatorError> {
        // Fast path: already cached.
        if let Some(model) = self.cache.get(pair) {
            return Ok(model.clone());
        }

        let model_dir = self.models_dir.join(pair);
        if !model_dir.exists() {
            return Err(TranslatorError::ModelNotFound(pair.to_string()));
        }

        // Slow path: load from disk on a blocking thread.
        let model = task::spawn_blocking(move || LoadedModel::load(&model_dir))
            .await
            .map_err(|e| TranslatorError::TranslationFailed(e.to_string()))??;

        let model = Arc::new(model);
        // Benign race: if another thread inserted concurrently, we drop our copy.
        self.cache.entry(pair.to_string()).or_insert_with(|| model.clone());

        Ok(model)
    }
}
