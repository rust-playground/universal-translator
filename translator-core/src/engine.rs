use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::detector::Detector;
use crate::error::TranslatorError;
use crate::model::LoadedModel;
use crate::scheduler::{ContinuousScheduler, InferRequest};
use crate::types::{TranslationBatch, TranslationResult, TranslationResultSet};
use tokio::task;

/// Decode strategy for the translation engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeMode {
    /// Greedy decoding — maximum throughput.
    #[default]
    Greedy,
    /// Batched beam search with width 2 (reserved for Phase 2 custom decoder).
    Beam2,
}


/// Fix character-encoding corruption produced by the MADLAD-400 model for Icelandic.
///
/// Three Icelandic characters are consistently substituted with visually similar
/// Latin Extended characters:
///   ó (U+00F3) → ķ (U+0137)
///   ð (U+00F0) → đ (U+0111)
///   þ (U+00FE) → ū (U+016B)
/// and their uppercase counterparts. None of these substitutes are valid Icelandic,
/// so the reversal is unambiguous.
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

/// Map ISO 639-1 code → MADLAD-400 language token.
/// Format: `<2{iso639-1}>` prepended to the source text before tokenization.
/// MADLAD's spiece.model vocabulary uses 2-letter ISO 639-1 codes only.
fn madlad_lang_token(lang: &str) -> Option<&'static str> {
    match lang {
        "af" => Some("<2af>"),
        "ar" => Some("<2ar>"),
        "az" => Some("<2az>"),
        "be" => Some("<2be>"),
        "bg" => Some("<2bg>"),
        "bn" => Some("<2bn>"),
        "ca" => Some("<2ca>"),
        "cs" => Some("<2cs>"),
        "cy" => Some("<2cy>"),
        "da" => Some("<2da>"),
        "de" => Some("<2de>"),
        "el" => Some("<2el>"),
        "en" => Some("<2en>"),
        "es" => Some("<2es>"),
        "et" => Some("<2et>"),
        "eu" => Some("<2eu>"),
        "fa" => Some("<2fa>"),
        "fi" => Some("<2fi>"),
        "fr" => Some("<2fr>"),
        "gu" => Some("<2gu>"),
        "he" => Some("<2he>"),
        "hi" => Some("<2hi>"),
        "hr" => Some("<2hr>"),
        "hu" => Some("<2hu>"),
        "hy" => Some("<2hy>"),
        "id" => Some("<2id>"),
        "is" => Some("<2is>"),
        "it" => Some("<2it>"),
        "ja" => Some("<2ja>"),
        "kk" => Some("<2kk>"),
        "ko" => Some("<2ko>"),
        "lt" => Some("<2lt>"),
        "lv" => Some("<2lv>"),
        "mk" => Some("<2mk>"),
        "ml" => Some("<2ml>"),
        "mn" => Some("<2mn>"),
        "mr" => Some("<2mr>"),
        "ms" => Some("<2ms>"),
        "nl" => Some("<2nl>"),
        "no" => Some("<2no>"),
        "pa" => Some("<2pa>"),
        "pl" => Some("<2pl>"),
        "pt" => Some("<2pt>"),
        "ro" => Some("<2ro>"),
        "ru" => Some("<2ru>"),
        "sk" => Some("<2sk>"),
        "sl" => Some("<2sl>"),
        "so" => Some("<2so>"),
        "sq" => Some("<2sq>"),
        "sr" => Some("<2sr>"),
        "sv" => Some("<2sv>"),
        "sw" => Some("<2sw>"),
        "ta" => Some("<2ta>"),
        "te" => Some("<2te>"),
        "th" => Some("<2th>"),
        "tr" => Some("<2tr>"),
        "uk" => Some("<2uk>"),
        "ur" => Some("<2ur>"),
        "vi" => Some("<2vi>"),
        "xh" => Some("<2xh>"),
        "yo" => Some("<2yo>"),
        "zh" => Some("<2zh>"),
        _ => None,
    }
}

/// Map regional locale codes to their base ISO 639-1 code for language token lookup.
///
/// Callers may pass BCP-47 regional variants like `zh-cn` or `fr-ca`; this normalises
/// them before the `madlad_lang_token` lookup so the right prefix is selected.
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

// Not supported — source detection impossible or absent from MADLAD-400 vocabulary:
//   gl (Galician): Latin script, statistically indistinct from Portuguese/Spanish;
//                  all speakers are fluent in the already-supported `es`.
//   mt (Maltese):  Latin script, ~520K speakers with near-universal `en` proficiency;
//                  lingua excludes it due to unreliable detection accuracy.
//   eo (Esperanto), tl (Tagalog): confirmed absent from MADLAD-400 vocabulary.
//   ga (Irish):    low-resource in CommonCrawl, likely absent.
//   bs (Bosnian):  maps to `sr` in MADLAD; detection overlaps heavily with `hr`/`sr`.
//   lg (Ganda), mi (Māori), ts (Tsonga), tn (Tswana): too uncertain without spiece verification.
// Malayalam (ml) is supported via script-based fallback detection in detector.rs.
// `nb`/`nn` (Norwegian Bokmål/Nynorsk) are normalised to `no` in normalize_lang_code().
//
// Removed after smoke test (3-text × 66-language run, 2026-02):
//   ka (Georgian):        garbage output across all 3 inputs — archaic/invalid codepoints,
//                         no real Georgian words produced.
//   zu (Zulu):            semantic failure — "Ngiyabonga" (Thank you) instead of greeting;
//                         date partially untranslated.
//   st (Southern Sotho):  semantic failure — greeting outputs "Thank you, I love you";
//                         date returns source text unchanged (passthrough).
//   sn (Shona):           date passthrough (source English returned verbatim);
//                         greeting uses non-Shona "Heano". Partial coverage only.

/// All target language codes supported by this engine.
/// Only languages verified to produce correct output via smoke testing.
pub fn supported_target_languages() -> &'static [&'static str] {
    &[
        "af", "ar", "az", "be", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et",
        "eu", "fa", "fi", "fr", "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "kk",
        "ko", "lt", "lv", "mk", "ml", "mn", "mr", "ms", "nl", "no", "pa", "pl", "pt", "ro", "ru",
        "sk", "sl", "so", "sq", "sr", "sv", "sw", "ta", "te", "th", "tr", "uk", "ur", "vi", "xh",
        "yo", "zh",
    ]
}

/// Language codes that are fully supported: detectable as source AND translatable as target.
/// 75 languages are detected by lingua; Malayalam (ml) is detected via Unicode script analysis.
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

/// The central translation engine. Cheap to clone — all heavy state is reference-counted.
#[derive(Clone)]
pub struct TranslationEngine {
    models_dir: PathBuf,
    /// Single MADLAD-400-3B-MT model shared across all language pairs.
    model_cache: Arc<OnceLock<Arc<LoadedModel>>>,
    detector: Arc<Detector>,
    /// Stored for potential future beam-search path selection; currently unused
    /// because the ContinuousScheduler always uses the custom greedy decoder.
    _decode_mode: DecodeMode,
    /// Sender half of the continuous-batching scheduler channel. Initialized lazily.
    work_tx: Arc<OnceLock<std::sync::mpsc::Sender<InferRequest>>>,
}

impl TranslationEngine {
    pub fn new(models_dir: impl AsRef<Path>, decode_mode: DecodeMode) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            model_cache: Arc::new(OnceLock::new()),
            detector: Arc::new(Detector::new()),
            _decode_mode: decode_mode,
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

        // Phase 2 — build a flat list of work items for all texts × target languages.
        // MADLAD translates directly from source to target with no English pivot step.
        // The language token (e.g. "<2fr>") is prepended to the source text AS A STRING
        // before SentencePiece tokenization — inserting it post-tokenization maps to <unk>.
        let mut work_texts: Vec<String> = vec![];
        let mut work_indices: Vec<(usize, String)> = vec![];

        for i in 0..n {
            let src = source_langs[i].as_str();
            for target_lang in &batch.target_languages {
                // Normalise regional variants (e.g. zh-cn → zh) for language token lookup.
                // We keep `target_lang` as the key in the result maps so callers see their
                // original code back; only the token lookup uses `norm_lang`.
                let norm_lang = normalize_lang_code(target_lang);

                if norm_lang == src || target_lang.as_str() == src {
                    // Same language (or regional variant of source) — return original text unchanged.
                    all_translations[i].insert(target_lang.clone(), batch.texts[i].clone());
                    continue;
                }

                match madlad_lang_token(norm_lang) {
                    None => {
                        all_errors[i].insert(
                            target_lang.clone(),
                            format!("No MADLAD token for language: {norm_lang}"),
                        );
                    }
                    Some(prefix) => {
                        work_texts.push(format!("{prefix} {}", batch.texts[i]));
                        work_indices.push((i, target_lang.clone()));
                    }
                }
            }
        }

        // Phase 3 — dispatch each work item individually to the continuous scheduler.
        // The scheduler maintains a slot pool and processes items concurrently; callers
        // wait only for their own items, not for an entire batch to complete.
        if !work_texts.is_empty() {
            let tx = self.get_or_start_worker();
            let mut reply_rxs = Vec::with_capacity(work_texts.len());
            for text in work_texts {
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                tx.send(InferRequest { text, reply_tx })
                    .map_err(|_| {
                        TranslatorError::TranslationFailed("scheduler stopped".into())
                    })?;
                reply_rxs.push(reply_rx);
            }
            let mut translated = Vec::with_capacity(reply_rxs.len());
            for rx in reply_rxs {
                translated.push(
                    rx.await
                        .map_err(|_| {
                            TranslatorError::TranslationFailed("scheduler dropped reply".into())
                        })??,
                );
            }
            for ((text_idx, target_lang), result) in work_indices.iter().zip(translated) {
                all_translations[*text_idx].insert(target_lang.clone(), result);
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
    ///
    /// Loads the model (blocking), then hands it to `ContinuousScheduler::run`
    /// which drives the slot pool for the lifetime of the process.
    fn get_or_start_worker(&self) -> &std::sync::mpsc::Sender<InferRequest> {
        self.work_tx.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            let model_cache = self.model_cache.clone();
            let model_dir = self.models_dir.join("madlad400-3b-mt");
            tokio::spawn(async move {
                tracing::info!("Continuous scheduler starting, loading model…");
                let model =
                    match task::spawn_blocking(move || LoadedModel::load(&model_dir, 4)).await {
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
