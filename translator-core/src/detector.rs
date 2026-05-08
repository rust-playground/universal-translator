use lingua::{Language as LinguaLanguage, LanguageDetector, LanguageDetectorBuilder};

use crate::dialect;
use crate::error::TranslatorError;

pub struct Detector {
    inner: LanguageDetector,
}

impl Detector {
    pub fn new() -> Self {
        let mut builder = LanguageDetectorBuilder::from_all_spoken_languages();
        builder.with_preloaded_language_models();
        let inner = builder.build();
        Self { inner }
    }

    /// Returns a BCP 47 code (`"en"`, `"zh-Hant"`, `"pt-BR"`, `"sr-Cyrl"`).
    ///
    /// Pipeline:
    /// 1. Lingua base detection (75 languages, lowercase ISO 639-1 / 639-3).
    /// 2. Script post-processing — refines `zh`, `sr`, `az`, `pa`, `mn` based
    ///    on Unicode ranges in the input.
    /// 3. Heuristic dialect refinement — refines `pt`, `en`, `fr` based on
    ///    region-specific marker words. Best-effort; falls back to base when
    ///    no commit.
    /// 4. Malayalam fallback (script-only) if lingua returns nothing.
    ///
    /// Detect's universe is broader than the translate-side `Language` enum —
    /// callers must be prepared for codes that don't round-trip into translate
    /// (e.g. `cy`, `ka`, `eu`).
    pub fn detect(&self, text: &str) -> Result<String, TranslatorError> {
        if let Some(lang) = self.inner.detect_language_of(text) {
            let base = lingua_to_bcp47(&lang);
            return Ok(refine(&base, text));
        }
        if let Some(code) = detect_script_only(text) {
            return Ok(code.to_string());
        }
        Err(TranslatorError::DetectionFailed(format!(
            "Could not detect language for text: {text:?}"
        )))
    }

    /// Returns `(bcp47_code, language_name, confidence)`.
    ///
    /// **Confidence semantics:** `top / (top + second)` — fraction of probability
    /// mass between the top two Lingua candidates that belongs to the winner.
    /// Relative score: "how clearly does the top language beat its nearest
    /// competitor?" not absolute certainty.
    ///
    /// Lingua's raw scores sum to 1.0 across ~75 languages, so even an obvious
    /// detection like English "Hello, how are you?" scores ~17% in absolute
    /// terms. The relative formula yields ~73% on that input, 95%+ on longer
    /// or script-distinctive text.
    ///
    /// Practical interpretation:
    /// - > 0.90 — strong, unambiguous signal
    /// - 0.70–0.90 — confident; short or common phrases land here
    /// - 0.50–0.70 — moderate; treat as a best guess
    /// - < 0.50 — weak; very short or genuinely ambiguous
    ///
    /// Script fallback (Malayalam) and refinement steps don't alter confidence;
    /// the score reflects the base-language pass only.
    pub fn detect_with_confidence(
        &self,
        text: &str,
    ) -> Result<(String, String, f64), TranslatorError> {
        let values = self.inner.compute_language_confidence_values(text.to_string());
        let mut iter = values.into_iter();
        if let Some((lang, top)) = iter.next() {
            let second = iter.next().map(|(_, s)| s).unwrap_or(0.0);
            let confidence = if top + second > 0.0 { top / (top + second) } else { 1.0 };
            let base = lingua_to_bcp47(&lang);
            let refined = refine(&base, text);
            let name = format!("{lang:?}");
            return Ok((refined, name, confidence));
        }
        if let Some(code) = detect_script_only(text) {
            return Ok((code.to_string(), "Malayalam".to_string(), 1.0));
        }
        Err(TranslatorError::DetectionFailed(format!(
            "Could not detect language for text: {text:?}"
        )))
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

/// BCP 47 codes the detector can emit, paired with their English names.
///
/// This is a superset of the translate-side `Language` enum. Includes lingua's
/// 75 base languages plus the script/region refinements produced by the
/// post-processing pipeline. Codes outside the translate enum (e.g. `cy`,
/// `ka`, `eu`, `sr-Cyrl`, `pa-Guru`) won't round-trip into translation; the
/// translate side rejects them with `UnsupportedLanguage`.
///
/// Sorted by code.
pub fn detect_supported_codes() -> &'static [(&'static str, &'static str)] {
    DETECT_SUPPORTED_CODES
}

static DETECT_SUPPORTED_CODES: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("az-Arab", "Azerbaijani (Arabic)"),
    ("az-Cyrl", "Azerbaijani (Cyrillic)"),
    ("az-Latn", "Azerbaijani (Latin)"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("en-GB", "English (United Kingdom) — heuristic"),
    ("en-US", "English (United States) — heuristic"),
    ("eo", "Esperanto"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("fr-CA", "Canadian French — heuristic"),
    ("fr-FR", "European French — heuristic"),
    ("ga", "Irish"),
    ("gu", "Gujarati"),
    ("ha", "Hausa"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lg", "Ganda"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mn-Cyrl", "Mongolian (Cyrillic)"),
    ("mn-Mong", "Mongolian (Traditional script)"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("nb", "Norwegian Bokmål"),
    ("nl", "Dutch"),
    ("nn", "Norwegian Nynorsk"),
    ("pa", "Punjabi"),
    ("pa-Arab", "Punjabi (Shahmukhi)"),
    ("pa-Guru", "Punjabi (Gurmukhi)"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("pt-BR", "Brazilian Portuguese — heuristic"),
    ("pt-PT", "European Portuguese — heuristic"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("sr-Cyrl", "Serbian (Cyrillic)"),
    ("sr-Latn", "Serbian (Latin)"),
    ("st", "Southern Sotho"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tl", "Tagalog"),
    ("tn", "Tswana"),
    ("tr", "Turkish"),
    ("ts", "Tsonga"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("xh", "Xhosa"),
    ("yo", "Yoruba"),
    ("zh", "Chinese"),
    ("zh-CN", "Simplified Chinese"),
    ("zh-TW", "Traditional Chinese"),
    ("zu", "Zulu"),
];

/// Refine a base language code with script and dialect post-processing.
///
/// Non-destructive: returns the base unchanged when no refinement applies.
fn refine(base: &str, text: &str) -> String {
    if let Some(refined) = script_disambiguate(base, text) {
        return refined.to_string();
    }
    if let Some(refined) = dialect::disambiguate(base, text) {
        return refined.to_string();
    }
    base.to_string()
}

/// Script-based regional disambiguation.
///
/// Uses Unicode block / character-set membership to refine languages whose
/// regional variants are written in different scripts. Deterministic, no
/// false positives — a script either appears or it doesn't.
fn script_disambiguate(base: &str, text: &str) -> Option<&'static str> {
    match base {
        "zh" => disambiguate_chinese(text),
        "sr" => disambiguate_by_script(
            text,
            &[is_cyrillic, is_basic_latin],
            &["sr-Cyrl", "sr-Latn"],
        ),
        "az" => disambiguate_by_script(
            text,
            &[is_cyrillic, is_arabic, is_basic_latin],
            &["az-Cyrl", "az-Arab", "az-Latn"],
        ),
        "pa" => disambiguate_by_script(
            text,
            &[is_gurmukhi, is_arabic],
            &["pa-Guru", "pa-Arab"],
        ),
        "mn" => disambiguate_by_script(
            text,
            &[is_cyrillic, is_mongolian],
            &["mn-Cyrl", "mn-Mong"],
        ),
        _ => None,
    }
}

/// Pick the first script (in priority order) whose predicate matches any
/// character in the text. Returns `None` if none match.
fn disambiguate_by_script(
    text: &str,
    predicates: &[fn(char) -> bool],
    codes: &[&'static str],
) -> Option<&'static str> {
    debug_assert_eq!(predicates.len(), codes.len());
    for c in text.chars() {
        for (i, pred) in predicates.iter().enumerate() {
            if pred(c) {
                return Some(codes[i]);
            }
        }
    }
    None
}

/// Distinguish Simplified vs Traditional Chinese by character-set membership.
///
/// Strategy: scan for characters that exist exclusively in one set. Most Han
/// characters are shared between Simplified and Traditional; only specific
/// codepoints are exclusive to one form. The lists below are a curated subset
/// of high-frequency exclusive characters — not the full unihan database.
///
/// First exclusive-character hit wins. If only shared characters appear,
/// returns `None` (caller keeps the base `zh`).
///
/// Returns the **region form** (`zh-CN` / `zh-TW`) rather than the script
/// form (`zh-Hans` / `zh-Hant`) for consistency with WMT24++ training set.
/// `FromStr` accepts the script form as an input alias, so callers passing
/// either form to `/translate` work unchanged.
fn disambiguate_chinese(text: &str) -> Option<&'static str> {
    for c in text.chars() {
        if SIMPLIFIED_EXCLUSIVE.contains(&c) {
            return Some("zh-CN");
        }
        if TRADITIONAL_EXCLUSIVE.contains(&c) {
            return Some("zh-TW");
        }
    }
    None
}

/// High-frequency Simplified-only Han characters (subset of HSK 1–4).
/// These have no Traditional counterpart at the same codepoint.
const SIMPLIFIED_EXCLUSIVE: &[char] = &[
    '简', '体', '国', '们', '会', '说', '话', '电', '脑', '车', '马', '鸟', '鱼', '门', '问',
    '间', '关', '开', '听', '见', '见', '为', '这', '个', '时', '间', '业', '东', '书', '买',
    '卖', '长', '专', '产', '从', '过', '里', '万', '与', '丰', '丽', '义', '乐', '习', '乡',
    '书', '买', '乱', '争', '于', '亏', '亚', '产', '亩', '亲', '亲', '什', '仅', '仆', '仇',
    '仑', '仓', '仪', '们', '价', '众', '优', '伙', '会', '伞', '伟', '传', '伤', '伦', '伪',
    '体', '余', '佣', '佥', '侠', '侣', '侥', '侦', '侧', '侨', '侩', '侪', '侬', '俣',
];

/// High-frequency Traditional-only Han characters.
/// These have no Simplified counterpart at the same codepoint.
const TRADITIONAL_EXCLUSIVE: &[char] = &[
    '繁', '體', '國', '們', '會', '說', '話', '電', '腦', '車', '馬', '鳥', '魚', '門', '問',
    '間', '關', '開', '聽', '見', '為', '這', '個', '時', '業', '東', '書', '買', '賣', '長',
    '專', '產', '從', '過', '裡', '萬', '與', '豐', '麗', '義', '樂', '習', '鄉', '亂', '爭',
    '虧', '亞', '畝', '親', '僅', '僕', '仇', '崙', '倉', '儀', '價', '眾', '優', '夥', '傘',
    '偉', '傳', '傷', '倫', '偽', '餘', '傭', '俠', '侶', '僥', '偵', '側', '僑', '儈', '儕',
];

/// Malayalam script (U+0D00–U+0D7F) — script-only fallback for when lingua
/// returns nothing. Lingua doesn't cover Malayalam in its base 75.
fn detect_script_only(text: &str) -> Option<&'static str> {
    if text.chars().any(|c| ('\u{0D00}'..='\u{0D7F}').contains(&c)) {
        return Some("ml");
    }
    None
}

fn lingua_to_bcp47(language: &LinguaLanguage) -> String {
    language.iso_code_639_1().to_string().to_lowercase()
}

// ── Script predicates ────────────────────────────────────────────────────────

fn is_cyrillic(c: char) -> bool {
    matches!(c as u32, 0x0400..=0x04FF | 0x0500..=0x052F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F)
}

fn is_basic_latin(c: char) -> bool {
    matches!(c as u32,
        0x0041..=0x005A     // A-Z
        | 0x0061..=0x007A   // a-z
        | 0x00C0..=0x024F   // Latin-1 Supplement, Latin Extended-A/B
    )
}

fn is_arabic(c: char) -> bool {
    matches!(c as u32, 0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF)
}

fn is_gurmukhi(c: char) -> bool {
    matches!(c as u32, 0x0A00..=0x0A7F)
}

fn is_mongolian(c: char) -> bool {
    matches!(c as u32, 0x1800..=0x18AF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malayalam_script_fallback() {
        // Standalone helper test (doesn't construct lingua detector).
        assert_eq!(detect_script_only("നമസ്കാരം"), Some("ml"));
        assert_eq!(detect_script_only("hello"), None);
    }

    #[test]
    fn chinese_traditional_chars() {
        assert_eq!(disambiguate_chinese("繁體中文測試"), Some("zh-TW"));
    }

    #[test]
    fn chinese_simplified_chars() {
        assert_eq!(disambiguate_chinese("简体中文测试"), Some("zh-CN"));
    }

    #[test]
    fn chinese_shared_chars_no_commit() {
        // "中文" appears in both Simplified and Traditional — shared.
        assert_eq!(disambiguate_chinese("中文"), None);
    }

    #[test]
    fn serbian_cyrillic() {
        assert_eq!(script_disambiguate("sr", "Здраво свете"), Some("sr-Cyrl"));
    }

    #[test]
    fn serbian_latin() {
        assert_eq!(script_disambiguate("sr", "Zdravo svete"), Some("sr-Latn"));
    }

    #[test]
    fn punjabi_gurmukhi() {
        assert_eq!(script_disambiguate("pa", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"), Some("pa-Guru"));
    }

    #[test]
    fn punjabi_arabic_script() {
        // Shahmukhi (Arabic-script Punjabi).
        assert_eq!(script_disambiguate("pa", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ سلام"), Some("pa-Guru"));
        // Pure Shahmukhi text without Gurmukhi.
        assert_eq!(script_disambiguate("pa", "سلام"), Some("pa-Arab"));
    }

    #[test]
    fn mongolian_cyrillic() {
        assert_eq!(script_disambiguate("mn", "Сайн байна уу"), Some("mn-Cyrl"));
    }

    #[test]
    fn unrecognized_base_returns_none() {
        assert_eq!(script_disambiguate("ja", "こんにちは"), None);
        assert_eq!(script_disambiguate("ko", "안녕하세요"), None);
    }

    #[test]
    fn refine_passthrough_when_no_signal() {
        assert_eq!(refine("ja", "こんにちは"), "ja");
        assert_eq!(refine("en", "I went to the store today."), "en");
    }

    #[test]
    fn refine_applies_script_disambig() {
        assert_eq!(refine("zh", "繁體中文測試"), "zh-TW");
    }

    #[test]
    fn refine_applies_dialect_disambig() {
        assert_eq!(
            refine("pt", "Onde fica o ônibus para o aeroporto? Eu uso o trem."),
            "pt-BR"
        );
    }
}
