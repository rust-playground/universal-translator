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
    /// 1. **Script-only fast path** — unique-script blocks (`ml`, `km`, `lo`,
    ///    `my`, `bo`, `si`, `am`, `or`) match deterministically and beat
    ///    anything lingua might guess. Lingua doesn't cover these languages
    ///    so without this check it would emit a wrong-but-confident guess.
    /// 2. Lingua base detection (75 languages, lowercase ISO 639-1 / 639-3).
    /// 3. Script post-processing — refines `zh`, `sr`, `az`, `pa`, `mn` based
    ///    on Unicode ranges in the input.
    /// 4. Heuristic dialect refinement — refines `pt`, `en`, `fr`, `es`,
    ///    `zh-TW` based on region-specific marker words. Best-effort;
    ///    falls back to base when no commit.
    ///
    /// Detect's universe is broader than the translate-side `Language` enum —
    /// callers must be prepared for codes that don't round-trip into translate
    /// (e.g. `cy`, `ka`, `eu`).
    pub fn detect(&self, text: &str) -> Result<String, TranslatorError> {
        if let Some(code) = detect_script_only(text) {
            return Ok(code.to_string());
        }
        if let Some(lang) = self.inner.detect_language_of(text) {
            let base = lingua_to_bcp47(&lang);
            return Ok(refine(&base, text));
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
    /// Script fallback paths report confidence = 1.0 (Unicode-block matches
    /// are deterministic). Refinement steps don't alter confidence.
    pub fn detect_with_confidence(
        &self,
        text: &str,
    ) -> Result<(String, String, f64), TranslatorError> {
        if let Some(code) = detect_script_only(text) {
            return Ok((code.to_string(), script_only_name(code).to_string(), 1.0));
        }
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
        Err(TranslatorError::DetectionFailed(format!(
            "Could not detect language for text: {text:?}"
        )))
    }
}

/// English name for a script-only fallback code. Used when reporting
/// confidence results (Lingua doesn't supply a `Language::*` for these).
fn script_only_name(code: &str) -> &'static str {
    match code {
        "ml" => "Malayalam",
        "km" => "Khmer",
        "lo" => "Lao",
        "my" => "Burmese",
        "bo" => "Tibetan",
        "si" => "Sinhala",
        "am" => "Amharic",
        "or" => "Oriya",
        _ => "",
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
    ("am", "Amharic — script fallback"),
    ("ar", "Arabic"),
    ("as", "Assamese — within-Bengali-script heuristic"),
    ("az", "Azerbaijani"),
    ("az-Arab", "Azerbaijani (Arabic)"),
    ("az-Cyrl", "Azerbaijani (Cyrillic)"),
    ("az-Latn", "Azerbaijani (Latin)"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bo", "Tibetan — script fallback"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("ckb", "Central Kurdish — within-Arabic-script heuristic"),
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
    ("es-ES", "European Spanish — heuristic"),
    ("es-MX", "Mexican Spanish — heuristic"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fil", "Filipino"),
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
    ("km", "Khmer — script fallback"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lg", "Ganda"),
    ("lo", "Lao — script fallback"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam — script fallback"),
    ("mn", "Mongolian"),
    ("mn-Cyrl", "Mongolian (Cyrillic)"),
    ("mn-Mong", "Mongolian (Traditional script)"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("my", "Burmese — script fallback"),
    ("ne", "Nepali — heuristic"),
    ("nl", "Dutch"),
    ("no", "Norwegian (Bokmål/Nynorsk normalized)"),
    ("or", "Oriya — script fallback"),
    ("pa", "Punjabi"),
    ("pa-Arab", "Punjabi (Shahmukhi)"),
    ("pa-Guru", "Punjabi (Gurmukhi)"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("pt-BR", "Brazilian Portuguese — heuristic"),
    ("pt-PT", "European Portuguese — heuristic"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("si", "Sinhala — script fallback"),
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
    ("tn", "Tswana"),
    ("tr", "Turkish"),
    ("ts", "Tsonga"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("xh", "Xhosa"),
    ("yi", "Yiddish — within-Hebrew-script heuristic"),
    ("yo", "Yoruba"),
    ("zh", "Chinese"),
    ("zh-CN", "Simplified Chinese"),
    ("zh-HK", "Hong Kong Chinese — heuristic"),
    ("zh-TW", "Traditional Chinese"),
    ("zu", "Zulu"),
];

/// Refine a base language code with script and dialect post-processing.
///
/// Pipeline: script disambiguation first (deterministic Unicode-block tests),
/// then dialect heuristics on the script-refined code. This lets dialect run
/// on top of a script refinement — e.g. `zh` → `zh-TW` (Traditional) → `zh-HK`
/// if Hong Kong markers fire. Each step is non-destructive: returns the input
/// unchanged when no refinement applies.
fn refine(base: &str, text: &str) -> String {
    let after_script = match script_disambiguate(base, text) {
        Some(refined) => refined.to_string(),
        None => base.to_string(),
    };
    if let Some(refined) = dialect::disambiguate(&after_script, text) {
        return refined.to_string();
    }
    after_script
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
        "bn" => disambiguate_by_script(text, &[is_assamese_distinctive], &["as"]),
        "ar" => disambiguate_by_script(text, &[is_sorani_distinctive], &["ckb"]),
        "he" => disambiguate_yiddish(text),
        _ => None,
    }
}

/// Detect Yiddish within Hebrew script.
///
/// Yiddish writes double-vav (וו) and double-yod (יי) as two adjacent characters
/// — common pattern in Yiddish words (`וועט`, `דייטש`), rare in modern Hebrew.
/// Also commits on the precomposed Hebrew Ligatures (`װ` U+05F0, `ױ` U+05F1,
/// `ײ` U+05F2) which are Yiddish-exclusive.
fn disambiguate_yiddish(text: &str) -> Option<&'static str> {
    for c in text.chars() {
        if matches!(c as u32, 0x05F0..=0x05F2) {
            return Some("yi");
        }
    }
    if text.contains("\u{05D5}\u{05D5}") || text.contains("\u{05D9}\u{05D9}") {
        return Some("yi");
    }
    None
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

/// Script-only fallback for when lingua misses or misroutes.
///
/// Each block is a script that maps unambiguously to a single language in
/// our enum. First codepoint hit wins; callers handle `None` as detection
/// failure.
///
/// Devanagari and Bengali scripts are deliberately **not** covered here —
/// they're shared across multiple supported languages (Devanagari: hi, ne,
/// mr, mai; Bengali: bn, as) and need lingua or heuristics to disambiguate.
///
/// Ethiopic (`am`): script is shared with `ti` (Tigrinya). Defaults to `am`
/// since short text in either is indistinguishable by script alone — the
/// Tigrinya-distinctive Ge'ez letters (qha series, U+1250–U+1258) appear
/// occasionally in Amharic too, so committing on them caused false positives.
fn detect_script_only(text: &str) -> Option<&'static str> {
    for c in text.chars() {
        let hit = match c as u32 {
            0x0D00..=0x0D7F => Some("ml"), // Malayalam
            0x0C80..=0x0CFF => Some("kn"), // Kannada
            0x0B80..=0x0BFF => Some("ta"), // Tamil
            0x0C00..=0x0C7F => Some("te"), // Telugu
            0x0A80..=0x0AFF => Some("gu"), // Gujarati
            0x0A00..=0x0A7F => Some("pa"), // Gurmukhi (Punjabi)
            0x0B00..=0x0B7F => Some("or"), // Oriya
            0x1780..=0x17FF => Some("km"), // Khmer
            0x0E80..=0x0EFF => Some("lo"), // Lao
            0x1000..=0x109F => Some("my"), // Burmese
            0x0F00..=0x0FFF => Some("bo"), // Tibetan
            0x0D80..=0x0DFF => Some("si"), // Sinhala
            0x1200..=0x137F => Some("am"), // Ethiopic (am; ti round-trips as am)
            _ => None,
        };
        if hit.is_some() {
            return hit;
        }
    }
    None
}

/// Convert a Lingua `Language` to the BCP 47 code our enum / consumers expect.
///
/// Most languages map 1:1 by lowercased ISO 639-1. A few exceptions
/// normalize lingua's emitted form into the canonical code we use:
/// - `tl` (Tagalog) → `fil` (Filipino). Same language pragmatically;
///   our enum holds Fil, WMT24++ uses fil_PH.
/// - `nb` (Bokmål) / `nn` (Nynorsk) → `no` (Norwegian macrolanguage).
///   Our enum has a single Norwegian variant.
fn lingua_to_bcp47(language: &LinguaLanguage) -> String {
    let raw = language.iso_code_639_1().to_string().to_lowercase();
    match raw.as_str() {
        "tl" => "fil".to_string(),
        "nb" | "nn" => "no".to_string(),
        other => other.to_string(),
    }
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

/// Assamese-distinctive letters within the Bengali script block.
/// `ৰ` (U+09F0, BENGALI LETTER RA WITH MIDDLE DIAGONAL) and
/// `ৱ` (U+09F1, BENGALI LETTER RA WITH LOWER DIAGONAL) are used in Assamese
/// orthography and don't appear in standard Bengali (which uses `র` U+09B0).
fn is_assamese_distinctive(c: char) -> bool {
    matches!(c as u32, 0x09F0 | 0x09F1)
}

/// Sorani Kurdish-distinctive letters within the Arabic script block.
/// These extended Arabic letters are used in Sorani Kurdish but not in
/// standard Arabic: ێ (U+06ED yeh barree with hamza below), ۆ (U+06C6 oe),
/// ڕ (U+0695 reh with small v below), ڵ (U+06B5 lam with small v below),
/// ڤ (U+06A4 veh).
fn is_sorani_distinctive(c: char) -> bool {
    matches!(c as u32, 0x06ED | 0x06C6 | 0x0695 | 0x06B5 | 0x06A4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_only_fallbacks() {
        // Each block tests one unique-script language.
        assert_eq!(detect_script_only("നമസ്കാരം"), Some("ml")); // Malayalam
        assert_eq!(detect_script_only("ನಮಸ್ಕಾರ"), Some("kn"));    // Kannada
        assert_eq!(detect_script_only("வணக்கம்"), Some("ta"));    // Tamil
        assert_eq!(detect_script_only("నమస్కారం"), Some("te"));   // Telugu
        assert_eq!(detect_script_only("નમસ્તે"), Some("gu"));      // Gujarati
        assert_eq!(detect_script_only("ਸਤ ਸ੍ਰੀ ਅਕਾਲ"), Some("pa")); // Gurmukhi (Punjabi)
        assert_eq!(detect_script_only("ସୁ ଆ ସ୍ୱାଗତ"), Some("or"));   // Oriya
        assert_eq!(detect_script_only("សួស្ដី"), Some("km"));     // Khmer
        assert_eq!(detect_script_only("ສະບາຍດີ"), Some("lo"));   // Lao
        assert_eq!(detect_script_only("မင်္ဂလာပါ"), Some("my"));   // Burmese
        assert_eq!(detect_script_only("བཀྲ་ཤིས་བདེ་ལེགས།"), Some("bo")); // Tibetan
        assert_eq!(detect_script_only("ආයුබෝවන්"), Some("si"));    // Sinhala
        assert_eq!(detect_script_only("ሰላም"), Some("am"));        // Ethiopic → am
        assert_eq!(detect_script_only("hello"), None);            // Latin → no match
    }

    #[test]
    fn assamese_distinctive_letter_refines_bn() {
        // ৰ (U+09F0) appears in Assamese, not standard Bengali.
        assert_eq!(script_disambiguate("bn", "ৰোগী আছে"), Some("as"));
        // ৱ (U+09F1) also Assamese-distinctive.
        assert_eq!(script_disambiguate("bn", "ৱাণিজ্য"), Some("as"));
    }

    #[test]
    fn bengali_without_assamese_letters_returns_none() {
        // Standard Bengali — uses র (U+09B0), not ৰ.
        assert_eq!(script_disambiguate("bn", "আমি বাংলা বলি"), None);
    }

    #[test]
    fn sorani_kurdish_distinctive_letter_refines_ar() {
        // ێ (U+06ED) and ڕ (U+0695) are Sorani-distinctive.
        assert_eq!(script_disambiguate("ar", "ئەو کوێرە"), Some("ckb"));
        assert_eq!(script_disambiguate("ar", "هاوڕێ"), Some("ckb"));
    }

    #[test]
    fn arabic_without_sorani_letters_returns_none() {
        assert_eq!(script_disambiguate("ar", "السلام عليكم"), None);
    }

    #[test]
    fn yiddish_double_letter_digraphs_refine_he() {
        // וו (double vav) in וועט — classic Yiddish.
        assert_eq!(script_disambiguate("he", "וועט קומען"), Some("yi"));
        // יי (double yod) in דייטש.
        assert_eq!(script_disambiguate("he", "דייטש"), Some("yi"));
    }

    #[test]
    fn yiddish_precomposed_ligature_refines_he() {
        // ײ (U+05F2 precomposed double yod).
        assert_eq!(script_disambiguate("he", "ײד"), Some("yi"));
    }

    #[test]
    fn hebrew_without_yiddish_markers_returns_none() {
        assert_eq!(script_disambiguate("he", "שלום עליכם"), None);
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

    #[test]
    fn refine_chains_script_then_dialect_for_zh_hk() {
        // Traditional Chinese with HK Cantonese markers should chain:
        // zh → script_disambig → zh-TW → dialect → zh-HK.
        let text = "我喺巴士站等的士，唔該你話我知點解咁耐。";
        assert_eq!(refine("zh", text), "zh-HK");
    }

    #[test]
    fn refine_applies_es_dialect() {
        assert_eq!(
            refine("es", "Voy a manejar el carro y rentar una computadora."),
            "es-MX"
        );
    }
}
