use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::error::TranslatorError;

/// Supported translate-side languages and locales.
///
/// 98 variants: WMT24++ validated set (53 base + 11 regional pairs), plus
/// codes added for HubSpot/harness coverage. Codes the 4B model cannot
/// actually produce were pruned by harness runs — it emits wrong-language or
/// script-salad output for them (e.g. `Pa` → Hindi, `Ti` → Amharic,
/// `Tk`/`Tt` → Kazakh, `Jv` → Indonesian, `Kac` → Burmese). Remaining
/// additions are codes the model can translate, even when the detector
/// struggles to disambiguate them.
///
/// The detector returns a `String` (BCP 47 code) and may produce codes outside
/// this enum (lingua-only base languages, script subtags such as `sr-Cyrl`,
/// `pa-Guru`). `FromStr` falls back to the base language when an unknown
/// region tag is supplied; codes with no recognized base error out.
///
/// Variant naming: regional variants use `lowercase + _ + UPPERCASE` so the
/// source reads as the BCP 47 code at a glance (`pt_BR`). Base variants stay
/// PascalCase (`Pt`).
///
/// Quality tiers (best-effort labels — see eval/ harness for systematic
/// evaluation):
/// - WMT24++ validated: 53 base + 11 regional pairs. Google has metrics.
/// - Inherited best-effort: `Af`, `Am`, `Ms`, `Ne`, `Si`, `Yi` —
///   in the original master enum but not in WMT24++.
/// - Evaluation-validated additions: confirmed translatable by harness
///   (eval/results/).
///
/// Serializes as the BCP 47 code (`"pt-BR"`, `"fr"`).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Af,
    Am,
    Ar,
    ar_EG,
    ar_SA,
    As,
    Ast,
    Az,
    Be,
    Bg,
    Bn,
    Bs,
    Ca,
    Ceb,
    Ckb,
    Cs,
    Cy,
    Da,
    De,
    El,
    En,
    en_GB,
    en_US,
    Es,
    es_ES,
    es_MX,
    Et,
    Eu,
    Fa,
    Fi,
    Fil,
    Fr,
    fr_CA,
    fr_FR,
    Gl,
    Gu,
    He,
    Hi,
    Hr,
    Hu,
    Hy,
    Id,
    Is,
    It,
    Ja,
    Ka,
    Kk,
    Km,
    Kn,
    Ko,
    Ky,
    Lb,
    Lo,
    Lt,
    Lv,
    Mk,
    Ml,
    Mn,
    Mr,
    Ms,
    My,
    Ne,
    Nl,
    No,
    Oc,
    Or,
    Pl,
    Ps,
    Pt,
    pt_BR,
    pt_PT,
    Ro,
    Ru,
    Rw,
    Si,
    Sk,
    Sl,
    Sq,
    Sr,
    Su,
    Sv,
    Sw,
    sw_KE,
    sw_TZ,
    Ta,
    Te,
    Tg,
    Th,
    Tr,
    Uk,
    Ur,
    Uz,
    Vi,
    Yi,
    Zh,
    zh_CN,
    zh_HK,
    zh_TW,
}

static ALL_LANGUAGES: [Language; 98] = [
    Language::Af,
    Language::Am,
    Language::Ar,
    Language::ar_EG,
    Language::ar_SA,
    Language::As,
    Language::Ast,
    Language::Az,
    Language::Be,
    Language::Bg,
    Language::Bn,
    Language::Bs,
    Language::Ca,
    Language::Ceb,
    Language::Ckb,
    Language::Cs,
    Language::Cy,
    Language::Da,
    Language::De,
    Language::El,
    Language::En,
    Language::en_GB,
    Language::en_US,
    Language::Es,
    Language::es_ES,
    Language::es_MX,
    Language::Et,
    Language::Eu,
    Language::Fa,
    Language::Fi,
    Language::Fil,
    Language::Fr,
    Language::fr_CA,
    Language::fr_FR,
    Language::Gl,
    Language::Gu,
    Language::He,
    Language::Hi,
    Language::Hr,
    Language::Hu,
    Language::Hy,
    Language::Id,
    Language::Is,
    Language::It,
    Language::Ja,
    Language::Ka,
    Language::Kk,
    Language::Km,
    Language::Kn,
    Language::Ko,
    Language::Ky,
    Language::Lb,
    Language::Lo,
    Language::Lt,
    Language::Lv,
    Language::Mk,
    Language::Ml,
    Language::Mn,
    Language::Mr,
    Language::Ms,
    Language::My,
    Language::Ne,
    Language::Nl,
    Language::No,
    Language::Oc,
    Language::Or,
    Language::Pl,
    Language::Ps,
    Language::Pt,
    Language::pt_BR,
    Language::pt_PT,
    Language::Ro,
    Language::Ru,
    Language::Rw,
    Language::Si,
    Language::Sk,
    Language::Sl,
    Language::Sq,
    Language::Sr,
    Language::Su,
    Language::Sv,
    Language::Sw,
    Language::sw_KE,
    Language::sw_TZ,
    Language::Ta,
    Language::Te,
    Language::Tg,
    Language::Th,
    Language::Tr,
    Language::Uk,
    Language::Ur,
    Language::Uz,
    Language::Vi,
    Language::Yi,
    Language::Zh,
    Language::zh_CN,
    Language::zh_HK,
    Language::zh_TW,
];

impl Language {
    /// BCP 47 code (`"fr"`, `"pt-BR"`, `"zh-CN"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Af => "af",
            Self::Am => "am",
            Self::Ar => "ar",
            Self::ar_EG => "ar-EG",
            Self::ar_SA => "ar-SA",
            Self::As => "as",
            Self::Ast => "ast",
            Self::Az => "az",
            Self::Be => "be",
            Self::Bg => "bg",
            Self::Bn => "bn",
            Self::Bs => "bs",
            Self::Ca => "ca",
            Self::Ceb => "ceb",
            Self::Ckb => "ckb",
            Self::Cs => "cs",
            Self::Cy => "cy",
            Self::Da => "da",
            Self::De => "de",
            Self::El => "el",
            Self::En => "en",
            Self::en_GB => "en-GB",
            Self::en_US => "en-US",
            Self::Es => "es",
            Self::es_ES => "es-ES",
            Self::es_MX => "es-MX",
            Self::Et => "et",
            Self::Eu => "eu",
            Self::Fa => "fa",
            Self::Fi => "fi",
            Self::Fil => "fil",
            Self::Fr => "fr",
            Self::fr_CA => "fr-CA",
            Self::fr_FR => "fr-FR",
            Self::Gl => "gl",
            Self::Gu => "gu",
            Self::He => "he",
            Self::Hi => "hi",
            Self::Hr => "hr",
            Self::Hu => "hu",
            Self::Hy => "hy",
            Self::Id => "id",
            Self::Is => "is",
            Self::It => "it",
            Self::Ja => "ja",
            Self::Ka => "ka",
            Self::Kk => "kk",
            Self::Km => "km",
            Self::Kn => "kn",
            Self::Ko => "ko",
            Self::Ky => "ky",
            Self::Lb => "lb",
            Self::Lo => "lo",
            Self::Lt => "lt",
            Self::Lv => "lv",
            Self::Mk => "mk",
            Self::Ml => "ml",
            Self::Mn => "mn",
            Self::Mr => "mr",
            Self::Ms => "ms",
            Self::My => "my",
            Self::Ne => "ne",
            Self::Nl => "nl",
            Self::No => "no",
            Self::Oc => "oc",
            Self::Or => "or",
            Self::Pl => "pl",
            Self::Ps => "ps",
            Self::Pt => "pt",
            Self::pt_BR => "pt-BR",
            Self::pt_PT => "pt-PT",
            Self::Ro => "ro",
            Self::Ru => "ru",
            Self::Rw => "rw",
            Self::Si => "si",
            Self::Sk => "sk",
            Self::Sl => "sl",
            Self::Sq => "sq",
            Self::Sr => "sr",
            Self::Su => "su",
            Self::Sv => "sv",
            Self::Sw => "sw",
            Self::sw_KE => "sw-KE",
            Self::sw_TZ => "sw-TZ",
            Self::Ta => "ta",
            Self::Te => "te",
            Self::Tg => "tg",
            Self::Th => "th",
            Self::Tr => "tr",
            Self::Uk => "uk",
            Self::Ur => "ur",
            Self::Uz => "uz",
            Self::Vi => "vi",
            Self::Yi => "yi",
            Self::Zh => "zh",
            Self::zh_CN => "zh-CN",
            Self::zh_HK => "zh-HK",
            Self::zh_TW => "zh-TW",
        }
    }

    /// Display name in English. Regional variants use a region-qualified label.
    pub fn full_name(self) -> &'static str {
        match self {
            Self::Af => "Afrikaans",
            Self::Am => "Amharic",
            Self::Ar => "Arabic",
            Self::ar_EG => "Egyptian Arabic",
            Self::ar_SA => "Saudi Arabic",
            Self::As => "Assamese",
            Self::Ast => "Asturian",
            Self::Az => "Azerbaijani",
            Self::Be => "Belarusian",
            Self::Bg => "Bulgarian",
            Self::Bn => "Bengali",
            Self::Bs => "Bosnian",
            Self::Ca => "Catalan",
            Self::Ceb => "Cebuano",
            Self::Ckb => "Central Kurdish",
            Self::Cs => "Czech",
            Self::Cy => "Welsh",
            Self::Da => "Danish",
            Self::De => "German",
            Self::El => "Greek",
            Self::En => "English",
            Self::en_GB => "British English",
            Self::en_US => "American English",
            Self::Es => "Spanish",
            Self::es_ES => "European Spanish",
            Self::es_MX => "Mexican Spanish",
            Self::Et => "Estonian",
            Self::Eu => "Basque",
            Self::Fa => "Persian",
            Self::Fi => "Finnish",
            Self::Fil => "Filipino",
            Self::Fr => "French",
            Self::fr_CA => "Canadian French",
            Self::fr_FR => "European French",
            Self::Gl => "Galician",
            Self::Gu => "Gujarati",
            Self::He => "Hebrew",
            Self::Hi => "Hindi",
            Self::Hr => "Croatian",
            Self::Hu => "Hungarian",
            Self::Hy => "Armenian",
            Self::Id => "Indonesian",
            Self::Is => "Icelandic",
            Self::It => "Italian",
            Self::Ja => "Japanese",
            Self::Ka => "Georgian",
            Self::Kk => "Kazakh",
            Self::Km => "Khmer",
            Self::Kn => "Kannada",
            Self::Ko => "Korean",
            Self::Ky => "Kyrgyz",
            Self::Lb => "Luxembourgish",
            Self::Lo => "Lao",
            Self::Lt => "Lithuanian",
            Self::Lv => "Latvian",
            Self::Mk => "Macedonian",
            Self::Ml => "Malayalam",
            Self::Mn => "Mongolian",
            Self::Mr => "Marathi",
            Self::Ms => "Malay",
            Self::My => "Burmese",
            Self::Ne => "Nepali",
            Self::Nl => "Dutch",
            Self::No => "Norwegian",
            Self::Oc => "Occitan",
            Self::Or => "Oriya",
            Self::Pl => "Polish",
            Self::Ps => "Pashto",
            Self::Pt => "Portuguese",
            Self::pt_BR => "Brazilian Portuguese",
            Self::pt_PT => "European Portuguese",
            Self::Ro => "Romanian",
            Self::Ru => "Russian",
            Self::Rw => "Kinyarwanda",
            Self::Si => "Sinhala",
            Self::Sk => "Slovak",
            Self::Sl => "Slovenian",
            Self::Sq => "Albanian",
            Self::Sr => "Serbian",
            Self::Su => "Sundanese",
            Self::Sv => "Swedish",
            Self::Sw => "Swahili",
            Self::sw_KE => "Kenyan Swahili",
            Self::sw_TZ => "Tanzanian Swahili",
            Self::Ta => "Tamil",
            Self::Te => "Telugu",
            Self::Tg => "Tajik",
            Self::Th => "Thai",
            Self::Tr => "Turkish",
            Self::Uk => "Ukrainian",
            Self::Ur => "Urdu",
            Self::Uz => "Uzbek",
            Self::Vi => "Vietnamese",
            Self::Yi => "Yiddish",
            Self::Zh => "Chinese",
            Self::zh_CN => "Simplified Chinese",
            Self::zh_HK => "Hong Kong Chinese",
            Self::zh_TW => "Traditional Chinese",
        }
    }

    /// Script group used for translation length estimation.
    /// 0 = CJK, 1 = Indic, 2 = RTL, 3 = Thai, 4 = Latin/Cyrillic.
    pub fn script_group(self) -> u8 {
        match self {
            Self::Zh | Self::zh_CN | Self::zh_HK | Self::zh_TW | Self::Ja | Self::Ko => 0,
            Self::Hi
            | Self::Bn
            | Self::Gu
            | Self::Kn
            | Self::Ml
            | Self::Mr
            | Self::Ne
            | Self::Si
            | Self::Ta
            | Self::Te => 1,
            Self::Ar
            | Self::ar_EG
            | Self::ar_SA
            | Self::Ckb
            | Self::Fa
            | Self::He
            | Self::Ps
            | Self::Ur
            | Self::Yi => 2,
            Self::Th => 3,
            _ => 4,
        }
    }

    /// All 98 supported translate-side languages, sorted by BCP 47 code.
    pub fn all() -> &'static [Language] {
        &ALL_LANGUAGES
    }
}

/// Estimate the token expansion ratio when translating between two languages.
///
/// Used to scale the expected output length for the EOS length bias.
pub fn expansion_ratio(src: Language, tgt: Language) -> f32 {
    let sg = src.script_group();
    let tg = tgt.script_group();
    if sg == tg {
        return 1.0;
    }
    match (sg, tg) {
        // CJK ↔ European
        (0, 4) => 1.5,
        (4, 0) => 0.55,
        // Indic ↔ European
        (1, 4) => 1.2,
        (4, 1) => 0.85,
        // RTL ↔ European
        (2, 4) => 1.1,
        (4, 2) => 0.9,
        // Thai ↔ European
        (3, 4) => 1.3,
        (4, 3) => 0.7,
        // CJK ↔ Indic
        (0, 1) => 1.3,
        (1, 0) => 0.7,
        // CJK ↔ RTL
        (0, 2) => 1.4,
        (2, 0) => 0.6,
        _ => 1.0,
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Lookup table for `FromStr`. Keys are lowercased BCP 47 codes with `-`
/// separators (the input normalizer converts `_` → `-` and lowercases).
fn lookup_exact(normalized: &str) -> Option<Language> {
    use Language::*;
    match normalized {
        // Base codes
        "af" => Some(Af),
        "am" => Some(Am),
        "ar" => Some(Ar),
        "as" => Some(As),
        "ast" => Some(Ast),
        "az" => Some(Az),
        "be" => Some(Be),
        "bg" => Some(Bg),
        "bn" => Some(Bn),
        "bs" => Some(Bs),
        "ca" => Some(Ca),
        "ceb" => Some(Ceb),
        "ckb" => Some(Ckb),
        "cs" => Some(Cs),
        "cy" => Some(Cy),
        "da" => Some(Da),
        "de" => Some(De),
        "el" => Some(El),
        "en" => Some(En),
        "es" => Some(Es),
        "et" => Some(Et),
        "eu" => Some(Eu),
        "fa" => Some(Fa),
        "fi" => Some(Fi),
        "fil" | "tl" => Some(Fil),
        "fr" => Some(Fr),
        "gl" => Some(Gl),
        "gu" => Some(Gu),
        "he" | "iw" => Some(He),
        "hi" => Some(Hi),
        "hr" => Some(Hr),
        "hu" => Some(Hu),
        "hy" => Some(Hy),
        "id" => Some(Id),
        "is" => Some(Is),
        "it" => Some(It),
        "ja" => Some(Ja),
        "ka" => Some(Ka),
        "kk" => Some(Kk),
        "km" => Some(Km),
        "kn" => Some(Kn),
        "ko" => Some(Ko),
        "ky" => Some(Ky),
        "lb" => Some(Lb),
        "lo" => Some(Lo),
        "lt" => Some(Lt),
        "lv" => Some(Lv),
        "mk" => Some(Mk),
        "ml" => Some(Ml),
        "mn" => Some(Mn),
        "mr" => Some(Mr),
        "ms" => Some(Ms),
        "my" => Some(My),
        "ne" => Some(Ne),
        "nl" => Some(Nl),
        "no" | "nb" | "nn" => Some(No),
        "oc" => Some(Oc),
        "or" => Some(Or),
        "pl" => Some(Pl),
        "ps" => Some(Ps),
        "pt" => Some(Pt),
        "ro" => Some(Ro),
        "ru" => Some(Ru),
        "rw" => Some(Rw),
        "si" => Some(Si),
        "sk" => Some(Sk),
        "sl" => Some(Sl),
        "sq" => Some(Sq),
        "sr" => Some(Sr),
        "su" => Some(Su),
        "sv" => Some(Sv),
        "sw" => Some(Sw),
        "ta" => Some(Ta),
        "te" => Some(Te),
        "tg" => Some(Tg),
        "th" => Some(Th),
        "tr" => Some(Tr),
        "uk" => Some(Uk),
        "ur" => Some(Ur),
        "uz" => Some(Uz),
        "vi" => Some(Vi),
        "yi" => Some(Yi),
        "zh" => Some(Zh),
        // Regional pairs (and BCP 47 script subtag aliases for Chinese)
        "ar-eg" => Some(ar_EG),
        "ar-sa" => Some(ar_SA),
        "en-gb" => Some(en_GB),
        "en-us" => Some(en_US),
        "es-es" => Some(es_ES),
        "es-mx" => Some(es_MX),
        "fr-ca" => Some(fr_CA),
        "fr-fr" => Some(fr_FR),
        "pt-br" => Some(pt_BR),
        "pt-pt" => Some(pt_PT),
        "sw-ke" => Some(sw_KE),
        "sw-tz" => Some(sw_TZ),
        "zh-cn" | "zh-hans" => Some(zh_CN),
        "zh-hk" => Some(zh_HK),
        "zh-tw" | "zh-hant" => Some(zh_TW),
        _ => None,
    }
}

impl FromStr for Language {
    type Err = TranslatorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized: String = s
            .chars()
            .map(|c| if c == '_' { '-' } else { c.to_ascii_lowercase() })
            .collect();

        if let Some(lang) = lookup_exact(&normalized) {
            return Ok(lang);
        }

        // Unknown region/script tag — fall back to the base language portion
        // (the part before the first `-`). e.g. `pt-AO` → `Pt`.
        if let Some((base, _)) = normalized.split_once('-')
            && let Some(lang) = lookup_exact(base)
        {
            tracing::debug!(input = %s, fallback = %lang.code(), "unknown region tag; falling back to base language");
            return Ok(lang);
        }

        Err(TranslatorError::UnsupportedLanguage(s.to_string()))
    }
}

impl Serialize for Language {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.code())
    }
}

impl<'de> Deserialize<'de> for Language {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LanguageVisitor;

        impl<'de> Visitor<'de> for LanguageVisitor {
            type Value = Language;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a BCP 47 language code (e.g. \"fr\", \"pt-BR\")")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Language, E> {
                v.parse::<Language>().map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(LanguageVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_98_languages() {
        assert_eq!(Language::all().len(), 98);
    }

    #[test]
    fn all_sorted_by_code() {
        let codes: Vec<&str> = Language::all().iter().map(|l| l.code()).collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        assert_eq!(codes, sorted);
    }

    #[test]
    fn roundtrip_code() {
        for &lang in Language::all() {
            let code = lang.code();
            let parsed: Language = code.parse().unwrap();
            assert_eq!(lang, parsed, "round-trip failed for {code}");
        }
    }

    #[test]
    fn display_is_code() {
        assert_eq!(Language::Fr.to_string(), "fr");
        assert_eq!(Language::pt_BR.to_string(), "pt-BR");
        assert_eq!(Language::zh_TW.to_string(), "zh-TW");
    }

    #[test]
    fn from_str_regional_dash_and_underscore() {
        assert_eq!("pt-BR".parse::<Language>().unwrap(), Language::pt_BR);
        assert_eq!("pt_BR".parse::<Language>().unwrap(), Language::pt_BR);
        assert_eq!("PT-br".parse::<Language>().unwrap(), Language::pt_BR);
        assert_eq!("PT_BR".parse::<Language>().unwrap(), Language::pt_BR);
        assert_eq!("pt-br".parse::<Language>().unwrap(), Language::pt_BR);
    }

    #[test]
    fn from_str_chinese_script_aliases() {
        assert_eq!("zh-Hans".parse::<Language>().unwrap(), Language::zh_CN);
        assert_eq!("zh-Hant".parse::<Language>().unwrap(), Language::zh_TW);
        assert_eq!("ZH_HANS".parse::<Language>().unwrap(), Language::zh_CN);
        assert_eq!("zh-CN".parse::<Language>().unwrap(), Language::zh_CN);
        assert_eq!("zh-TW".parse::<Language>().unwrap(), Language::zh_TW);
    }

    #[test]
    fn from_str_norwegian_aliases() {
        assert_eq!("nb".parse::<Language>().unwrap(), Language::No);
        assert_eq!("nn".parse::<Language>().unwrap(), Language::No);
        assert_eq!("no".parse::<Language>().unwrap(), Language::No);
    }

    #[test]
    fn from_str_filipino_alias() {
        assert_eq!("fil".parse::<Language>().unwrap(), Language::Fil);
        assert_eq!("tl".parse::<Language>().unwrap(), Language::Fil);
    }

    #[test]
    fn from_str_hebrew_alias() {
        assert_eq!("he".parse::<Language>().unwrap(), Language::He);
        assert_eq!("iw".parse::<Language>().unwrap(), Language::He);
    }

    #[test]
    fn from_str_unknown_region_falls_back_to_base() {
        // pt-AO is real CLDR but not in our enum — fall back to Pt.
        assert_eq!("pt-AO".parse::<Language>().unwrap(), Language::Pt);
        // en-AE / de-AT have no enum variant — fall back to base.
        assert_eq!("en-AE".parse::<Language>().unwrap(), Language::En);
        assert_eq!("de-AT".parse::<Language>().unwrap(), Language::De);
    }

    #[test]
    fn from_str_unknown_base_errors() {
        assert!("xx".parse::<Language>().is_err());
        assert!("xx-YY".parse::<Language>().is_err());
        assert!("".parse::<Language>().is_err());
    }

    #[test]
    fn serde_base_roundtrip() {
        let lang = Language::Fr;
        let json = serde_json::to_string(&lang).unwrap();
        assert_eq!(json, r#""fr""#);
        let parsed: Language = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, lang);
    }

    #[test]
    fn serde_regional_roundtrip() {
        let lang = Language::pt_BR;
        let json = serde_json::to_string(&lang).unwrap();
        assert_eq!(json, r#""pt-BR""#);
        let parsed: Language = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, lang);
    }

    #[test]
    fn serde_vec() {
        let langs = vec![Language::En, Language::pt_BR, Language::zh_TW];
        let json = serde_json::to_string(&langs).unwrap();
        assert_eq!(json, r#"["en","pt-BR","zh-TW"]"#);
        let parsed: Vec<Language> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, langs);
    }

    #[test]
    fn full_name_all_populated() {
        for &lang in Language::all() {
            assert!(!lang.full_name().is_empty(), "{} has empty full_name", lang.code());
        }
    }

    #[test]
    fn regional_full_names_distinguish_variants() {
        assert_eq!(Language::pt_BR.full_name(), "Brazilian Portuguese");
        assert_eq!(Language::pt_PT.full_name(), "European Portuguese");
        assert_eq!(Language::zh_CN.full_name(), "Simplified Chinese");
        assert_eq!(Language::zh_TW.full_name(), "Traditional Chinese");
    }

    #[test]
    fn expansion_ratio_same_script_is_one() {
        assert_eq!(expansion_ratio(Language::En, Language::Fr), 1.0);
        assert_eq!(expansion_ratio(Language::Zh, Language::Ja), 1.0);
        assert_eq!(expansion_ratio(Language::zh_CN, Language::zh_TW), 1.0);
        assert_eq!(expansion_ratio(Language::Hi, Language::Bn), 1.0);
        assert_eq!(expansion_ratio(Language::Ar, Language::ar_EG), 1.0);
    }

    #[test]
    fn expansion_ratio_cjk_to_european() {
        assert!(expansion_ratio(Language::Zh, Language::En) > 1.0);
        assert!(expansion_ratio(Language::zh_TW, Language::pt_BR) > 1.0);
    }

    #[test]
    fn script_group_coverage() {
        assert_eq!(Language::Zh.script_group(), 0);
        assert_eq!(Language::zh_CN.script_group(), 0);
        assert_eq!(Language::Hi.script_group(), 1);
        assert_eq!(Language::Ar.script_group(), 2);
        assert_eq!(Language::ar_EG.script_group(), 2);
        assert_eq!(Language::He.script_group(), 2);
        assert_eq!(Language::Th.script_group(), 3);
        assert_eq!(Language::En.script_group(), 4);
        assert_eq!(Language::pt_BR.script_group(), 4);
    }
}
