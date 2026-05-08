use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::error::TranslatorError;

/// Supported translate-side languages and locales.
///
/// 70 variants: 55 base ISO 639-1 codes + 4 base codes added for WMT24++
/// coverage (`He`, `Is`, `Fil`, `Zu`) + 11 regional pairs from WMT24++
/// (`ar_EG`, `ar_SA`, `es_MX`, `fr_CA`, `fr_FR`, `pt_BR`, `pt_PT`, `sw_KE`,
/// `sw_TZ`, `zh_CN`, `zh_TW`).
///
/// The detector returns a `String` (BCP 47 code) and may produce codes outside
/// this enum (lingua-only base languages such as `cy`, `ka`, `eu`; script
/// subtags such as `sr-Cyrl`, `pa-Guru`). `FromStr` falls back to the base
/// language when an unknown region tag is supplied; codes with no recognized
/// base error out.
///
/// Variant naming: regional variants use `lowercase + _ + UPPERCASE` so the
/// source reads as the BCP 47 code at a glance (`pt_BR`). Base variants stay
/// PascalCase (`Pt`).
///
/// 8 base codes (`Af`, `Am`, `Ha`, `Ms`, `Mt`, `Ne`, `Si`, `Yi`) are kept
/// outside the WMT24++ training distribution as best-effort. Gemma 3 instruct
/// generalizes broadly, but quality on these is not guaranteed.
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
    Bg,
    Bn,
    Ca,
    Cs,
    Da,
    De,
    El,
    En,
    Es,
    es_MX,
    Et,
    Fa,
    Fi,
    Fil,
    Fr,
    fr_CA,
    fr_FR,
    Gu,
    Ha,
    He,
    Hi,
    Hr,
    Hu,
    Id,
    Is,
    It,
    Ja,
    Kn,
    Ko,
    Lt,
    Lv,
    Ml,
    Mr,
    Ms,
    Mt,
    Ne,
    Nl,
    No,
    Pa,
    Pl,
    Pt,
    pt_BR,
    pt_PT,
    Ro,
    Ru,
    Si,
    Sk,
    Sl,
    Sr,
    Sv,
    Sw,
    sw_KE,
    sw_TZ,
    Ta,
    Te,
    Th,
    Tr,
    Uk,
    Ur,
    Vi,
    Yi,
    Zh,
    zh_CN,
    zh_TW,
    Zu,
}

static ALL_LANGUAGES: [Language; 70] = [
    Language::Af,
    Language::Am,
    Language::Ar,
    Language::ar_EG,
    Language::ar_SA,
    Language::Bg,
    Language::Bn,
    Language::Ca,
    Language::Cs,
    Language::Da,
    Language::De,
    Language::El,
    Language::En,
    Language::Es,
    Language::es_MX,
    Language::Et,
    Language::Fa,
    Language::Fi,
    Language::Fil,
    Language::Fr,
    Language::fr_CA,
    Language::fr_FR,
    Language::Gu,
    Language::Ha,
    Language::He,
    Language::Hi,
    Language::Hr,
    Language::Hu,
    Language::Id,
    Language::Is,
    Language::It,
    Language::Ja,
    Language::Kn,
    Language::Ko,
    Language::Lt,
    Language::Lv,
    Language::Ml,
    Language::Mr,
    Language::Ms,
    Language::Mt,
    Language::Ne,
    Language::Nl,
    Language::No,
    Language::Pa,
    Language::Pl,
    Language::Pt,
    Language::pt_BR,
    Language::pt_PT,
    Language::Ro,
    Language::Ru,
    Language::Si,
    Language::Sk,
    Language::Sl,
    Language::Sr,
    Language::Sv,
    Language::Sw,
    Language::sw_KE,
    Language::sw_TZ,
    Language::Ta,
    Language::Te,
    Language::Th,
    Language::Tr,
    Language::Uk,
    Language::Ur,
    Language::Vi,
    Language::Yi,
    Language::Zh,
    Language::zh_CN,
    Language::zh_TW,
    Language::Zu,
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
            Self::Bg => "bg",
            Self::Bn => "bn",
            Self::Ca => "ca",
            Self::Cs => "cs",
            Self::Da => "da",
            Self::De => "de",
            Self::El => "el",
            Self::En => "en",
            Self::Es => "es",
            Self::es_MX => "es-MX",
            Self::Et => "et",
            Self::Fa => "fa",
            Self::Fi => "fi",
            Self::Fil => "fil",
            Self::Fr => "fr",
            Self::fr_CA => "fr-CA",
            Self::fr_FR => "fr-FR",
            Self::Gu => "gu",
            Self::Ha => "ha",
            Self::He => "he",
            Self::Hi => "hi",
            Self::Hr => "hr",
            Self::Hu => "hu",
            Self::Id => "id",
            Self::Is => "is",
            Self::It => "it",
            Self::Ja => "ja",
            Self::Kn => "kn",
            Self::Ko => "ko",
            Self::Lt => "lt",
            Self::Lv => "lv",
            Self::Ml => "ml",
            Self::Mr => "mr",
            Self::Ms => "ms",
            Self::Mt => "mt",
            Self::Ne => "ne",
            Self::Nl => "nl",
            Self::No => "no",
            Self::Pa => "pa",
            Self::Pl => "pl",
            Self::Pt => "pt",
            Self::pt_BR => "pt-BR",
            Self::pt_PT => "pt-PT",
            Self::Ro => "ro",
            Self::Ru => "ru",
            Self::Si => "si",
            Self::Sk => "sk",
            Self::Sl => "sl",
            Self::Sr => "sr",
            Self::Sv => "sv",
            Self::Sw => "sw",
            Self::sw_KE => "sw-KE",
            Self::sw_TZ => "sw-TZ",
            Self::Ta => "ta",
            Self::Te => "te",
            Self::Th => "th",
            Self::Tr => "tr",
            Self::Uk => "uk",
            Self::Ur => "ur",
            Self::Vi => "vi",
            Self::Yi => "yi",
            Self::Zh => "zh",
            Self::zh_CN => "zh-CN",
            Self::zh_TW => "zh-TW",
            Self::Zu => "zu",
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
            Self::Bg => "Bulgarian",
            Self::Bn => "Bengali",
            Self::Ca => "Catalan",
            Self::Cs => "Czech",
            Self::Da => "Danish",
            Self::De => "German",
            Self::El => "Greek",
            Self::En => "English",
            Self::Es => "Spanish",
            Self::es_MX => "Mexican Spanish",
            Self::Et => "Estonian",
            Self::Fa => "Persian",
            Self::Fi => "Finnish",
            Self::Fil => "Filipino",
            Self::Fr => "French",
            Self::fr_CA => "Canadian French",
            Self::fr_FR => "European French",
            Self::Gu => "Gujarati",
            Self::Ha => "Hausa",
            Self::He => "Hebrew",
            Self::Hi => "Hindi",
            Self::Hr => "Croatian",
            Self::Hu => "Hungarian",
            Self::Id => "Indonesian",
            Self::Is => "Icelandic",
            Self::It => "Italian",
            Self::Ja => "Japanese",
            Self::Kn => "Kannada",
            Self::Ko => "Korean",
            Self::Lt => "Lithuanian",
            Self::Lv => "Latvian",
            Self::Ml => "Malayalam",
            Self::Mr => "Marathi",
            Self::Ms => "Malay",
            Self::Mt => "Maltese",
            Self::Ne => "Nepali",
            Self::Nl => "Dutch",
            Self::No => "Norwegian",
            Self::Pa => "Punjabi",
            Self::Pl => "Polish",
            Self::Pt => "Portuguese",
            Self::pt_BR => "Brazilian Portuguese",
            Self::pt_PT => "European Portuguese",
            Self::Ro => "Romanian",
            Self::Ru => "Russian",
            Self::Si => "Sinhala",
            Self::Sk => "Slovak",
            Self::Sl => "Slovenian",
            Self::Sr => "Serbian",
            Self::Sv => "Swedish",
            Self::Sw => "Swahili",
            Self::sw_KE => "Kenyan Swahili",
            Self::sw_TZ => "Tanzanian Swahili",
            Self::Ta => "Tamil",
            Self::Te => "Telugu",
            Self::Th => "Thai",
            Self::Tr => "Turkish",
            Self::Uk => "Ukrainian",
            Self::Ur => "Urdu",
            Self::Vi => "Vietnamese",
            Self::Yi => "Yiddish",
            Self::Zh => "Chinese",
            Self::zh_CN => "Simplified Chinese",
            Self::zh_TW => "Traditional Chinese",
            Self::Zu => "Zulu",
        }
    }

    /// Script group used for translation length estimation.
    /// 0 = CJK, 1 = Indic, 2 = RTL, 3 = Thai, 4 = Latin/Cyrillic.
    pub fn script_group(self) -> u8 {
        match self {
            Self::Zh | Self::zh_CN | Self::zh_TW | Self::Ja | Self::Ko => 0,
            Self::Hi
            | Self::Bn
            | Self::Gu
            | Self::Kn
            | Self::Ml
            | Self::Mr
            | Self::Ne
            | Self::Pa
            | Self::Si
            | Self::Ta
            | Self::Te => 1,
            Self::Ar | Self::ar_EG | Self::ar_SA | Self::Fa | Self::He | Self::Ur | Self::Yi => 2,
            Self::Th => 3,
            _ => 4,
        }
    }

    /// All 70 supported translate-side languages, sorted by BCP 47 code.
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
        "bg" => Some(Bg),
        "bn" => Some(Bn),
        "ca" => Some(Ca),
        "cs" => Some(Cs),
        "da" => Some(Da),
        "de" => Some(De),
        "el" => Some(El),
        "en" => Some(En),
        "es" => Some(Es),
        "et" => Some(Et),
        "fa" => Some(Fa),
        "fi" => Some(Fi),
        "fil" | "tl" => Some(Fil),
        "fr" => Some(Fr),
        "gu" => Some(Gu),
        "ha" => Some(Ha),
        "he" | "iw" => Some(He),
        "hi" => Some(Hi),
        "hr" => Some(Hr),
        "hu" => Some(Hu),
        "id" => Some(Id),
        "is" => Some(Is),
        "it" => Some(It),
        "ja" => Some(Ja),
        "kn" => Some(Kn),
        "ko" => Some(Ko),
        "lt" => Some(Lt),
        "lv" => Some(Lv),
        "ml" => Some(Ml),
        "mr" => Some(Mr),
        "ms" => Some(Ms),
        "mt" => Some(Mt),
        "ne" => Some(Ne),
        "nl" => Some(Nl),
        "no" | "nb" | "nn" => Some(No),
        "pa" => Some(Pa),
        "pl" => Some(Pl),
        "pt" => Some(Pt),
        "ro" => Some(Ro),
        "ru" => Some(Ru),
        "si" => Some(Si),
        "sk" => Some(Sk),
        "sl" => Some(Sl),
        "sr" => Some(Sr),
        "sv" => Some(Sv),
        "sw" => Some(Sw),
        "ta" => Some(Ta),
        "te" => Some(Te),
        "th" => Some(Th),
        "tr" => Some(Tr),
        "uk" => Some(Uk),
        "ur" => Some(Ur),
        "vi" => Some(Vi),
        "yi" => Some(Yi),
        "zh" => Some(Zh),
        "zu" => Some(Zu),
        // Regional pairs (and BCP 47 script subtag aliases for Chinese)
        "ar-eg" => Some(ar_EG),
        "ar-sa" => Some(ar_SA),
        "es-mx" => Some(es_MX),
        "fr-ca" => Some(fr_CA),
        "fr-fr" => Some(fr_FR),
        "pt-br" => Some(pt_BR),
        "pt-pt" => Some(pt_PT),
        "sw-ke" => Some(sw_KE),
        "sw-tz" => Some(sw_TZ),
        "zh-cn" | "zh-hans" => Some(zh_CN),
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
    fn all_70_languages() {
        assert_eq!(Language::all().len(), 70);
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
        // pt-AO is real CLDR but not a WMT24++ pair — fall back to Pt.
        assert_eq!("pt-AO".parse::<Language>().unwrap(), Language::Pt);
        // en-GB has no enum variant — fall back to En.
        assert_eq!("en-GB".parse::<Language>().unwrap(), Language::En);
        assert_eq!("en_US".parse::<Language>().unwrap(), Language::En);
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
