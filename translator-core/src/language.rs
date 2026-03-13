use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::error::TranslatorError;

/// All 55 languages supported by TranslateGemma 4B.
///
/// Implements `Copy + Eq + Hash`. Serializes as the ISO 639-1 code string
/// (e.g. `"fr"`) for JSON backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Af,
    Am,
    Ar,
    Bg,
    Bn,
    Ca,
    Cs,
    Da,
    De,
    El,
    En,
    Es,
    Et,
    Fa,
    Fi,
    Fr,
    Gu,
    Ha,
    Hi,
    Hr,
    Hu,
    Id,
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
    Ro,
    Ru,
    Si,
    Sk,
    Sl,
    Sr,
    Sv,
    Sw,
    Ta,
    Te,
    Th,
    Tr,
    Uk,
    Ur,
    Vi,
    Yi,
    Zh,
}

static ALL_LANGUAGES: [Language; 55] = [
    Language::Af,
    Language::Am,
    Language::Ar,
    Language::Bg,
    Language::Bn,
    Language::Ca,
    Language::Cs,
    Language::Da,
    Language::De,
    Language::El,
    Language::En,
    Language::Es,
    Language::Et,
    Language::Fa,
    Language::Fi,
    Language::Fr,
    Language::Gu,
    Language::Ha,
    Language::Hi,
    Language::Hr,
    Language::Hu,
    Language::Id,
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
    Language::Ro,
    Language::Ru,
    Language::Si,
    Language::Sk,
    Language::Sl,
    Language::Sr,
    Language::Sv,
    Language::Sw,
    Language::Ta,
    Language::Te,
    Language::Th,
    Language::Tr,
    Language::Uk,
    Language::Ur,
    Language::Vi,
    Language::Yi,
    Language::Zh,
];

impl Language {
    /// ISO 639-1 code (e.g. `"fr"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::Af => "af",
            Self::Am => "am",
            Self::Ar => "ar",
            Self::Bg => "bg",
            Self::Bn => "bn",
            Self::Ca => "ca",
            Self::Cs => "cs",
            Self::Da => "da",
            Self::De => "de",
            Self::El => "el",
            Self::En => "en",
            Self::Es => "es",
            Self::Et => "et",
            Self::Fa => "fa",
            Self::Fi => "fi",
            Self::Fr => "fr",
            Self::Gu => "gu",
            Self::Ha => "ha",
            Self::Hi => "hi",
            Self::Hr => "hr",
            Self::Hu => "hu",
            Self::Id => "id",
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
            Self::Ro => "ro",
            Self::Ru => "ru",
            Self::Si => "si",
            Self::Sk => "sk",
            Self::Sl => "sl",
            Self::Sr => "sr",
            Self::Sv => "sv",
            Self::Sw => "sw",
            Self::Ta => "ta",
            Self::Te => "te",
            Self::Th => "th",
            Self::Tr => "tr",
            Self::Uk => "uk",
            Self::Ur => "ur",
            Self::Vi => "vi",
            Self::Yi => "yi",
            Self::Zh => "zh",
        }
    }

    /// Full English name used in translation prompts.
    pub fn full_name(self) -> &'static str {
        match self {
            Self::Af => "Afrikaans",
            Self::Am => "Amharic",
            Self::Ar => "Arabic",
            Self::Bg => "Bulgarian",
            Self::Bn => "Bengali",
            Self::Ca => "Catalan",
            Self::Cs => "Czech",
            Self::Da => "Danish",
            Self::De => "German",
            Self::El => "Greek",
            Self::En => "English",
            Self::Es => "Spanish",
            Self::Et => "Estonian",
            Self::Fa => "Persian",
            Self::Fi => "Finnish",
            Self::Fr => "French",
            Self::Gu => "Gujarati",
            Self::Ha => "Hausa",
            Self::Hi => "Hindi",
            Self::Hr => "Croatian",
            Self::Hu => "Hungarian",
            Self::Id => "Indonesian",
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
            Self::Ro => "Romanian",
            Self::Ru => "Russian",
            Self::Si => "Sinhala",
            Self::Sk => "Slovak",
            Self::Sl => "Slovenian",
            Self::Sr => "Serbian",
            Self::Sv => "Swedish",
            Self::Sw => "Swahili",
            Self::Ta => "Tamil",
            Self::Te => "Telugu",
            Self::Th => "Thai",
            Self::Tr => "Turkish",
            Self::Uk => "Ukrainian",
            Self::Ur => "Urdu",
            Self::Vi => "Vietnamese",
            Self::Yi => "Yiddish",
            Self::Zh => "Chinese",
        }
    }

    /// Script group for length estimation.
    /// 0 = CJK, 1 = Indic, 2 = RTL, 3 = Thai, 4 = Latin/Cyrillic.
    pub fn script_group(self) -> u8 {
        match self {
            Self::Zh | Self::Ja | Self::Ko => 0,
            Self::Hi | Self::Bn | Self::Gu | Self::Kn | Self::Ml | Self::Mr | Self::Ne
            | Self::Pa | Self::Si | Self::Ta | Self::Te => 1,
            Self::Ar | Self::Fa | Self::Ur | Self::Yi => 2,
            Self::Th => 3,
            _ => 4,
        }
    }

    /// All 55 supported languages, sorted by ISO code.
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

impl FromStr for Language {
    type Err = TranslatorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Normalize: lowercase, handle regional codes.
        let lower = s.to_lowercase();
        let normalized = match lower.as_str() {
            "zh-hk" | "zh-cn" | "zh-tw" => "zh",
            "fr-ca" => "fr",
            "es-mx" => "es",
            "pt-br" | "pt-pt" => "pt",
            "nb" | "nn" => "no",
            other => other,
        };
        match normalized {
            "af" => Ok(Self::Af),
            "am" => Ok(Self::Am),
            "ar" => Ok(Self::Ar),
            "bg" => Ok(Self::Bg),
            "bn" => Ok(Self::Bn),
            "ca" => Ok(Self::Ca),
            "cs" => Ok(Self::Cs),
            "da" => Ok(Self::Da),
            "de" => Ok(Self::De),
            "el" => Ok(Self::El),
            "en" => Ok(Self::En),
            "es" => Ok(Self::Es),
            "et" => Ok(Self::Et),
            "fa" => Ok(Self::Fa),
            "fi" => Ok(Self::Fi),
            "fr" => Ok(Self::Fr),
            "gu" => Ok(Self::Gu),
            "ha" => Ok(Self::Ha),
            "hi" => Ok(Self::Hi),
            "hr" => Ok(Self::Hr),
            "hu" => Ok(Self::Hu),
            "id" => Ok(Self::Id),
            "it" => Ok(Self::It),
            "ja" => Ok(Self::Ja),
            "kn" => Ok(Self::Kn),
            "ko" => Ok(Self::Ko),
            "lt" => Ok(Self::Lt),
            "lv" => Ok(Self::Lv),
            "ml" => Ok(Self::Ml),
            "mr" => Ok(Self::Mr),
            "ms" => Ok(Self::Ms),
            "mt" => Ok(Self::Mt),
            "ne" => Ok(Self::Ne),
            "nl" => Ok(Self::Nl),
            "no" => Ok(Self::No),
            "pa" => Ok(Self::Pa),
            "pl" => Ok(Self::Pl),
            "pt" => Ok(Self::Pt),
            "ro" => Ok(Self::Ro),
            "ru" => Ok(Self::Ru),
            "si" => Ok(Self::Si),
            "sk" => Ok(Self::Sk),
            "sl" => Ok(Self::Sl),
            "sr" => Ok(Self::Sr),
            "sv" => Ok(Self::Sv),
            "sw" => Ok(Self::Sw),
            "ta" => Ok(Self::Ta),
            "te" => Ok(Self::Te),
            "th" => Ok(Self::Th),
            "tr" => Ok(Self::Tr),
            "uk" => Ok(Self::Uk),
            "ur" => Ok(Self::Ur),
            "vi" => Ok(Self::Vi),
            "yi" => Ok(Self::Yi),
            "zh" => Ok(Self::Zh),
            _ => Err(TranslatorError::UnsupportedLanguage(s.to_string())),
        }
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
                formatter.write_str("an ISO 639-1 language code")
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
    fn all_55_languages() {
        assert_eq!(Language::all().len(), 55);
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
            assert_eq!(lang, parsed);
        }
    }

    #[test]
    fn display_is_code() {
        assert_eq!(Language::Fr.to_string(), "fr");
        assert_eq!(Language::Zh.to_string(), "zh");
    }

    #[test]
    fn regional_normalization() {
        assert_eq!("zh-cn".parse::<Language>().unwrap(), Language::Zh);
        assert_eq!("zh-tw".parse::<Language>().unwrap(), Language::Zh);
        assert_eq!("fr-ca".parse::<Language>().unwrap(), Language::Fr);
        assert_eq!("es-mx".parse::<Language>().unwrap(), Language::Es);
        assert_eq!("pt-br".parse::<Language>().unwrap(), Language::Pt);
        assert_eq!("nb".parse::<Language>().unwrap(), Language::No);
        assert_eq!("nn".parse::<Language>().unwrap(), Language::No);
    }

    #[test]
    fn case_insensitive_parse() {
        assert_eq!("FR".parse::<Language>().unwrap(), Language::Fr);
        assert_eq!("ZH-CN".parse::<Language>().unwrap(), Language::Zh);
    }

    #[test]
    fn unsupported_language_error() {
        assert!("xx".parse::<Language>().is_err());
        assert!("".parse::<Language>().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let lang = Language::Fr;
        let json = serde_json::to_string(&lang).unwrap();
        assert_eq!(json, r#""fr""#);
        let parsed: Language = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, lang);
    }

    #[test]
    fn serde_vec() {
        let langs = vec![Language::En, Language::Fr, Language::Ja];
        let json = serde_json::to_string(&langs).unwrap();
        assert_eq!(json, r#"["en","fr","ja"]"#);
        let parsed: Vec<Language> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, langs);
    }

    #[test]
    fn expansion_ratio_same_script_is_one() {
        assert_eq!(expansion_ratio(Language::En, Language::Fr), 1.0);
        assert_eq!(expansion_ratio(Language::Zh, Language::Ja), 1.0);
        assert_eq!(expansion_ratio(Language::Hi, Language::Bn), 1.0);
        assert_eq!(expansion_ratio(Language::Ar, Language::Fa), 1.0);
    }

    #[test]
    fn expansion_ratio_cjk_to_european() {
        assert!(expansion_ratio(Language::Zh, Language::En) > 1.0);
        assert!(expansion_ratio(Language::Ja, Language::Fr) > 1.0);
    }

    #[test]
    fn expansion_ratio_european_to_cjk() {
        assert!(expansion_ratio(Language::En, Language::Zh) < 1.0);
        assert!(expansion_ratio(Language::Fr, Language::Ja) < 1.0);
    }

    #[test]
    fn expansion_ratio_symmetry() {
        let forward = expansion_ratio(Language::Zh, Language::En);
        let backward = expansion_ratio(Language::En, Language::Zh);
        assert!(forward > 1.0);
        assert!(backward < 1.0);
    }

    #[test]
    fn script_group_coverage() {
        assert_eq!(Language::Zh.script_group(), 0);
        assert_eq!(Language::Hi.script_group(), 1);
        assert_eq!(Language::Ar.script_group(), 2);
        assert_eq!(Language::Th.script_group(), 3);
        assert_eq!(Language::En.script_group(), 4);
        assert_eq!(Language::De.script_group(), 4);
    }

    #[test]
    fn full_name_all_populated() {
        for &lang in Language::all() {
            assert!(!lang.full_name().is_empty(), "{} has empty full_name", lang.code());
        }
    }
}
