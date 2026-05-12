//! Heuristic regional dialect disambiguation.
//!
//! Layered on top of lingua's base-language detection, this module refines a
//! base code (e.g. `pt`) to a regional BCP 47 code (e.g. `pt-BR` or `pt-PT`)
//! when the input text contains enough region-specific marker words to commit.
//!
//! **Best-effort, not authoritative.** Same-script regional variants share most
//! of their vocabulary, so most short or neutral text yields no markers and
//! returns the base code unchanged. False positives are possible on adversarial
//! input. The heuristic never overrides lingua's base-language decision; it
//! only ever *refines* the suffix.
//!
//! Streaming + early-return: marker scanning aborts as soon as one side has
//! enough lead to commit, so cost stays bounded on large inputs.
//!
//! Pairs not handled here (intentionally):
//! - `ar-EG` / `ar-SA`: Modern Standard Arabic in writing is regionally
//!   uniform; signal too weak for a heuristic.
//! - `sw-KE` / `sw-TZ`: lexical differences are minor.

use std::sync::OnceLock;

use aho_corasick::AhoCorasick;

/// Refine a base language code into a regional variant when markers commit.
/// Returns `Some(refined_code)` on commit, `None` otherwise (caller keeps base).
///
/// The `base` argument is the input to disambiguate from — typically a base
/// language code (`pt`, `en`, `fr`, `es`), but may also be a script-refined
/// code (`zh-TW`) to chain a further dialect step on top of script
/// disambiguation.
pub fn disambiguate(base: &str, text: &str) -> Option<&'static str> {
    let pair = match base {
        "pt" => pt_pair(),
        "en" => en_pair(),
        "fr" => fr_pair(),
        "es" => es_pair(),
        "zh-TW" => zh_tw_pair(),
        "hi" => hi_ne_pair(),
        _ => return None,
    };
    pair.run(text)
}

// Commit when one side has at least this many hits and beats the other by at
// least this margin. Conservative thresholds keep false-positive rate low.
const COMMIT_MIN_HITS: u32 = 2;
const COMMIT_MARGIN: u32 = 2;

#[derive(Copy, Clone)]
enum Side {
    A,
    B,
}

struct Pair {
    matcher: AhoCorasick,
    sides: Vec<Side>,
    code_a: &'static str,
    code_b: &'static str,
}

impl Pair {
    fn build(
        code_a: &'static str,
        markers_a: &[&str],
        code_b: &'static str,
        markers_b: &[&str],
    ) -> Self {
        let mut patterns: Vec<&str> = Vec::with_capacity(markers_a.len() + markers_b.len());
        let mut sides: Vec<Side> = Vec::with_capacity(markers_a.len() + markers_b.len());
        for m in markers_a {
            patterns.push(m);
            sides.push(Side::A);
        }
        for m in markers_b {
            patterns.push(m);
            sides.push(Side::B);
        }
        let matcher = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("dialect markers must compile");
        Self { matcher, sides, code_a, code_b }
    }

    fn run(&self, text: &str) -> Option<&'static str> {
        let bytes = text.as_bytes();
        let mut hits_a: u32 = 0;
        let mut hits_b: u32 = 0;

        for mat in self.matcher.find_iter(text) {
            // Word-boundary check: surrounding byte must not be ASCII alphanumeric.
            // High-bit (UTF-8 multibyte) bytes are treated as boundaries here —
            // a small false-positive risk near accented characters, accepted as
            // simplification. Most markers occur after spaces in real text.
            let start = mat.start();
            let end = mat.end();
            let prev_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
            let next_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
            if !prev_ok || !next_ok {
                continue;
            }

            match self.sides[mat.pattern().as_usize()] {
                Side::A => {
                    hits_a += 1;
                    if commit(hits_a, hits_b) {
                        return Some(self.code_a);
                    }
                }
                Side::B => {
                    hits_b += 1;
                    if commit(hits_b, hits_a) {
                        return Some(self.code_b);
                    }
                }
            }
        }
        None
    }
}

fn commit(winner: u32, loser: u32) -> bool {
    winner >= COMMIT_MIN_HITS && winner.saturating_sub(loser) >= COMMIT_MARGIN
}

// ── Markers ──────────────────────────────────────────────────────────────────
// Each marker is a high-precision word/phrase that's strongly biased to one
// region. Ambiguous terms (e.g. "tu", "lift", "metro") are deliberately
// excluded to keep false-positive rate low.

fn pt_pair() -> &'static Pair {
    static P: OnceLock<Pair> = OnceLock::new();
    P.get_or_init(|| {
        let br: &[&str] = &[
            "você", "vocês", "ônibus", "trem", "geladeira", "celular",
            "café da manhã", "bonde", "metrô", "esporte", "esportes",
            "banheiro", "abacaxi", "carro", // 'carro' BR-leaning vs PT 'automóvel'
        ];
        let pt: &[&str] = &[
            "autocarro", "comboio", "frigorífico", "telemóvel",
            "pequeno-almoço", "eléctrico", "desporto", "casa de banho",
            "ananás", "fixe", // 'fixe' = cool, PT slang
            "rapariga", // PT for girl (BR: 'menina'); rapariga is offensive in BR
        ];
        Pair::build("pt-BR", br, "pt-PT", pt)
    })
}

fn en_pair() -> &'static Pair {
    static P: OnceLock<Pair> = OnceLock::new();
    P.get_or_init(|| {
        // US markers — distinctive spellings and lexical choices.
        let us: &[&str] = &[
            "color", "behavior", "favorite", "neighbor", "honor",
            "flavor", "labor", "humor",
            "organize", "recognize", "analyze", "criticize", "realize",
            "center", "theater", "meter",
            "apartment", "elevator", "truck", "trash",
            "cookie", "diaper", "sidewalk", "vacation",
            "movie", "soccer", "gasoline", "pants", // pants = US trousers
            "fall", // autumn (US) — accept some false positives in exchange for signal
        ];
        // GB markers — distinctive spellings and lexical choices.
        let gb: &[&str] = &[
            "colour", "behaviour", "favourite", "neighbour", "honour",
            "flavour", "labour", "humour",
            "organise", "recognise", "analyse", "criticise", "realise",
            "centre", "theatre", "metre",
            "lorry", "petrol", "biscuit", "rubbish",
            "whilst", "amongst", "autumn", "holiday", // holiday = vacation
            "trousers", "nappy", "pavement",
        ];
        Pair::build("en-US", us, "en-GB", gb)
    })
}

fn fr_pair() -> &'static Pair {
    static P: OnceLock<Pair> = OnceLock::new();
    P.get_or_init(|| {
        // Quebec/Canadian markers — careful: most written French is shared.
        let ca: &[&str] = &[
            "courriel", // CA email (FR uses 'mail' or 'e-mail')
            "fin de semaine", // CA (FR: 'week-end')
            "magasiner", // CA verb to shop (FR: 'faire les courses')
            "dépanneur", // CA convenience store
            "stationnement", // CA (FR also uses 'parking')
            "breuvage", // CA beverage (FR: 'boisson')
            "présentement", // CA 'currently' (FR: 'actuellement')
        ];
        // European French markers.
        let fr: &[&str] = &[
            "week-end", // FR (CA: 'fin de semaine')
            "shopping", // FR loanword
            "parking", // FR (also CA but CA preferentially uses stationnement)
            "footing", // FR jogging
            "smoking", // FR tuxedo
            "baskets", // FR sneakers (CA: 'espadrilles')
        ];
        Pair::build("fr-CA", ca, "fr-FR", fr)
    })
}

fn es_pair() -> &'static Pair {
    static P: OnceLock<Pair> = OnceLock::new();
    P.get_or_init(|| {
        // Mexican / Latin-American Spanish markers.
        let mx: &[&str] = &[
            "carro",       // car (Spain: coche)
            "computadora", // computer (Spain: ordenador)
            "celular",     // cell phone (Spain: móvil)
            "manejar",     // to drive (Spain: conducir)
            "platicar",    // to chat (Spain: charlar)
            "rentar",      // to rent (Spain: alquilar)
            "papa",        // potato (Spain: patata)
            "jugo",        // juice (Spain: zumo)
            "frijoles",    // beans (Spain: alubias)
            "chévere",     // cool (LA)
            "padrísimo",   // awesome (Mexican)
            "qué onda",    // what's up (Mexican)
        ];
        // European (Castilian) Spanish markers.
        let es: &[&str] = &[
            "vosotros",  // 2nd plural informal — Spain only
            "vosotras",
            "vuestro",   // 2nd plural possessive — Spain only
            "vuestra",
            "ordenador", // computer
            "móvil",     // mobile phone
            "patata",    // potato
            "zumo",      // juice
            "alubias",   // beans
            "tío",       // dude (Spain colloquial)
            "tía",       // gal (Spain colloquial)
            "vale",      // OK
            "guay",      // cool
            "ostras",    // wow / oh
        ];
        Pair::build("es-MX", mx, "es-ES", es)
    })
}

/// Hindi vs Nepali — both Devanagari, lingua tends to default short text to `hi`.
/// Refines `hi` → `ne` when Nepali-specific copula / verb forms / day names appear.
/// Symmetric: strong Hindi markers commit `hi` (no change); strong Nepali markers
/// commit `ne`. Neutral text returns `None` and keeps the base `hi`.
fn hi_ne_pair() -> &'static Pair {
    static P: OnceLock<Pair> = OnceLock::new();
    P.get_or_init(|| {
        // Hindi-distinctive: copula, question word, day names, future forms,
        // vocabulary that differs from Nepali.
        let hi: &[&str] = &[
            "हैं",         // plural copula (Nepali: छन्)
            " है",          // singular copula (Nepali: छ)
            "क्या",        // question word (Nepali: के)
            "की ",          // feminine genitive (Nepali uses को universally)
            "मंगलवार",    // Tuesday (Nepali: मङ्गलबार)
            "गुरुवार",    // Thursday (Nepali: बिहीबार)
            "बुधवार",     // Wednesday
            "होगा",        // future masc (Nepali: हुनेछ)
            "होगी",        // future fem
            "होंगे",       // future plural
            "के लिए",     // "for" (Nepali: को लागि)
            "नया ",         // new (Nepali: नयाँ)
            "अगला",        // next (Nepali: अर्को)
            "बड़ा",         // big (Nepali: ठूलो)
            "छोटा",        // small (Nepali: सानो)
            "लाल",          // red (Nepali: रातो)
            "रहा है",      // continuous Hindi
            "रहे हैं",     // continuous plural Hindi
            "रहा था",      // past continuous
            "करते हैं",    // habitual present plural
            "सकते हैं",    // modal plural
            "पहुँच",        // arrive (Nepali: आइपुग)
        ];
        // Nepali-distinctive: copula छ family, verb morphology, distinctive
        // vocabulary, polite pronouns. Multiple high-frequency forms so the
        // commit threshold (≥2 hits, ≥2 margin) fires on typical sentences.
        let ne: &[&str] = &[
            // copula / aux
            "छन्",          // plural copula
            "हुनेछ",        // future copula
            "हुन्छ",        // present
            "हुनुहोस्",    // polite imperative copula
            "हुनुपर्छ",   // modal "must be"
            "हो ",           // standalone "is" (Nepali)
            "थिए",          // past plural copula
            "थियो",         // past singular
            "होइन",         // "is not"
            // verb morphology
            "गर्नुहोस्",   // polite imperative
            "गर्नुपर्छ",  // modal "must do"
            "सक्नुहुन्छ",  // polite "can"
            "गरिने",        // passive participle / gerund
            "गरिएको",      // past passive
            "गरिए",         // past passive short
            "गर्नु",        // infinitive
            "गर्ने",        // gerund
            "गर्दछ",       // habitual present
            "गरे",          // past
            "गरेको",       // past habitual
            "गरेका",       // past habitual plural
            "गरेर",         // absolutive
            "भएको",        // past participle
            "भएका",        // past participle plural
            "भएकी",        // past participle fem
            "रहेको",       // past habitual
            "रहेका",       // past habitual plural
            "रहेछ",         // hearsay
            "आइपुग्यो",    // past arrive (Nepali; Hindi uses पहुँच)
            "आउँछ",         // present "comes"
            "जान्छ",        // present "goes"
            // pronouns / possessives
            "तपाईं",        // polite "you"
            "तपाईंले",     // polite ergative
            "आफ्नो",        // reflexive
            "आफू",          // self
            "हामी",         // we
            "हाम्रो",       // our
            "उनले",         // 3sg ergative
            // distinctive vocabulary
            "नयाँ",         // new
            "अर्को",        // next/other
            "ठूलो",         // big
            "सानो",         // small
            "राम्रो",       // good
            "रातो",         // red
            "सेतो",         // white
            "गाउँ",         // village
            "बुबा",         // father
            "आमा",          // mother
            "सबैभन्दा",   // most (superlative)
            "अगाडि",        // before
            "पछाडि",        // after
            "नजिकै",        // near (emphatic)
            "लागि",         // "for" (Nepali postposition; Hindi: के लिए)
            "जस्तै",        // like
            "त्यस्तै",     // such
            "केही",         // some
            // day / time names
            "बिहीबार",     // Thursday
            "बिहान",        // morning
            "नेपाल",        // Nepal
            "नेपाली",      // Nepali
            // subordinator
            "भन्ने",        // "saying"
            "ल्याउनुहोस्", // polite imperative "bring"
        ];
        Pair::build("hi", hi, "ne", ne)
    })
}

fn zh_tw_pair() -> &'static Pair {
    static P: OnceLock<Pair> = OnceLock::new();
    P.get_or_init(|| {
        // Hong Kong written Chinese — refines zh-TW (Traditional) into zh-HK
        // when Cantonese-influenced vocab / particles appear. Both regions
        // share Traditional Han characters, so this is the only signal that
        // distinguishes them. Asymmetric: side B is empty because zh-TW is
        // the default Traditional output and only HK-specific markers commit.
        let hk: &[&str] = &[
            "巴士",         // bus (TW: 公車 / 公共汽車)
            "的士",         // taxi (TW: 計程車)
            "唔該",         // thank you / excuse me (Cantonese)
            "唔好意思",     // sorry (Cantonese)
            "點解",         // why (Cantonese; TW: 為什麼)
            "喺",           // at (Cantonese particle)
            "嘅",           // possessive 's (Cantonese)
            "咗",           // past-tense marker (Cantonese)
            "係",           // is (Cantonese; standard uses 是)
            "冇",           // not have (Cantonese)
        ];
        let tw: &[&str] = &[];
        Pair::build("zh-HK", hk, "zh-TW", tw)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pt_br_commits_on_strong_markers() {
        let text = "Onde fica o ônibus para o aeroporto? Eu uso o trem todos os dias.";
        assert_eq!(disambiguate("pt", text), Some("pt-BR"));
    }

    #[test]
    fn pt_pt_commits_on_strong_markers() {
        let text = "Onde fica o autocarro para o aeroporto? Eu uso o comboio todos os dias.";
        assert_eq!(disambiguate("pt", text), Some("pt-PT"));
    }

    #[test]
    fn pt_neutral_returns_none() {
        let text = "Bom dia, como está?";
        assert_eq!(disambiguate("pt", text), None);
    }

    #[test]
    fn pt_single_marker_does_not_commit() {
        let text = "Vou pegar o ônibus.";
        // Only one marker — below COMMIT_MIN_HITS.
        assert_eq!(disambiguate("pt", text), None);
    }

    #[test]
    fn en_us_commits_on_spellings() {
        let text = "I love the color of my favorite sweater.";
        assert_eq!(disambiguate("en", text), Some("en-US"));
    }

    #[test]
    fn en_gb_commits_on_spellings() {
        let text = "I love the colour of my favourite jumper, whilst the centre is bright.";
        assert_eq!(disambiguate("en", text), Some("en-GB"));
    }

    #[test]
    fn en_neutral_returns_none() {
        let text = "I went to the store today.";
        assert_eq!(disambiguate("en", text), None);
    }

    #[test]
    fn fr_ca_commits_on_quebec_markers() {
        let text = "Je vais magasiner au dépanneur ce courriel à la fin de semaine.";
        assert_eq!(disambiguate("fr", text), Some("fr-CA"));
    }

    #[test]
    fn fr_fr_commits_on_european_markers() {
        let text = "Ce week-end je vais faire du shopping puis aller au parking.";
        assert_eq!(disambiguate("fr", text), Some("fr-FR"));
    }

    #[test]
    fn unknown_base_returns_none() {
        assert_eq!(disambiguate("ja", "こんにちは"), None);
        assert_eq!(disambiguate("zh", "你好"), None);
    }

    #[test]
    fn es_mx_commits_on_lat_am_markers() {
        let text = "Voy a manejar el carro a mi casa y luego rentar una computadora.";
        assert_eq!(disambiguate("es", text), Some("es-MX"));
    }

    #[test]
    fn es_es_commits_on_castilian_markers() {
        let text = "Vosotros tenéis un ordenador móvil que es muy guay, vale.";
        assert_eq!(disambiguate("es", text), Some("es-ES"));
    }

    #[test]
    fn es_neutral_returns_none() {
        assert_eq!(disambiguate("es", "Hola, ¿cómo estás?"), None);
    }

    #[test]
    fn ne_commits_on_nepali_markers() {
        let text = "यस सम्मेलन बिहीबार बिहान शहरको केन्द्रमा आयोजना गरिने छ। नेपाल मा हुनेछ।";
        assert_eq!(disambiguate("hi", text), Some("ne"));
    }

    #[test]
    fn hi_commits_on_hindi_markers() {
        let text = "क्या आप कृपया पुष्टि कर सकते हैं कि पैकेज मंगलवार को आ गया है?";
        assert_eq!(disambiguate("hi", text), Some("hi"));
    }

    #[test]
    fn devanagari_neutral_returns_none() {
        // Short, no distinctive markers either way.
        assert_eq!(disambiguate("hi", "नमस्ते"), None);
    }

    #[test]
    fn zh_hk_commits_on_hk_markers() {
        // Cantonese-influenced Traditional Chinese with bus / taxi / particles.
        let text = "我喺巴士站等的士，唔該你話我知點解咁耐。";
        assert_eq!(disambiguate("zh-TW", text), Some("zh-HK"));
    }

    #[test]
    fn zh_tw_no_hk_markers_returns_none() {
        // Clean Taiwan Traditional — no Cantonese-specific markers.
        let text = "我在計程車站等公車,請問你知道為什麼這麼久嗎?";
        assert_eq!(disambiguate("zh-TW", text), None);
    }

    #[test]
    fn word_boundary_blocks_substring_matches() {
        // "color" should NOT match inside "colorblind"-style compounds where it's
        // followed by an alphanumeric byte.
        let text = "colorblindness colorblindness";
        assert_eq!(disambiguate("en", text), None);
    }

    #[test]
    fn early_return_on_large_input() {
        // Lots of pt-BR markers up front, then a giant tail. Should commit on
        // the early markers without scanning the tail (correctness assertion;
        // performance is implicit).
        let prefix = "Você quer pegar o ônibus para a geladeira do café da manhã? ";
        let tail = "x".repeat(1_000_000);
        let text = format!("{prefix}{tail}");
        assert_eq!(disambiguate("pt", &text), Some("pt-BR"));
    }
}
