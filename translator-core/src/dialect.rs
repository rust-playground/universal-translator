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
pub fn disambiguate(base: &str, text: &str) -> Option<&'static str> {
    let pair = match base {
        "pt" => pt_pair(),
        "en" => en_pair(),
        "fr" => fr_pair(),
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
