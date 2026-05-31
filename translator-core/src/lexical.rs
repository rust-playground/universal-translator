//! Lexical cross-language reclassification.
//!
//! Lingua only knows ~75 base languages. Several supported targets are written
//! in the Latin or Ge'ez script and sit *inside* a parent lingua does know
//! (Galician inside Spanish, Luxembourgish inside German, Tigrinya inside the
//! Amharic Ethiopic block, …). Lingua emits the parent, so the detector would
//! otherwise mislabel a genuinely-correct translation.
//!
//! This module scores high-precision, distinctive marker words for each minority
//! language and commits only when one candidate clearly wins. It mirrors the
//! commit discipline of [`crate::dialect`] (≥2 hits, ≥2 margin) but generalizes
//! from a 2-way A/B pair to an N-candidate scorer competing against an implicit
//! "stay as the parent" baseline (the parent contributes zero markers, so
//! `MIN_HITS = 2` already means "beat the parent by ≥2").
//!
//! **Precision over recall.** Markers are curated to be distinctive against
//! *every* Latin-script supported language, not just the confused parent — e.g.
//! Asturian drops `centru`/`metru` (Romanian collision) and Occitan drops
//! `melhor`/`folhas` (Portuguese collision). When in doubt a marker is omitted,
//! so the text falls through to the parent unchanged.

use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, MatchKind};

// Commit when the top candidate has at least this many hits and beats the
// runner-up by this margin. `MIN_HITS = 2` is the real precision guard: the
// parent contributes zero markers, so a parent (Spanish, German, …) must hit
// two *distinctive* minority tokens by accident to fire — which curation makes
// near-impossible. `MARGIN = 1` only disambiguates between two minority
// siblings (gl vs ast vs oc share orthography); requiring margin 2 over a
// sibling would let a shared token block an otherwise-clear winner.
const COMMIT_MIN_HITS: u32 = 2;
const COMMIT_MARGIN: u32 = 1;

/// Reclassify predominantly-Latin text to a minority language when its markers
/// clearly win. Returns `None` (caller keeps the parent code) otherwise.
pub fn reclassify_latin(text: &str) -> Option<&'static str> {
    if !is_latin_dominant(text) {
        return None;
    }
    latin_set().run(text)
}

/// Split the Ethiopic (Ge'ez) script between Amharic and Tigrinya.
///
/// Defaults to `am` — the Ge'ez script is dominated by Amharic and short text
/// is indistinguishable. Commits to `ti` only when Tigrinya-distinctive function
/// words (copula `እዩ`, genitive `ናይ`, locative `ኣብ`) outscore Amharic anchors
/// by the margin. Real Amharic carries `ነው`/`ናቸው`/`ይካሄዳል` forms and stays `am`.
pub fn disambiguate_geez(text: &str) -> &'static str {
    geez_set().run(text).unwrap_or("am")
}

struct LexicalSet {
    matcher: AhoCorasick,
    /// `candidate_of[pattern_index]` = index into `codes` for that marker.
    candidate_of: Vec<u16>,
    codes: Vec<&'static str>,
}

impl LexicalSet {
    fn build(candidates: &[(&'static str, &[&str])]) -> Self {
        let mut patterns: Vec<&str> = Vec::new();
        let mut candidate_of: Vec<u16> = Vec::new();
        let mut codes: Vec<&'static str> = Vec::with_capacity(candidates.len());
        for (idx, (code, markers)) in candidates.iter().enumerate() {
            codes.push(code);
            for marker in *markers {
                patterns.push(marker);
                candidate_of.push(idx as u16);
            }
        }
        // LeftmostLongest so a longer marker wins over a prefix marker at the
        // same position (e.g. "hunne" beats "hunn") — otherwise the shorter
        // match is reported and then rejected by the word-boundary check,
        // dropping the token entirely.
        let matcher = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .expect("lexical markers must compile");
        Self { matcher, candidate_of, codes }
    }

    fn run(&self, text: &str) -> Option<&'static str> {
        let bytes = text.as_bytes();
        let mut hits = vec![0u32; self.codes.len()];

        for mat in self.matcher.find_iter(text) {
            // Word-boundary check: surrounding byte must not be ASCII
            // alphanumeric. High-bit (UTF-8 multibyte) bytes count as
            // boundaries — the same accepted simplification as dialect.rs.
            let start = mat.start();
            let end = mat.end();
            let prev_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
            let next_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
            if !prev_ok || !next_ok {
                continue;
            }
            hits[self.candidate_of[mat.pattern().as_usize()] as usize] += 1;
        }

        // Highest and runner-up. Ties leave margin 0 → no commit.
        let mut best = 0u32;
        let mut best_idx = 0usize;
        let mut second = 0u32;
        for (idx, &count) in hits.iter().enumerate() {
            if count > best {
                second = best;
                best = count;
                best_idx = idx;
            } else if count > second {
                second = count;
            }
        }

        if best >= COMMIT_MIN_HITS && best - second >= COMMIT_MARGIN {
            Some(self.codes[best_idx])
        } else {
            None
        }
    }
}

/// True when most alphabetic characters are Latin-script. Keeps the Latin
/// lexical scorer away from Cyrillic / CJK / Indic / Ge'ez text entirely.
fn is_latin_dominant(text: &str) -> bool {
    let mut latin = 0usize;
    let mut non_latin = 0usize;
    for c in text.chars() {
        if !c.is_alphabetic() {
            continue;
        }
        let is_latin = matches!(
            c as u32,
            0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF
        );
        if is_latin {
            latin += 1;
        } else {
            non_latin += 1;
        }
    }
    latin > non_latin
}

// ── Marker sets ────────────────────────────────────────────────────────────
// Each marker is distinctive against ALL Latin-script supported languages, not
// just the parent lingua confuses the language with. Forms that collide with a
// neighbor (Portuguese `melhor`, Romanian `centru`/`metru`, Italian `nel`,
// French `moi`) are deliberately excluded.

fn latin_set() -> &'static LexicalSet {
    static SET: OnceLock<LexicalSet> = OnceLock::new();
    SET.get_or_init(|| {
        // Galician — distinctive vs Spanish AND Portuguese (accents / x-words /
        // contractions). `-ou` preterites and `terá` omitted (shared with pt).
        let gl: &[&str] = &[
            "máis", "mañá", "súa", "aínda", "mellor", "vermello", "enerxía",
            "emerxencia", "xardín", "viaxe", "viaxou", "viaxes", "auga", "pola",
            "polas", "polo", "polos", "dúas", "nenos", "nenas", "facer", "tamén",
            "concello", "deixar", "nunha", "persoas", "civilizacións", "moitas",
            "ningunha", "ningún",
        ];
        // Asturian — Spanish-rooted `-u` masculine forms that do NOT collide
        // with Romanian's own `-u` words (centru/metru dropped), plus genuinely
        // unique vocabulary (fueyes, dambes, conceyu, güei).
        let ast: &[&str] = &[
            "fueyes", "fueya", "dambes", "pueblu", "próximu", "pequeñu", "roxu",
            "preciu", "contratu", "proxetu", "ríu", "abuelu", "almuerzu", "conceyu",
            "güei", "ésitu", "esitu", "trabayu", "traballu", "vieyu", "fíu",
            "ciudá", "augua", "viaxó", "principiu", "retrasu",
        ];
        // Occitan — grave-accent and `-cion` forms distinctive vs Catalan,
        // Spanish, French, Portuguese, Italian. Conservative (low recall):
        // `lo`/`melhor`/`folhas`/`vila`/`farà` dropped (collide with es/pt/ca).
        let oc: &[&str] = &[
            "nòstre", "nòstra", "aquò", "tanben", "tanbén", "mercé", "fòrça",
            "pòble", "nòva", "nòu", "exposicion", "meditacion", "poblacion",
            "informacion", "nacion", "ostal", "dab", "ambe", "trabalh", "paire",
            "conferéncia", "emergéncia", "paciéncia", "premètz", "arribatz",
            "utilizatz",
        ];
        // Luxembourgish — spellings unique vs German (gëtt≠gibt, vun≠von,
        // dräi≠drei) and the `-éiert` verb ending.
        let lb: &[&str] = &[
            "gëtt", "vun", "vum", "dräi", "huet", "hunn", "hunne", "iwwer", "joer",
            "moien", "waasser", "maachen", "gemaach", "véier", "ënner",
            "annoncéiert", "recommandéiert", "presentéiert", "studéiert",
            "geschriwwen", "wëssenschaft", "sinn", "wërt", "ier", "awer", "déi",
            "ass", "séng", "neie", "neien",
        ];
        // Cebuano — distinctive vs Tagalog/Filipino. Shared particles
        // (sa/ang/mga/nga/siya) and fil-shared nouns (nasod/siyudad) omitted.
        let ceb: &[&str] = &[
            "gikan", "ug", "kini", "kana", "dili", "kinahanglan", "magkinahanglan",
            "duha", "adlaw", "adlawng", "iyang", "imong", "human", "kaayo", "unsa",
            "maayo", "pinakamaayo", "daghan", "kadaghanan", "niadtong", "mahimo",
            "sunod", "bag-o", "bag-ong", "walay", "samtang", "makahatag", "nakabaton",
        ];
        // Sundanese — distinctive vs Indonesian/Malay (function words and the
        // accented é forms).
        let su: &[&str] = &[
            "éta", "anjeun", "abdi", "teu", "nyaéta", "nyéta", "sareng", "jeung",
            "jeun", "dina", "pikeun", "ngeunaan", "waktos", "gaduh", "saé",
            "pangsaéna", "nginum", "nagara", "anu", "aya", "kedah", "bakal", "ieu",
            "yén", "désa", "sakabéh", "parantos", "réa",
        ];
        LexicalSet::build(&[
            ("gl", gl),
            ("ast", ast),
            ("oc", oc),
            ("lb", lb),
            ("ceb", ceb),
            ("su", su),
        ])
    })
}

fn geez_set() -> &'static LexicalSet {
    static SET: OnceLock<LexicalSet> = OnceLock::new();
    SET.get_or_init(|| {
        // Tigrinya-distinctive function words: copula እዩ/ኢዩ/እዮም, genitive ናይ,
        // locative ኣብ, existential ኣለ/ዘሎ, negation prefix ኣይ.
        let ti: &[&str] = &[
            "እዩ", "ኢዩ", "እዮም", "እያ", "ናይ", "ኣብ", "ኣለ", "ዘሎ", "ኣይ", "ብሓደ", "ማእከል",
        ];
        // Amharic anchors: copula ነው/ናቸው, future ይሆናል, passive ይካሄዳል/ይጠብቃል,
        // past ነበር, locatives ላይ/ውስጥ. Keep Tigrinya from winning on real Amharic.
        let am: &[&str] = &[
            "ነው", "ናቸው", "ይሆናል", "ይካሄዳል", "ይጠብቃል", "ነበር", "ላይ", "ውስጥ", "ማድረግ",
        ];
        LexicalSet::build(&[("ti", ti), ("am", am)])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Latin reclassification: positives (real harness outputs) ────────────

    #[test]
    fn galician_commits() {
        let text = "A conferencia terá lugar no centro da cidade o martes pola mañá. \
                    O médico recomendou beber máis auga e facer exercicio.";
        assert_eq!(reclassify_latin(text), Some("gl"));
    }

    #[test]
    fn asturian_commits() {
        let text = "El so abuelu serviu na marina. Los niños ríen y corren polos \
                    fueyes que caen. El preciu inclúe el almuerzu nel pueblu.";
        assert_eq!(reclassify_latin(text), Some("ast"));
    }

    #[test]
    fn occitan_commits() {
        let text = "La nòva exposicion del museu presenta artefactes. Aquò es \
                    nòstre ostal, tanben la poblacion de la nacion.";
        assert_eq!(reclassify_latin(text), Some("oc"));
    }

    #[test]
    fn luxembourgish_commits() {
        let text = "D'Konferenz gëtt am Zentrum vun der Stad op Moien. Mir hunn \
                    dräi Joer geschafft an huet Waasser gedronk.";
        assert_eq!(reclassify_latin(text), Some("lb"));
    }

    #[test]
    fn cebuano_commits() {
        let text = "Dili dapat iwanan ang mga bata. Kinahanglan nga adunay duha ka \
                    tasa ug daghan nga tubig gikan sa imong balay.";
        assert_eq!(reclassify_latin(text), Some("ceb"));
    }

    #[test]
    fn sundanese_commits() {
        let text = "Konferensi éta bakal dilaksanakeun dina waktos pagi. Anjeun \
                    gaduh pengalaman sareng anu saé pikeun nginum.";
        assert_eq!(reclassify_latin(text), Some("su"));
    }

    // ── Latin reclassification: negatives (parents must stay parent) ─────────

    #[test]
    fn real_spanish_does_not_commit() {
        let text = "La conferencia se celebrará en el centro de la ciudad el martes \
                    por la mañana. El médico recomendó beber más agua y hacer ejercicio.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_portuguese_does_not_commit() {
        // melhor / folhas would have collided with the Occitan set if kept.
        let text = "O médico recomendou beber mais água. As folhas caem e o melhor \
                    pão da cidade é vendido na padaria.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_romanian_does_not_commit() {
        // centru / metru are Romanian -u words; must not trip Asturian.
        let text = "Conferința va avea loc în centrul orașului. Am mers un metru \
                    până la lucrul principal din oraș.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_german_does_not_commit() {
        let text = "Die Konferenz findet am Dienstagmorgen im Stadtzentrum statt. \
                    Der Arzt empfahl, mehr Wasser zu trinken und Sport zu treiben.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_indonesian_does_not_commit() {
        let text = "Konferensi akan diadakan di pusat kota pada pagi hari Selasa. \
                    Dokter menyarankan minum lebih banyak air dan berolahraga.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_tagalog_does_not_commit() {
        let text = "Ang kumperensya ay gaganapin sa sentro ng lungsod sa Martes ng \
                    umaga. Hindi dapat iwanan ang mga bata na walang bantay.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_catalan_does_not_commit() {
        // Occitan's parent. Catalan -ència / conferència must not trip oc's
        // -éncia markers, and `lo`/`melhor`/`vila` are deliberately absent.
        let text = "La conferència se celebrarà al centre de la ciutat dimarts al \
                    matí. El metge va recomanar beure més aigua i fer exercici.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_french_does_not_commit() {
        let text = "La conférence se tiendra au centre-ville mardi matin. Le médecin \
                    a recommandé de boire plus d'eau et de faire de l'exercice.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_italian_does_not_commit() {
        let text = "La conferenza si terrà nel centro della città martedì mattina. \
                    Il medico ha raccomandato di bere più acqua.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn real_english_does_not_commit() {
        let text = "The conference will be held in the city center on Tuesday \
                    morning. The human team finished the project ahead of schedule.";
        assert_eq!(reclassify_latin(text), None);
    }

    #[test]
    fn non_latin_text_short_circuits() {
        assert_eq!(reclassify_latin("ሰላም ነው ላይ ውስጥ"), None);
        assert_eq!(reclassify_latin("你好世界"), None);
    }

    // ── Ge'ez split ─────────────────────────────────────────────────────────

    #[test]
    fn tigrinya_commits_on_distinctive_function_words() {
        // Real harness Tigrinya rows.
        assert_eq!(disambiguate_geez("ዋሓኑ ኣብ ናይ ወጻኢ ወርሒ፣ ብሓደ።"), "ti");
        assert_eq!(
            disambiguate_geez("ኣብ ማእከል፣ ሓደ እውን ብምስክን፣ ዕድል ይገኝ እዩ ብሞ ከምዚ ኣለ።"),
            "ti"
        );
    }

    #[test]
    fn real_amharic_stays_am() {
        // Real harness Amharic rows — copula / passive forms anchor `am`.
        assert_eq!(
            disambiguate_geez("የተሰጠው ኮንፈረንስ፣ ሰኞ ጠዋት ከተማ ማዕከል ላይ ይካሄዳል።"),
            "am"
        );
        assert_eq!(
            disambiguate_geez("አቶ አባቱ በሁለተኛው ጦርነት ወቅት በሰለጠኑ ሰራዊት ውስጥ ተጠቅልሎ ነበር።"),
            "am"
        );
    }

    #[test]
    fn short_geez_defaults_to_am() {
        assert_eq!(disambiguate_geez("ሰላም"), "am");
    }
}
