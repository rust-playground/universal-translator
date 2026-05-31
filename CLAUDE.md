# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build                          # debug build (CPU)
cargo build --release                # release build
cargo build --features metal         # macOS GPU (Metal)
cargo build --features cuda          # Linux GPU (CUDA/NVIDIA)
cargo clippy --workspace -- -D warnings   # lint (CI enforces -D warnings)
cargo test --workspace               # unit tests

# Load test (requires running API server)
cargo run -p translator-api-client --bin load-test
```

## Model Setup

Run once before using the CLI or API:

```bash
cargo run -p translator-cli -- setup              # download Q8_0 (default, ~4.1 GB)
cargo run -p translator-cli -- setup --url <url>   # download custom GGUF
```

Downloads TranslateGemma 4B GGUF weights directly from HuggingFace (no Python/hf CLI dependency).
Default output: `<cache>/ut/models/translategemma-4b/model-q8_0.gguf` (tokenizer embedded in GGUF).

Q4_K_M (~2.6 GB) available at:
`https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf`

Select model file at runtime: `--model-path <path>` flag or `MODEL_PATH=<path>` env var.
Default `MODEL_PATH`: platform cache directory (via `dirs` crate) + `ut/models/translategemma-4b/model-q8_0.gguf`.

## Running

```bash
# CLI (`ut` binary)
cargo run -p translator-cli -- translate -t "Hello world" -l fr,de,ja          # base codes
cargo run -p translator-cli -- translate -t "Hello world" -l pt-BR,zh-Hant     # regional variants (dash or underscore, case-insensitive)
cargo run -p translator-cli -- detect -t "Bonjour"                              # returns BCP 47 string
cargo run -p translator-cli -- detect-language "Bonjour" --output json          # full result with confidence + translate_language
cargo run -p translator-cli -- languages                                        # translate-supported (default; 98 entries)
cargo run -p translator-cli -- languages --for detect                            # broader detect-supported list (108 entries)

# API server (http://localhost:3000)
cargo run -p translator-api
```

Subcommands: `translate`, `detect`, `detect-language`, `languages`, `setup`.

Global CLI flags: `--model-path`, plus per-command `--output pretty|json`.
Subcommand-specific: `languages --for translate|detect`.
Key API env vars: `MODEL_PATH`, `RUST_LOG`.

## Architecture

4-crate workspace:

- **`translator-core`** — library: engine, scheduler, model loader, detector, dialect heuristics, language enum, types, error
- **`translator-cli`** — `ut` binary, Clap CLI
- **`translator-api`** — Axum HTTP server
- **`translator-api-client`** — typed Rust client for the API (used by `load-test`)

### API endpoints

- `POST /translate` — batch translate; accepts `texts`, `target_languages` (list or `["all"]`), optional `source_language`
- `POST /translate/stream` — same input, SSE stream of per-text results
- `POST /detect-language` — returns `{ language, translate_language, confidence }`
- `GET /languages?for=translate|detect` — list supported codes (default `translate`)
- `GET /health` — liveness

### Inference stack

- **Model**: TranslateGemma 4B (Gemma 3 4B instruction-tuned, decoder-only). Translate-side `Language` enum has **98 variants**: WMT24++ validated set (base codes + 11 regional pairs `ar_EG`, `ar_SA`, `es_MX`, `fr_CA`, `fr_FR`, `pt_BR`, `pt_PT`, `sw_KE`, `sw_TZ`, `zh_CN`, `zh_TW`), 3 added base codes (`He`, `Is`, `Fil`), 4 added regional (`en_GB`, `en_US`, `es_ES`, `zh_HK`), and harness-validated Tier A/B/C/D additions the model can actually translate. Two harness-driven curation rounds dropped codes the model could not produce (wrong-language or script-salad output): first 18 (`ff`, `jv`, `kac`, `ln`, `lu`, `luo`, `mai`, `nso`, `ny`, `om`, `sd`, `sn`, `so`, `to`, `wo`, `xh`, `yo`, `zu`), then 13 more (`bo`, `ha`, `ig`, `lg`, `mg`, `mi`, `mt`, `pa`, `rup`, `ti`, `tk`, `tt`, `yue`) — e.g. `pa`→Hindi (naming the Gurmukhi script in the prompt didn't help), `ti`→Amharic, `tk`/`tt`→Kazakh, `mi` fluent but unusable quality. These codes stay detect-supported (broader universe); translate rejects them with `UnsupportedLanguage`. Variant naming: `pt_BR` style under `#[allow(non_camel_case_types)]`; `code()` returns BCP 47 dash form (`"pt-BR"`). `full_name()` returns English label (`"Brazilian Portuguese"`).
- **Framework**: [llama.cpp](https://github.com/ggerganov/llama.cpp) via `llama-cpp-2`
- **Tokenizer**: embedded in the GGUF file
- **Prompt**: `translate_gemma_prompt` uses **full English language names** plus a target-script anchor (`"Translate from English to Brazilian Portuguese:"` + system instruction pinning the target's native script). Switched from BCP 47 codes after diagnostics showed the 4B model misinterpreted ambiguous 2-letter codes (`si` → Slovene, `or` → Spanish, `af` → Hindi). Names eliminate that confusion at the cost of a few extra prompt tokens.

### Language detection (`detector.rs` + `dialect.rs`)

Pipeline. Each step is non-destructive — falls through to the previous result on no commit. Full pipeline returns a `String` (BCP 47) — broader than the translate enum.

1. **Within-script overrides** (deterministic, *before* lingua) — distinctive single codepoints that commit regardless of lingua: `bn`→`as` (ৰ/ৱ), `ar`→`ckb` (Sorani ێ/ۆ/ڕ/ڵ/ڤ), Arabic-block→`ps` (Pashto-unique ښ/ږ/ګ/ړ/ډ/ټ/ڼ — zero false positives, like Sorani).
2. **Script-only fast path** (deterministic, before lingua) — unique-script blocks committed by Unicode range: `ml` (Malayalam), `kn` (Kannada), `ta` (Tamil), `te` (Telugu), `gu` (Gujarati), `pa` (Gurmukhi), `or` (Oriya), `km` (Khmer), `lo` (Lao), `my` (Burmese), `bo` (Tibetan), `si` (Sinhala). Ethiopic is routed through `lexical::disambiguate_geez` — defaults to `am` (Amharic), commits `ti` (Tigrinya) only on Tigrinya-distinctive function words (copula እዩ, genitive ናይ, locative ኣብ).
3. **Lingua** — 75 base ISO 639-1 / 639-3 codes (parallel detection via Rayon for batches). Lingua codes normalized via `lingua_to_bcp47` (`tl` → `fil`, `nb`/`nn` → `no`).
4. **Latin cross-language reclassification** (`lexical.rs`, best-effort) — lingua only knows the *parent*, so minority Latin-script languages are recovered from distinctive marker words via an N-candidate Aho-Corasick scorer (`LeftmostLongest`; commit when winner ≥ 2 hits and beats the runner-up sibling by ≥ 1; the parent has 0 markers so ≥2 hits is the precision guard). Covers `gl` (Galician), `oc` (Occitan), `ast` (Asturian), `lb` (Luxembourgish), `ceb` (Cebuano), `su` (Sundanese). Markers are curated distinctive against *all* Latin neighbors (e.g. Asturian drops `centru`/`metru` for Romanian collision, Occitan drops `melhor`/`folhas` for Portuguese). Skips non-Latin text. On commit, refine is bypassed (no sub-dialects). Harness: high precision (no parent regressions); recall is capped below PASS for the closest pairs because the 4B model itself emits hybrid output for them.
5. **Within-script disambiguation** (deterministic, Unicode-block / distinctive-letter tests, no false positives) — `zh-CN`/`zh-TW` (Han Simplified vs Traditional), `sr-Cyrl`/`sr-Latn`, `az-Cyrl`/`az-Latn`/`az-Arab`, `pa-Guru`/`pa-Arab`, `mn-Cyrl`/`mn-Mong`, `he` → `yi` (Yiddish niqqud + precomposed ligatures U+05F0–U+05F2).
6. **Heuristic dialect markers** (`dialect.rs`, best-effort) — Aho-Corasick scoring with streaming early-return (commit when winner ≥ 2 hits and beats loser by ≥ 2). Currently covers `pt-BR`/`pt-PT`, `en-US`/`en-GB`, `fr-CA`/`fr-FR`, `es-MX`/`es-ES`, `zh-TW` → `zh-HK` (Cantonese particles), `hi` → `ne` (Nepali copula/verb markers).

`LanguageDetectionResult { language: String, translate_language: Option<Language>, confidence: f64 }`. `language` is the raw detector output; `translate_language` is the same code parsed via `Language::FromStr` — surfaces aliases (`nb`/`nn` → `no`, `tl` → `fil`, `iw` → `he`, `zh-Hans` → `zh-CN`, `pt-AO` → `pt`). One mapping table, two consumers (input parsing on `/translate` and output mapping on `/detect-language`).

**Boundary contract:** detect's universe is broader than the translate enum. The engine's auto-detect-source path returns `UnsupportedLanguage` (not `DetectionFailed`) when the detected code is lingua-only (e.g. `cy` Welsh, `ka` Georgian). Pass an explicit `source_language` to translate from those.

See `README.md` for the canonical support table. Source of truth for live counts: `GET /languages?for=translate|detect`.

### Core data flow (`engine.rs`)

1. **Detection** — parallel Lingua + post-processing per text (or pass-through if caller supplied `source_language`). Detect output is parsed via `Language::from_str`; lingua-only codes → `UnsupportedLanguage`.
2. **Work building** — flatten texts × target languages; build Gemma instruct-format prompt with system turn and `Translate from <src-code> to <tgt-code>:` user turn (BCP 47 codes, not English names).
3. **Worker dispatch** — send to dedicated scheduler thread via crossbeam channel; concurrent requests coalesce into a single batch.
4. **Inference** — `LoadedGemmaModel` + `ContinuousScheduler` runs batched decode via `LlamaContext` with temperature / top-k / top-p sampling, repetition penalty, no-repeat n-gram, length bias.

### Key design decisions

- `TranslationEngine` is `Clone`-cheap (Arc-backed internals)
- Single model loaded synchronously at startup, shared read-only via `Arc<LoadedGemmaModel>`
- Token limits: 4 096 tokens max output (SLOT_CAPACITY)
- Same-language shortcut: returns original text without inference
- Auto-chunking at `\n\n` paragraph boundaries (Unicode sentence fallback) for long inputs; reassembled before returning
- `Language::FromStr` accepts dash and underscore, case-insensitive; unknown region tags fall back to base with a `tracing::debug!`

### Key dependencies

- `llama-cpp-2` (vendored llama.cpp) — inference
- `lingua` — base language detection
- `aho-corasick` — dialect heuristic matcher
- `rayon` — parallel detection
- `crossbeam-channel` — scheduler queue
- `axum` — HTTP server, `tokio` runtime; CLI is plain sync `fn main`

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Ubuntu (x86_64 + arm64) and macOS (Apple Silicon):
`cargo build` → `cargo test --workspace` → `cargo clippy --workspace -- -D warnings`
