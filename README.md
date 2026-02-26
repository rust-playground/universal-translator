# universal-translator

A universal text translator built in Rust. Uses [Candle](https://github.com/huggingface/candle) for fast, fully local inference of [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt), a single seq2seq model covering 62 languages, and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection.

No API keys required. No network calls at runtime. Everything runs on your machine.

**License:** MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
**Model attributions:** see [ATTRIBUTIONS.md](ATTRIBUTIONS.md) — covers MADLAD-400-3B-MT and the runtime libraries used for inference.

[![CI](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Platform: Linux | macOS](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

---

## Supported languages

62 supported languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| af | Afrikaans | gu | Gujarati | pt | Portuguese |
| ar | Arabic | he | Hebrew | ro | Romanian |
| az | Azerbaijani | hi | Hindi | ru | Russian |
| be | Belarusian | hr | Croatian | sk | Slovak |
| bg | Bulgarian | hu | Hungarian | sl | Slovenian |
| bn | Bengali | hy | Armenian | so | Somali |
| ca | Catalan | id | Indonesian | sq | Albanian |
| cs | Czech | is | Icelandic | sr | Serbian |
| cy | Welsh | it | Italian | sv | Swedish |
| da | Danish | ja | Japanese | sw | Swahili |
| de | German | kk | Kazakh | ta | Tamil |
| el | Greek | ko | Korean | te | Telugu |
| **en** | **English** | lt | Lithuanian | th | Thai |
| es | Spanish | lv | Latvian | tr | Turkish |
| et | Estonian | mk | Macedonian | uk | Ukrainian |
| eu | Basque | ml | Malayalam | ur | Urdu |
| fa | Persian | mn | Mongolian | vi | Vietnamese |
| fi | Finnish | mr | Marathi | xh | Xhosa |
| fr | French | ms | Malay | yo | Yoruba |
| | | nl | Dutch | zh | Chinese |
| | | no | Norwegian | | |
| | | pa | Punjabi | | |
| | | pl | Polish | | |

Source language is detected automatically — no configuration required. Use `-s` to
supply a known source language and skip detection when it is already known.
All 62 languages are auto-detectable as source. Lingua handles 75+ languages;
Malayalam (ml) uses Unicode script analysis (U+0D00–U+0D7F) as a fallback.

---

## Workspace layout

```
universal-translator/
├── translator-core/   # Core library: engine, language detector, types
├── translator-api/    # Axum HTTP API server
├── translator-cli/    # Command-line interface
├── models/            # Model directories (not checked in — see below)
│   └── download.sh    # Script to download and convert all models
└── docs/models.md     # Model management guide
```

## Prerequisites

- Rust toolchain (stable, via [rustup](https://rustup.rs))
- Tested on Linux (x86\_64, arm64) and macOS (Apple Silicon).
- Models installed in the default directory (`~/.cache/ut/models` on Linux, `~/Library/Caches/ut/models` on macOS) — see [docs/models.md](docs/models.md)

## Quick start

### Build

```bash
cargo build --release
```

### Get the models

```bash
# Requires: pip install huggingface_hub[cli]
bash models/download.sh
```

This downloads MADLAD-400-3B-MT (~1.65 GB, GGUF int4 quantised) into the default model directory.
See [docs/models.md](docs/models.md) for details, directory layout, and alternative hosting options.

### Run the API server

```bash
cargo run -p translator-api
```

The server listens on `http://localhost:3000` by default.

### Run the CLI

```bash
# Translate to French
cargo run -p translator-cli -- translate -t "Hello world" -l fr

# Translate to multiple languages (comma-separated or repeated flag)
cargo run -p translator-cli -- translate -t "Hello world" -l fr,de,ja

# Translate with known source language (skips auto-detection)
cargo run -p translator-cli -- translate -t "Bonjour le monde" -s fr -l en,de

# List all supported languages
cargo run -p translator-cli -- languages
```

The `-l` flag accepts ISO 639-1 codes, and can be repeated (`-l fr -l de`) or
comma-separated (`-l fr,de`). Use `-s` to supply a known source language and skip
auto-detection.

## Beam width: speed vs quality

By default the engine **auto-selects beam width per request** based on the length of the
longest input text, calibrated at ~4 chars/token for English:

| Input length | Approx tokens | Beam | Notes |
|---|---|---|---|
| 0–60 chars | ≤15 | 0 (greedy) | Single clauses, names, short phrases — greedy is indistinguishable from beam search; no quality loss. |
| 61–160 chars | 15–40 | 2 | Full sentences where greedy occasionally makes mid-sequence errors. Beam=2 captures ~85% of beam=4's quality gain at roughly half the extra cost. |
| 161+ chars | >40 | 4 | Long or complex inputs where error accumulation matters most. |

Greedy decoding is ~3–4× faster than beam=4. The auto tiers ensure short inputs always run
at greedy speed while long inputs get the full quality treatment.

**Override with `--beam N`** to lock all requests to a fixed width — useful when you need
consistent latency or are benchmarking quality:

```bash
# CLI: force beam=4 for all inputs regardless of length
ut translate --beam 4 -t "Hi" -l fr

# API server: start with a fixed beam width
translator-api --beam 4

# API server: override via environment variable
BEAM_WIDTH=4 translator-api
```

When `--beam` / `BEAM_WIDTH` is unset, auto-selection is used (default for both CLI and API).

> **Concurrent API requests**: when multiple requests arrive simultaneously they are coalesced
> into a single GPU batch. With auto-beam enabled, requests are additionally grouped by their
> computed tier so a short request coalescing with a long one is never dragged up to beam=4
> speed.

## Integration tests

The integration test suite (`tests/integration.py`) calls the CLI binary and compares
translations against a golden CSV fixture.

### Prerequisites

- Python 3
- Models downloaded (see [Get the models](#get-the-models) above)
- CLI binary built (`cargo build -p translator-cli`)

### Run the tests

```bash
cargo build -p translator-cli && python3 tests/integration.py \
  --binary ./target/debug/ut
```

Pass `--models-dir /custom/path` to override the default model directory.
Add `--verbose` to see full actual vs. expected diffs on failures.

### Regenerate the golden fixture

Run this after changing `TEST_INPUTS` or when intentionally updating expected output:

```bash
cargo build -p translator-cli && python3 tests/integration.py --seed \
  --binary ./target/debug/ut
```

Review the generated `tests/fixtures/translations.csv` before committing — the seed script
prints warnings for any translation errors detected during generation.

> **Note on single-word inputs:** Very short inputs (e.g. `"Name"`, `"Username"`) may be
> detected as a language other than English. The seed script records the *actual detected
> language* in the CSV so that test mode always passes. Detection warnings during seeding
> are expected for these rows.

---

## Language detection

Lingua is fully local — no API keys, no network calls. Detection data ships as compiled-in Rust crates. All 62 supported languages are detectable as source. Malayalam (ml) is detected via Unicode script analysis (U+0D00–U+0D7F block) as a fallback when lingua returns no result.

## Limits

**Per-text token limit: 512 subword tokens** (HuggingFace fast tokenizer; roughly 300–500 words depending on language and script).

- **Input exceeds 512 tokens:** the input is silently truncated before inference. The
  translation will correspond only to the truncated portion with no error returned.
- **Output exceeds 1 024 tokens:** generation stops at 1 024 tokens. The translation will be
  incomplete with no error returned.

Split long documents into paragraphs or sentences before translating and reassemble the results.
