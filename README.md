# universal-translator

A universal text translator built in Rust. It uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for fast, local inference of [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) models, and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection.

No API keys required. No network calls at runtime. Everything runs on your machine.

**License:** MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
**Model attributions:** see [ATTRIBUTIONS.md](ATTRIBUTIONS.md).

[![CI](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Platform: Linux | macOS](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

---

## Supported languages

43 supported languages (42 translation targets + English):

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| af | Afrikaans | hi | Hindi | sk | Slovak |
| ar | Arabic | hu | Hungarian | sq | Albanian |
| bg | Bulgarian | hy | Armenian | sv | Swedish |
| ca | Catalan | id | Indonesian | sw | Swahili |
| cs | Czech | is | Icelandic | tl | Tagalog |
| cy | Welsh | it | Italian | tr | Turkish |
| da | Danish | ja | Japanese | uk | Ukrainian |
| de | German | lt | Lithuanian | ur | Urdu |
| el | Greek | lv | Latvian | vi | Vietnamese |
| **en** | **English** | mk | Macedonian | zh | Chinese |
| eo | Esperanto | ml | Malayalam | | |
| es | Spanish | mr | Marathi | | |
| et | Estonian | nl | Dutch | | |
| eu | Basque | pt | Portuguese | | |
| fi | Finnish | ro | Romanian | | |
| fr | French | ru | Russian | | |
| he | Hebrew | | | | |

Source language is detected automatically — no configuration required. Use `-s` to
supply a known source language and skip detection when it is already known.
43 of the 43 supported languages are detectable as source. Malayalam detection uses script analysis (U+0D00–U+0D7F); the remaining 42 use lingua.

Galician (gl) and Maltese (mt) are not supported: both use the Latin script, making
automatic source-language detection unreliable without additional dependencies.
Galician speakers are covered by Spanish (`es`); Maltese speakers are covered by
English (`en`).

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
# Requires: cmake, pip install ctranslate2 transformers sentencepiece torch
bash models/download.sh
```

This downloads and converts all supported OPUS-MT models (~4 GB total).
See [docs/models.md](docs/models.md) for details and alternative hosting options.

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

Lingua is fully local — no API keys, no network calls. Detection data ships as compiled-in Rust crates. 42 of the 43 supported languages are detected by lingua. Malayalam (ml) is detected via Unicode script analysis (U+0D00–U+0D7F block) as a fallback when lingua returns no result.

## Limits

**Per-text token limit: 512 SentencePiece tokens** (roughly 300–500 words depending on language and script).

OPUS-MT models have positional embeddings up to 512 tokens. Both encoder (input) and decoder (output) are bounded by this limit:

- **Input exceeds 512 tokens:** the input is silently truncated before inference. The
  translation will correspond only to the truncated portion with no error returned.
- **Output exceeds 512 tokens:** generation stops at 512 tokens. The translation will be
  incomplete with no error returned.

Split long documents into paragraphs or sentences before translating and reassemble the results.

## Adding language pairs

See [docs/models.md](docs/models.md) for step-by-step instructions on converting and installing OPUS-MT models.
