# universal-translator

A universal text translator built in Rust. Uses [Candle](https://github.com/huggingface/candle) for fast, fully local inference of [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) (Gemma 3 4B instruction-tuned), covering 55 languages, and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection.

No API keys required. No network calls at runtime. Everything runs on your machine.

**License:** MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
**Model attributions:** see [ATTRIBUTIONS.md](ATTRIBUTIONS.md) — covers TranslateGemma 4B and the runtime libraries used for inference.
**Model license:** TranslateGemma 4B weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — see [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE).

[![CI](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Platform: Linux | macOS](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

---

## Supported languages

55 supported languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| af | Afrikaans | hr | Croatian | pt | Portuguese |
| am | Amharic | hu | Hungarian | ro | Romanian |
| ar | Arabic | id | Indonesian | ru | Russian |
| bg | Bulgarian | it | Italian | si | Sinhala |
| bn | Bengali | ja | Japanese | sk | Slovak |
| ca | Catalan | kn | Kannada | sl | Slovenian |
| cs | Czech | ko | Korean | sr | Serbian |
| da | Danish | lt | Lithuanian | sv | Swedish |
| de | German | lv | Latvian | sw | Swahili |
| el | Greek | ml | Malayalam | ta | Tamil |
| **en** | **English** | mr | Marathi | te | Telugu |
| es | Spanish | ms | Malay | th | Thai |
| et | Estonian | mt | Maltese | tr | Turkish |
| fa | Persian | ne | Nepali | uk | Ukrainian |
| fi | Finnish | nl | Dutch | ur | Urdu |
| fr | French | no | Norwegian | vi | Vietnamese |
| gu | Gujarati | pa | Punjabi | yi | Yiddish |
| ha | Hausa | pl | Polish | zh | Chinese |
| hi | Hindi | | | | |

Source language is detected automatically — no configuration required. Use `-s` to
supply a known source language and skip detection when it is already known.
All 55 languages are auto-detectable as source. Lingua handles 75+ languages;
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
# For the gated tokenizer repo: hf auth login
bash models/download.sh
```

This downloads TranslateGemma 4B (~2.6 GB, Q4_K_M quantised, gated — requires HF login and
[Gemma license acceptance](https://huggingface.co/google/translategemma-4b-it)) into the
default model directory.
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

Lingua is fully local — no API keys, no network calls. Detection data ships as compiled-in Rust crates. All 55 supported languages are valid translation targets; most are also detectable as source via Lingua. Malayalam (ml) is detected via Unicode script analysis (U+0D00–U+0D7F block) as a fallback when lingua returns no result.

## Limits

**Output token limit: 4 096 tokens** (SLOT_CAPACITY). Generation stops at this limit; very
long outputs will be truncated with no error returned.

Split very long documents into paragraphs or sentences before translating and reassemble the
results.

## Licensing

The source code in this repository is dual-licensed under the
[Apache License, Version 2.0](LICENSE-APACHE) and the [MIT License](LICENSE-MIT).
You may choose either license for your use of the source code.

### Model license

The **TranslateGemma** model weights and any model derivatives are **not** covered
by the Apache or MIT licenses. They are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
By downloading or using the model, you agree to abide by those terms.
See [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE).
