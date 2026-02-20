# universal-translator

A universal text translator built in Rust. It uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for fast, local inference of [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) models, and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection.

No API keys required. No network calls at runtime. Everything runs on your machine.

**License:** MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
**Model attributions:** see [ATTRIBUTIONS.md](ATTRIBUTIONS.md).

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

Source language is detected automatically — no configuration required.
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
- Models installed in `models/` — see [docs/models.md](docs/models.md)

## Quick start

### Build

```bash
cargo build --release
```

### Get the models

```bash
# Requires: pip install ctranslate2 transformers sentencepiece torch
bash models/download.sh
```

This downloads and converts all supported OPUS-MT models (~4 GB total).
See [docs/models.md](docs/models.md) for details and alternative hosting options.

### Run the API server

```bash
MODELS_DIR=./models cargo run -p translator-api
```

The server listens on `http://localhost:3000` by default.

### Run the CLI

```bash
# Translate to French
cargo run -p translator-cli -- translate -t "Hello world" -l fr

# Translate to multiple languages
cargo run -p translator-cli -- translate -t "Hello world" -l fr -l de -l ja

# List all supported languages
cargo run -p translator-cli -- languages
```

The `-l` flag accepts ISO 639-1 codes. The source language is detected automatically.

## Language detection

Lingua is fully local — no API keys, no network calls. Detection data ships as compiled-in Rust crates. 42 of the 43 supported languages are detected by lingua. Malayalam (ml) is detected via Unicode script analysis (U+0D00–U+0D7F block) as a fallback when lingua returns no result.

## Adding language pairs

See [docs/models.md](docs/models.md) for step-by-step instructions on converting and installing OPUS-MT models.
