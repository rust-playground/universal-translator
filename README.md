# universal-translator

A universal text translator built in Rust. It uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for fast, local inference of [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) models, and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection.

No API keys required. No network calls at runtime. Everything runs on your machine.

---

## Workspace layout

```
universal-translator/
├── translator-core/   # Core library: engine, language detector, types
├── translator-api/    # Axum HTTP API server
├── translator-cli/    # Command-line interface
└── models/            # Converted CTranslate2 model directories (not checked in)
```

## Prerequisites

- Rust toolchain (stable, via [rustup](https://rustup.rs))
- A converted OPUS-MT model in `models/` — see [docs/models.md](docs/models.md)

## Quick start

### Build

```bash
cargo build --release
```

### Run the API server

```bash
MODELS_DIR=./models cargo run -p translator-api
```

The server listens on `http://localhost:3000` by default.

### Run the CLI

```bash
# Translate text to French
cargo run -p translator-cli -- translate -t "Hello world" -l fr

# Translate text to German
cargo run -p translator-cli -- translate -t "Hello world" -l de
```

The `-l` flag specifies the target language as a two-letter ISO 639-1 code. The source language is detected automatically by Lingua.

## Language detection

Lingua is fully local — no API keys, no network calls, no external services. All language model data ships as compiled-in Rust crates. Detection supports 75+ languages out of the box.

## Adding language pairs

See [docs/models.md](docs/models.md) for step-by-step instructions on converting and installing OPUS-MT models.
