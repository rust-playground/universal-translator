# universal-translator


[![CI](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Platform: Linux | macOS](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

A universal text translator built in Rust. Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) (via the `llama-cpp-2` Rust crate) for fast, fully local inference of [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) (Gemma 3 4B instruction-tuned), covering 55 languages, and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection.

No API keys required. No network calls at runtime. Everything runs on your machine.

**License:** MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
**Model attributions:** see [ATTRIBUTIONS.md](ATTRIBUTIONS.md) — covers TranslateGemma 4B and the runtime libraries used for inference.
**Model license:** TranslateGemma 4B weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — see [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE).


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

Source language is detected automatically — no configuration required.

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

- Rust toolchain (stable) — [rustup.rs](https://rustup.rs)
- Tested on Linux (x86_64, arm64) and macOS (Apple Silicon)
- HuggingFace CLI for model download:
  ```bash
  pip install huggingface_hub[cli]
  hf auth login   # required: accept the Gemma licence at huggingface.co/google/translategemma-4b-it
  ```

## Quick start

### Build

```bash
cargo build --release
```

### Get the models

```bash
bash models/download.sh
```

This downloads TranslateGemma 4B (~4.1 GB, Q8_0 quantised, gated — requires HF login and
[Gemma license acceptance](https://huggingface.co/google/translategemma-4b-it)) into the
default model directory. For a smaller Q4_K_M variant (~2.6 GB):

```bash
bash models/download.sh --q4
```

Use `--model-file model-q4k.gguf` to select Q4_K_M at runtime. Q8_0 is the default (higher precision, comparable throughput under llama.cpp).
See [docs/models.md](docs/models.md) for details, directory layout, and alternative hosting options.

### Run the API server

```bash
cargo run -p translator-api
```

The server listens on `http://localhost:3000`. See [API.md](API.md) for endpoint
reference and request/response schemas.

To run with full observability (traces, metrics, logs pushed to a local monitoring stack):

```bash
docker compose -f docker/docker-compose.yml up -d
OTLP_ENDPOINT=http://localhost:4317 \
  cargo run -p translator-api --features opentelemetry
```

Grafana opens at http://localhost:3001 with a pre-built dashboard. See [METRICS.md](METRICS.md)
for the full setup guide and metrics catalogue.

### Run the CLI

```bash
cargo run -p translator-cli -- translate -t "Hello world" -l fr
```

See [CLI.md](CLI.md) for the full command reference (`translate`, `detect-language`,
`detect`, `languages`) including all flags and output formats.

## Documentation

- [CLI Reference](CLI.md) — all subcommands, flags, and output formats
- [API Reference](API.md) — HTTP endpoints, request/response schemas, examples
- [Engine Internals](ENGINE.md) — inference limits, concurrency, sampling, language detection
- [Observability](METRICS.md) — OTel traces/metrics/logs, Grafana dashboard, metrics catalogue

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
