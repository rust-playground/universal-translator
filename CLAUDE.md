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

# Integration tests (requires built CLI binary)
cargo build -p translator-cli
python3 tests/integration.py --binary ./target/debug/ut
```

## Model Setup

Run once before using the CLI or API:

```bash
bash models/download.sh   # requires Python + pip packages: ctranslate2 transformers sentencepiece torch
```

Downloads and converts MADLAD-400-3B-MT. Required files in `${MODELS_DIR}/madlad400-3b-mt/`:
- `model-q4k.gguf` (~1.65 GB, GGUF int4 quantised weights)
- `config.json`, `tokenizer.json`

Default `MODELS_DIR`: platform cache directory (via `dirs` crate). Override with `--models-dir` flag or `MODELS_DIR` env var.

## Running

```bash
# CLI
cargo run -p translator-cli -- translate -t "Hello world" -l fr,de,ja
cargo run -p translator-cli -- languages
cargo run -p translator-cli -- detect -t "Bonjour"

# API server (http://localhost:3000)
cargo run -p translator-api
```

Key CLI flags: `--beam N` (default 4; 0/1 = greedy), `--models-dir`, `--output json`.
Key API env vars: `MODELS_DIR`, `BEAM_WIDTH` (default 0 = greedy), `RUST_LOG`.

## Architecture

3-crate workspace:

- **`translator-core`** — library crate: engine, model, detector, types, error
- **`translator-cli`** — `ut` binary: Clap CLI with `translate`, `detect`, `languages` subcommands
- **`translator-api`** — Axum HTTP server: `POST /translate`, `GET /languages`, `GET /health`

### Inference stack

- **Model**: MADLAD-400-3B-MT — single seq2seq model for all 62 languages
- **Framework**: Candle (candle-core, candle-transformers, candle-nn)
- **Tokenizer**: HuggingFace fast tokenizer (`tokenizers` crate, `tokenizer.json`)
- **Language detection**: Lingua (75+ languages) with script-based fallback for Malayalam

### Core data flow (`engine.rs`)

1. **Detection** — parallel Lingua detection or normalise user-supplied source language
2. **Work building** — flatten texts × target languages; prepend MADLAD control token `<2xx> ` to each input
3. **Worker dispatch** — send to background Tokio task; concurrent requests are coalesced into a single batch
4. **Inference** — `LoadedModel` runs greedy (beam ≤ 1) or beam search decoding in `model.rs`
5. **Post-processing** — Icelandic character-corruption fix (ó→ķ, ð→đ, þ→ū mappings)

### Key design decisions

- `TranslationEngine` is `Clone`-cheap (Arc-backed internals)
- `OnceLock<Arc<LoadedModel>>` — single model instance, lock-free read path after init
- Token limits: 512 tokens input (silently truncated); 1 024 tokens max output
- Same-language shortcut: returns original text without inference
- `beam_width` is startup config only — not part of the JSON request body

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Ubuntu (x86_64 + arm64) and macOS (Apple Silicon):
`cargo build` → `cargo test --workspace` → `cargo clippy --workspace -- -D warnings`
