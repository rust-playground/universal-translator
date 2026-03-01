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
bash models/download.sh   # requires: pip install huggingface_hub[cli] && hf auth login
```

Downloads TranslateGemma 4B. Required files in `${MODELS_DIR}/translategemma-4b/`:
- `model-q4k.gguf` (~2.6 GB, GGUF Q4_K_M quantised weights)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`

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

Key CLI flags: `--models-dir`, `--output json`.
Key API env vars: `MODELS_DIR`, `RUST_LOG`.

## Architecture

3-crate workspace:

- **`translator-core`** — library crate: engine, model, detector, types, error
- **`translator-cli`** — `ut` binary: Clap CLI with `translate`, `detect`, `languages` subcommands
- **`translator-api`** — Axum HTTP server: `POST /translate`, `GET /languages`, `GET /health`

### Inference stack

- **Model**: TranslateGemma 4B — Gemma 3 4B instruction-tuned decoder-only model for all 55 languages
- **Framework**: Candle (candle-core, candle-transformers, candle-nn)
- **Tokenizer**: HuggingFace fast tokenizer (`tokenizers` crate, `tokenizer.json`)
- **Language detection**: Lingua (75+ languages) with script-based fallback for Malayalam

### Core data flow (`engine.rs`)

1. **Detection** — parallel Lingua detection or normalise user-supplied source language
2. **Work building** — flatten texts × target languages; build Gemma instruct-format prompt with system turn and `Translate from X to Y:` user turn
3. **Worker dispatch** — send to background Tokio task; concurrent requests are coalesced into a single batch
4. **Inference** — `LoadedGemmaModel` + `ContinuousScheduler` runs batched decode with temperature/top-k/top-p sampling

### Key design decisions

- `TranslationEngine` is `Clone`-cheap (Arc-backed internals)
- `OnceCell<Arc<LoadedGemmaModel>>` — single model instance, async initialisation, lock-free read path after init
- Token limits: 4 096 tokens max output (SLOT_CAPACITY)
- Same-language shortcut: returns original text without inference

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Ubuntu (x86_64 + arm64) and macOS (Apple Silicon):
`cargo build` → `cargo test --workspace` → `cargo clippy --workspace -- -D warnings`
