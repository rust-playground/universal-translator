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
# CLI
cargo run -p translator-cli -- translate -t "Hello world" -l fr,de,ja
cargo run -p translator-cli -- languages
cargo run -p translator-cli -- detect -t "Bonjour"

# API server (http://localhost:3000)
cargo run -p translator-api
```

Key CLI flags: `--model-path`, `--output json`.
Key API env vars: `MODEL_PATH`, `RUST_LOG`.

## Architecture

3-crate workspace:

- **`translator-core`** — library crate: engine, model, detector, types, error
- **`translator-cli`** — `ut` binary: Clap CLI with `translate`, `detect`, `languages` subcommands
- **`translator-api`** — Axum HTTP server: `POST /translate`, `GET /languages`, `GET /health`

### Inference stack

- **Model**: TranslateGemma 4B — Gemma 3 4B instruction-tuned decoder-only model for all 55 languages
- **Framework**: [llama.cpp](https://github.com/ggerganov/llama.cpp) via the `llama-cpp-2` Rust crate
- **Tokenizer**: embedded in the GGUF file (no separate `tokenizer.json` needed)
- **Language detection**: Lingua (75+ languages) with script-based fallback for Malayalam

### Core data flow (`engine.rs`)

1. **Detection** — parallel Lingua detection or normalise user-supplied source language
2. **Work building** — flatten texts × target languages; build Gemma instruct-format prompt with system turn and `Translate from X to Y:` user turn
3. **Worker dispatch** — send to dedicated scheduler thread via crossbeam channel; concurrent requests are coalesced into a single batch
4. **Inference** — `LoadedGemmaModel` + `ContinuousScheduler` runs batched decode via `LlamaContext` with temperature/top-k/top-p sampling

### Key design decisions

- `TranslationEngine` is `Clone`-cheap (Arc-backed internals)
- Single model instance loaded synchronously at startup, shared read-only via `Arc<LoadedGemmaModel>`
- Token limits: 4 096 tokens max output (SLOT_CAPACITY)
- Same-language shortcut: returns original text without inference

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Ubuntu (x86_64 + arm64) and macOS (Apple Silicon):
`cargo build` → `cargo test --workspace` → `cargo clippy --workspace -- -D warnings`
