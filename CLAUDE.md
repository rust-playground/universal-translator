# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build                          # debug build (CPU)
cargo build --release                # release build
cargo build --features coreml        # macOS GPU/ANE (CoreML)
cargo build --features cuda          # Linux GPU (CUDA/NVIDIA + TensorRT)
cargo clippy --workspace -- -D warnings   # lint (CI enforces -D warnings)
cargo test --workspace               # unit tests

# Integration tests (requires built CLI binary)
cargo build -p translator-cli
python3 tests/integration.py --binary ./target/debug/ut
```

## Model Setup

Run once before using the CLI or API:

```bash
bash models/download.sh
# requires: pip install "optimum[exporters]>=1.19" transformers sentencepiece onnxruntime
```

Exports MADLAD-400-3B-MT to ONNX format. Required files in `${MODELS_DIR}/madlad400-3b-mt-onnx/`:
- `encoder_model.onnx` (~4 GB, fp32)
- `decoder_model.onnx` (~4 GB, fp32)
- `decoder_with_past_model.onnx` (~4 GB, fp32)
- `config.json`, `tokenizer.json`

Total disk: ~12 GB. Export requires ~8 GB RAM and 10–30 min.

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

Key CLI flags: `--beam N` (omit for auto-selection; 0/1 = greedy, 2/4 = beam search), `--models-dir`, `--output json`.
Key API env vars: `MODELS_DIR`, `BEAM_WIDTH` (omit for auto-selection; 0/1 = greedy), `RUST_LOG`.

## Architecture

3-crate workspace:

- **`translator-core`** — library crate: engine, model, detector, types, error
- **`translator-cli`** — `ut` binary: Clap CLI with `translate`, `detect`, `languages` subcommands
- **`translator-api`** — Axum HTTP server: `POST /translate`, `GET /languages`, `GET /health`

### Inference stack

- **Model**: MADLAD-400-3B-MT — single seq2seq model for all 62 languages
- **Framework**: ONNX Runtime via the `ort` crate (v2), with 3 ONNX sessions
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
- `beam_width` is startup config only — not part of the JSON request body; omit for auto-selection (tiers: greedy ≤15 tokens, beam=2 15–40 tokens, beam=4 >40 tokens)

### ORT Session structure (`model.rs`)

Three ORT sessions loaded from the ONNX model directory:
- `encoder_model.onnx` — `input_ids [B,S]`, `attention_mask [B,S]` → `last_hidden_state [B,S,d]`
- `decoder_model.onnx` — first decode step (no past KV); outputs `logits [B,1,V]` + 96 `present_key_values` tensors
- `decoder_with_past_model.onnx` — subsequent steps; takes 96 `past_key_values` tensors, outputs same shape present KVs

KV cache layout per layer: `past_key_values.{N}.decoder.key/value` [B, heads, seq, head_dim] (grows each step) and `past_key_values.{N}.encoder.key/value` [B, heads, enc_seq, head_dim] (constant; shared via `Arc` in beam search).

### Execution providers (feature flags)

| Feature | Providers registered | Platform |
|---------|---------------------|----------|
| (none)  | CPU only            | all      |
| `coreml`| CoreML (ANE/GPU) → CPU | macOS |
| `cuda`  | TensorRT → CUDA → CPU | NVIDIA |

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Ubuntu (x86_64 + arm64) and macOS (Apple Silicon):
`cargo build` → `cargo test --workspace` → `cargo clippy --workspace -- -D warnings`
