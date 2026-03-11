# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] — 2026-03-10

### Changed

- **Paragraph-first text chunking** — long inputs now split at `\n\n` paragraph
  boundaries first, falling back to Unicode sentence boundaries only for
  oversized paragraphs; keeps related sentences together for better translation
  context
- **Configurable chunk limits** — `MAX_CHUNK_CHARS` (hard ceiling) and
  `PARAGRAPH_TARGET_CHARS` (~60% default, quality-oriented) tuneable via env
  vars or CLI/API flags
- **Chunking shared across CLI and API** — logic moved to `translator-core`;
  CLI now handles long inputs instead of failing
- **Chunk reassembly preserves structure** — translated chunks rejoin with
  correct `\n\n` separators so paragraph boundaries survive round-trip
  translation

## [0.0.2] — 2026-03-10

### Added

- **Text chunking** — long inputs automatically split at Unicode sentence boundaries,
  translated per chunk, and reassembled; invisible to API clients
  (`translator-api/src/routes/translate.rs`)
- **Prefill accumulation window** — configurable delay (`PREFILL_ACCUMULATION_MS`,
  default 10 ms) coalesces concurrent requests into a single batched prefill
- **`EngineConfig` struct** — declarative engine initialisation via `from_config()`;
  all scheduler parameters (slots, KV budget, queue capacity, prefill delay) now
  configurable
- **Flash attention** enabled by default in llama.cpp context
- **`SchedulerGuard`** — deterministic scheduler-thread cleanup on drop
- **`InputTooLong` error** — scheduler rejects prompts exceeding the per-slot KV budget

### Changed

- **Inference backend → llama.cpp** — replaced Candle with `llama-cpp-2` Rust crate
  (vendored llama.cpp C++); tokenizer now embedded in GGUF (no separate
  `tokenizer.json`)
- **Continuous-batching scheduler** — single `LlamaBatch` + `ctx.decode()` call per
  step across all active slots; KV cache managed by llama.cpp internally
- **Default model → Q8_0** — higher-precision quantisation preferred over Q4_K
  (comparable throughput under llama.cpp's batching, better quality)
- **Language detection parallelised** via Rayon (`par_iter`) instead of serial loop
- **Eager engine init** — model and scheduler thread spawn at construction (was lazy)
- **Auto slot count** — compile-time default per backend (Metal 32, CUDA 64, CPU 4);
  overridable via `MAX_DECODE_SLOTS` env var

### Removed

- **Candle backend** — `candle-core`, `candle-transformers`, `candle-nn`, `tokenizers`
  dependencies dropped
- `model_batched.rs` (~715 lines) and `scheduler/decoder.rs` (~85 lines) — replaced by
  llama.cpp native batching
- Manual per-slot KV cache management (now internal to llama.cpp)

## [0.0.1] — 2026-03-05

### Added

- **TranslateGemma 4B inference engine** — llama.cpp-based (via `llama-cpp-2` Rust crate) local
  inference using GGUF quantised weights (Q8_0 default ~4.1 GB, Q4_K_M opt-in ~2.6 GB); supports
  CPU, Metal (macOS), and CUDA (Linux/NVIDIA) backends
- **Continuous-batching scheduler** — 24-slot batched decode via `LlamaContext`; llama.cpp-managed
  KV cache; EOS/length-penalty sampling
- **55-language support** — full TranslateGemma language set with Lingua-based automatic
  language detection and script-based fallback for Malayalam
- **CLI** (`ut` binary) — `translate`, `detect`, and `languages` subcommands; `--output json`
  flag; `--models-dir` / `MODELS_DIR` override
- **Axum HTTP API** — `POST /translate`, `GET /languages`, `GET /health`,
  `POST /detect-language` endpoints; runs on port 3000 by default
- **OpenTelemetry observability** (optional `opentelemetry` feature flag) — OTLP export of
  metrics, distributed traces, and logs; instruments for request counts, batch size,
  translation latency, scheduler slot utilisation, prefill time, prompt tokens, and error rates
- **Docker Compose observability stack** — OTel Collector, Prometheus, Grafana, Tempo, and
  Loki wired together for local development



[Unreleased]: https://github.com/rust-playground/universal-translator/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/rust-playground/universal-translator/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/rust-playground/universal-translator/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/rust-playground/universal-translator/compare/0044ce9fe79ee4ea73fa57dca12485b0bd22a5fb...v0.0.1