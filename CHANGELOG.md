# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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



[Unreleased]: https://github.com/rust-playground/universal-translator/compare/v0.0.1...HEAD

[0.0.1]: https://github.com/rust-playground/universal-translator/compare/0044ce9fe79ee4ea73fa57dca12485b0bd22a5fb...v0.0.1