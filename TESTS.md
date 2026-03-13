# Tests

## Build

Choose the variant that matches your hardware:

```bash
cargo build --release                        # CPU (default)
cargo build --release --features metal       # macOS GPU (Apple Silicon / Metal)
cargo build --release --features cuda        # Linux GPU (NVIDIA / CUDA)
```

For quick correctness checks a debug build (`cargo build`, without `--release`) is fine.
Release builds are required for representative load-test results.

## Unit Tests

```bash
cargo test --workspace
```

## Lint

```bash
cargo clippy --workspace -- -D warnings
```

## Integration Tests (CLI golden-fixture)

Tests CLI output against a committed CSV of expected translations.

```bash
# Build the CLI first
# Add --features metal (macOS) or --features cuda (Linux) for GPU inference
cargo build -r -p translator-cli

# Run tests against golden fixtures
python3 tests/integration.py --binary ./target/debug/ut

# (Re)generate golden fixtures from current CLI output
python3 tests/integration.py --binary ./target/debug/ut --seed

# Show full actual vs expected on failures
python3 tests/integration.py --binary ./target/debug/ut --verbose
```

Requires: models downloaded via `bash models/download.sh` (see docs/models.md).

## Load Tests (API throughput/latency)

Rust binary that measures throughput and latency under concurrent load across all API endpoints.
Uses the `translator-api-client` crate — no external dependencies required.

```bash
# Start the API server first
# Add --features metal (macOS) or --features cuda (Linux) for GPU inference
cargo build --release -p translator-api
RUST_LOG=info ./target/release/translator-api &

# All scenarios with defaults
cargo run -p translator-api-client --bin load-test

# Specific scenarios
cargo run -p translator-api-client --bin load-test -- translate-en-fr detect

# Custom concurrency and request count
cargo run -p translator-api-client --bin load-test -- --concurrency 20 --requests 500 translate-en-all

# Release mode (minimal client overhead)
cargo run -p translator-api-client --bin load-test --release -- --requests 200
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--url URL` | `http://localhost:3000` | API base URL |
| `--concurrency N` | `10` | Max concurrent in-flight requests |
| `--requests N` | `100` | Total requests per scenario |
| `--warmup N` | `3` | Sequential warmup requests before measurement |

Scenarios (positional args, default: all):

| Scenario | Endpoint | Description |
|----------|----------|-------------|
| `translate-en-fr` | `POST /translate` | Single target language (fr) — isolates per-request latency |
| `translate-en-all` | `POST /translate` | All 54 non-English languages — measures fan-out / batch efficiency |
| `detect` | `POST /detect-language` | Language detection |
| `languages` | `GET /languages` | List supported languages |
| `health` | `GET /health` | Health check |
