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

Requires: `pip install huggingface_hub[cli]` and models downloaded via `bash models/download.sh`.

## Load Tests (API throughput/latency)

Measures throughput and latency under concurrent load. Two scenarios:

- **en-fr** — many concurrent requests, one source text → French. Isolates single-pair throughput and per-request latency.
- **en-all** — many concurrent requests, one source text → all 61 non-English languages. Measures fan-out throughput and batch efficiency.

```bash
# Requires: pip install aiohttp

# Start the API server first
# Add --features metal (macOS) or --features cuda (Linux) for GPU inference
cargo build --release -p translator-api
RUST_LOG=info ./target/release/translator-api &

# Quick smoke run (verifies the script works)
python3 tests/load_test.py --requests 10 --warmup 1 --scenario en-fr

# Full throughput test (both scenarios)
python3 tests/load_test.py --concurrency 20 --requests 200 --scenario both

# High-concurrency fan-out test
python3 tests/load_test.py --concurrency 5 --requests 30 --scenario en-all

# Custom API URL
python3 tests/load_test.py --url http://localhost:8080 --scenario en-fr
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--url URL` | `http://localhost:3000` | API base URL |
| `--concurrency N` | `10` | Max concurrent in-flight requests |
| `--requests N` | `100` | Total requests per scenario |
| `--warmup N` | `3` | Sequential warmup requests before measurement |
| `--scenario` | `both` | `en-fr` \| `en-all` \| `both` |

Example output:

```
=== Scenario: en-fr (1 language) ===
Concurrency: 20  |  Requests: 200  |  Warmup: 3
Duration:   18.4s
Throughput: 10.87 req/s  |  10.9 translations/s
Latency (ms): min=180  mean=512  p50=490  p75=640  p95=920  p99=1150  max=1380
Errors: 0 / 200

=== Scenario: en-all (61 languages) ===
Concurrency: 10  |  Requests: 50  |  Warmup: 3
Duration:   68.2s
Throughput: 0.73 req/s  |  44.6 translations/s
Latency (ms): min=2100  mean=5800  p50=5600  p75=7100  p95=9200  p99=10100  max=11000
Errors: 0 / 50
```
