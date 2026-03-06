# Observability

`translator-api` ships built-in OpenTelemetry instrumentation for traces, metrics, and logs.
All three signals are **pushed** via OTLP/gRPC from the running process to an OTel Collector,
which routes them to Prometheus (metrics), Tempo (traces), and Loki (logs).
A pre-built Grafana dashboard ties everything together.
The instrumentation is a **compile-time feature** (`opentelemetry`). The default build has
zero telemetry overhead — no extra dependencies are compiled in, no background goroutines,
no exporter threads.

---

## Quick start

### 1. Start the monitoring stack

```bash
docker-compose -f docker/docker-compose.yml up -d
```

| Service        | URL                   | Purpose                         |
|----------------|-----------------------|---------------------------------|
| Grafana        | http://localhost:3001 | Dashboards — no login required  |
| Prometheus     | http://localhost:9090 | Raw metric queries              |
| OTel Collector | localhost:4317 (gRPC) | Receives OTLP push from the API |

Grafana opens with the **Universal Translator** dashboard pre-loaded.
No login form — anonymous access with Admin role is provisioned automatically.

### 2. Run the API with telemetry enabled

```bash
# Build with the opentelemetry feature
cargo build -p translator-api --features opentelemetry

# Run — point at the collector started above
OTLP_ENDPOINT=http://localhost:4317 \
  ./target/debug/translator-api
```

Or in a single step:

```bash
OTLP_ENDPOINT=http://localhost:4317 \
  cargo run -p translator-api --features opentelemetry
```

On startup you will see:

```
INFO translator_api: OpenTelemetry OTLP telemetry enabled (traces + metrics + logs) endpoint=http://localhost:4317
INFO translator_api: Listening on 0.0.0.0:3000
```

### 3. Send requests and view data

```bash
curl -s -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{"texts":["Hello world"],"target_languages":["fr","de"]}'
```

- **Grafana → Dashboards → Universal Translator** — panels populate within one scrape interval (~15 s)
- **Grafana → Explore → Tempo** — select service `translator-api` to browse traces
- **Grafana → Explore → Loki** — filter `{service_name="translator-api"}` for structured log events

---

## Configuration

| Variable        | Default                      | Description                                                  |
|-----------------|------------------------------|--------------------------------------------------------------|
| `OTLP_ENDPOINT` | `http://localhost:4317`      | OTLP gRPC endpoint the API pushes signals to                 |
| `RUST_LOG`      | `info`                       | Log level filter (`trace`, `debug`, `info`, `warn`, `error`) |
| `MODELS_DIR`    | platform cache / `ut/models` | Model directory (unchanged from non-OTel build)              |

The `OTLP_ENDPOINT` must be reachable from the host where the API runs. When running the API
in Docker alongside the stack, use `http://otel-collector:4317` and attach it to the `obs`
network defined in `docker/docker-compose.yml`.

---

## Grafana dashboard

The provisioned dashboard (`uid = ut-observability-v1`) contains ten panels:

| Panel                               | Type        | What it shows                                       |
|-------------------------------------|-------------|-----------------------------------------------------|
| Request Rate                        | time series | Translations per second (`rate` over 1 m)           |
| Active Slots                        | gauge       | Number of decode slots currently in use (0–24)      |
| Queue Depth                         | stat        | Pending requests waiting for a slot                 |
| Translation Latency p50 / p95 / p99 | time series | End-to-end latency in ms per `translate_batch` call |
| Error Rate by Type                  | time series | Per-error-category rate — see error types below     |
| Tokens / s                          | time series | Generated tokens per second across all active slots |
| Slot Completions by Cause           | time series | EOS completions vs. capacity-limit truncations      |
| Decode Forward Latency p50 / p95    | time series | Time for one batched `forward_batched` call         |
| Prefill Latency p50 / p95           | time series | Time for `prefill` (prompt encoding + first token)  |
| Prompt Tokens p50 / p95             | time series | Token count of the formatted prompt per request     |

Dashboard JSON is at `docker/grafana/dashboards/universal-translator.json` and is provisioned
automatically on first `docker compose up`.

---

## Metrics catalogue

All metrics live under the `translator.*` namespace and are exported to Prometheus by the OTel
Collector. Prometheus converts `.` to `_` in metric names and appends `_total` to counters, so
`translator.translation.requests` becomes `translator_translation_requests_total`.

### Engine (`translator-core/src/engine.rs`)

| Metric                                  | Prometheus name                         | Type      | Labels | Description                                            |
|-----------------------------------------|-----------------------------------------|-----------|--------|--------------------------------------------------------|
| `translator.translation.requests`       | `translator_translation_requests_total` | Counter   | —      | Total `translate_batch` calls (non-empty batches only) |
| `translator.translation.batch_size`     | `translator_translation_batch_size`     | Histogram | —      | Number of source texts per batch                       |
| `translator.translation.duration_ms`    | `translator_translation_duration_ms`    | Histogram | —      | End-to-end wall-clock time per batch call, in ms       |

Histogram boundaries for `duration_ms`: 100, 250, 500, 1 000, 2 000, 5 000, 10 000, 30 000, 60 000, 120 000 ms.

### Scheduler (`translator-core/src/scheduler/continuous.rs`)

| Metric                                        | Prometheus name                               | Type      | Labels  | Description                                                                               |
|-----------------------------------------------|-----------------------------------------------|-----------|---------|-------------------------------------------------------------------------------------------|
| `translator.scheduler.active_slots`           | `translator_scheduler_active_slots`           | Gauge     | —       | Number of slots actively decoding at the start of each batch pass                         |
| `translator.scheduler.queue_depth`            | `translator_scheduler_queue_depth`            | Gauge     | —       | Pending requests in the work queue                                                        |
| `translator.scheduler.decode_forward_ms`      | `translator_scheduler_decode_forward_ms`      | Histogram | —       | Time for one batched `forward_batched` call, in ms                                        |
| `translator.scheduler.prefill_ms`             | `translator_scheduler_prefill_ms`             | Histogram | —       | Time for prompt prefill (encoding + first token sample), in ms                            |
| `translator.scheduler.prompt_tokens`          | `translator_scheduler_prompt_tokens`          | Histogram | —       | Number of prompt tokens per slot at prefill time                                          |
| `translator.scheduler.slots_completed`        | `translator_scheduler_slots_completed_total`  | Counter   | `cause` | Slots retired — `cause=eos` (natural end) or `cause=capacity` (truncated at 4 096 tokens) |
| `translator.scheduler.tokens_generated`       | `translator_scheduler_tokens_generated_total` | Counter   | —       | Total output tokens produced across all retired slots                                     |

Histogram boundaries:

- `decode_forward_ms`: 1, 5, 10, 25, 50, 100, 250, 500, 1 000, 2 500, 5 000 ms
- `prefill_ms`: 50, 100, 200, 500, 1 000, 2 000, 5 000, 10 000, 30 000 ms
- `prompt_tokens`: 10, 20, 50, 100, 200, 400, 600, 1 024, 2 048 tokens

### API layer (`translator-api/src/routes/translate.rs`)

| Metric                                | Prometheus name                       | Type    | Labels       | Description                                  |
|---------------------------------------|---------------------------------------|---------|--------------|----------------------------------------------|
| `translator.translation.errors`       | `translator_translation_errors_total` | Counter | `error_type` | Errors returned by the `/translate` endpoint |

`error_type` values: `model_not_found`, `detection_failed`, `unsupported_language`,
`translation_failed`, `io`, `model`.

### Useful PromQL queries

```promql
# Requests per second
rate(translator_translation_requests_total[1m])

# Translation latency percentiles
histogram_quantile(0.95, rate(translator_translation_duration_ms_bucket[5m]))

# Tokens per second (throughput)
rate(translator_scheduler_tokens_generated_total[1m])

# Proportion of slots completing naturally vs. being truncated
rate(translator_scheduler_slots_completed_total{cause="eos"}[5m])
  /
rate(translator_scheduler_slots_completed_total[5m])

# Error rate by type
rate(translator_translation_errors_total[5m])
```

---

## Traces

Each call to `translate_batch` produces a span named `translate_batch` with fields:

| Field       | Type    | Value                                |
|-------------|---------|--------------------------------------|
| `n_texts`   | integer | Number of source texts in the batch  |
| `n_targets` | integer | Number of target languages requested |

Each call to `prefill_slot` (inside the scheduler's blocking thread) produces a
`prefill_slot` span with fields:

| Field                 | Type    | Value                                                 |
|-----------------------|---------|-------------------------------------------------------|
| `prompt_len`          | integer | Byte length of the formatted prompt string            |
| `expected_output_len` | integer | Estimated output token count used for EOS calibration |
| `eos_token_id`        | integer | EOS token ID                                          |

HTTP spans (method, path, status code) are added automatically by `tower-http::TraceLayer`.
Spans are exported to Tempo via the OTel Collector. Browse them in
**Grafana → Explore → Tempo**, filtering by `service.name = translator-api`.

---

## Logs

All `tracing::info!` / `warn!` / `error!` events emitted by the API are forwarded to Loki
via the `opentelemetry-appender-tracing` bridge. Query them in
**Grafana → Explore → Loki**:

```logql
{service_name="translator-api"}
{service_name="translator-api"} |= "error"
{service_name="translator-api"} | json | level="WARN"
```

Trace IDs embedded in log events link directly to the corresponding Tempo span via the
configured derived field in the Loki datasource.

---

## Privacy

Translation inputs and outputs are **never** emitted to any telemetry signal:

- `translate_batch` is instrumented with `skip(self, batch)` — the batch texts are not
  recorded as span fields; only `n_texts` and `n_targets` (integer counts) are captured.
- `prefill_slot` skips the formatted prompt string; only `prompt_len` (byte count) is
  recorded.
- No `tracing::debug!` or `info!` call in the instrumented code emits translated content.
- All metric attribute values are language codes (`"eos"`, `"capacity"`, error type strings)
  — never text content.
- `tower-http::TraceLayer` records HTTP method, path, and status code only — never request
  or response bodies.

---

## Signal flow

```
translator-api (host)
  │
  │  OTLP/gRPC push (port 4317)
  ▼
otel-collector (Docker)
  ├── traces  ──► Tempo  (port 4317 internal) ◄── Grafana reads (port 3200)
  ├── metrics ──► Prometheus exporter (port 8889) ◄── Prometheus scrapes
  │                                                       ▼
  │                                               Grafana reads (port 9090)
  └── logs    ──► Loki (port 3100) ◄─────────── Grafana reads (port 3100)
```

No scrape endpoint is exposed by the API process. All data flows out via push.

---

## Building without telemetry

```bash
# Default build — no OTel deps compiled in, zero runtime overhead
cargo build -p translator-api
cargo run -p translator-api
```

`translator-cli` has no telemetry instrumentation and is unaffected by the feature in all
build configurations.

---

## Stopping the stack

```bash
docker-compose -f docker/docker-compose.yml down

# To also remove stored metric / trace / log data:
docker-compose -f docker/docker-compose.yml down -v
```
