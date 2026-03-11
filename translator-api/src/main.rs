use std::path::PathBuf;

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use translator_core::{engine::TranslationEngine, EngineConfig};

mod error;
mod routes;
mod state;

use state::AppState;

fn default_models_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("ut/models")
}

#[derive(Parser)]
#[command(name = "translator-api", about = "Universal translation HTTP API")]
struct Args {
    /// Directory containing model files.
    /// [default: platform cache dir / ut/models]
    #[arg(long, env = "MODELS_DIR")]
    models_dir: Option<PathBuf>,

    /// GGUF model file name (e.g. "model-q8_0.gguf").
    /// Overrides auto-detection. Also settable via MODEL_FILE env var.
    #[arg(long, env = "MODEL_FILE")]
    model_file: Option<String>,

    /// Number of concurrent decode slots.
    #[arg(long, env = "MAX_DECODE_SLOTS")]
    n_slots: Option<usize>,

    /// Maximum tokens per translation request.
    #[arg(long, env = "KV_BUDGET_PER_SLOT")]
    max_tokens: Option<u32>,

    /// Prefill accumulation delay in milliseconds.
    #[arg(long, env = "PREFILL_ACCUMULATION_MS")]
    prefill_delay_ms: Option<u64>,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

#[cfg(not(feature = "opentelemetry"))]
fn init_tracing() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[cfg(feature = "opentelemetry")]
fn init_telemetry() -> opentelemetry_sdk::metrics::SdkMeterProvider {
    use opentelemetry::trace::TracerProvider as _; // bring .tracer() into scope
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::{
        logs::SdkLoggerProvider, metrics::SdkMeterProvider, trace::SdkTracerProvider,
    };

    let endpoint = std::env::var("OTLP_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:4317".to_string());

    // ── Traces ────────────────────────────────────────────────────────────
    let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("trace exporter");
    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(trace_exporter)
        .build();
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());
    let tracer = tracer_provider.tracer("translator-api");

    // ── Metrics ───────────────────────────────────────────────────────────
    let metrics_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("metrics exporter");
    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(metrics_exporter)
        .with_interval(std::time::Duration::from_secs(10))
        .build();
    let meter_provider = SdkMeterProvider::builder().with_reader(reader).build();
    opentelemetry::global::set_meter_provider(meter_provider.clone());

    // ── Logs ──────────────────────────────────────────────────────────────
    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("log exporter");
    let logger_provider = SdkLoggerProvider::builder()
        .with_batch_exporter(log_exporter)
        .build();
    // Bridge: tracing events → OTel logs
    let log_layer = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
        &logger_provider,
    );

    // ── Subscriber ────────────────────────────────────────────────────────
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer()) // console
        .with(tracing_opentelemetry::layer().with_tracer(tracer)) // traces → OTLP
        .with(log_layer) // logs → OTLP
        .init();

    tracing::info!(%endpoint, "OpenTelemetry OTLP telemetry enabled (traces + metrics + logs)");
    meter_provider
}

#[tokio::main]
async fn main() {
    #[cfg(feature = "opentelemetry")]
    let meter_provider = init_telemetry();
    #[cfg(not(feature = "opentelemetry"))]
    init_tracing();

    let args = Args::parse();
    let models_dir = args.models_dir.unwrap_or_else(default_models_dir);
    let addr = format!("0.0.0.0:{}", args.port);

    tracing::info!(?models_dir, "Loading translation engine");

    let config = EngineConfig {
        models_dir,
        model_file: args.model_file,
        n_slots: args.n_slots,
        max_tokens: args.max_tokens,
        queue_capacity: None,
        prefill_delay_ms: args.prefill_delay_ms,
    };
    let engine = TranslationEngine::from_config(config).unwrap_or_else(|e| {
        tracing::error!("Failed to load model: {e}");
        std::process::exit(1);
    });
    let state = AppState {
        engine,
        #[cfg(feature = "opentelemetry")]
        error_ctr: opentelemetry::global::meter("translator")
            .u64_counter("translator.translation.errors")
            .build(),
    };

    let app = Router::new()
        .route("/translate", post(routes::translate::translate))
        .route("/detect-language", post(routes::detect_language::detect_language))
        .route("/languages", get(routes::languages::languages))
        .route("/health", get(|| async { "OK" }))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    #[cfg(feature = "opentelemetry")]
    let _ = meter_provider.shutdown();
}
