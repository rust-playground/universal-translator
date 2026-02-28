use std::path::PathBuf;

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use translator_core::engine::{DecodeMode, TranslationEngine};

mod error;
mod routes;
mod state;

use state::AppState;

fn default_models_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("ut/models")
}

#[derive(Clone, clap::ValueEnum)]
enum DecodeModeArg {
    /// Greedy decoding — maximum throughput.
    Greedy,
    /// Beam search with width 2 (reserved for Phase 2 custom decoder).
    Beam2,
}

impl From<DecodeModeArg> for DecodeMode {
    fn from(m: DecodeModeArg) -> Self {
        match m {
            DecodeModeArg::Greedy => DecodeMode::Greedy,
            DecodeModeArg::Beam2 => DecodeMode::Beam2,
        }
    }
}

#[derive(Parser)]
#[command(name = "translator-api", about = "Universal translation HTTP API")]
struct Args {
    /// Directory containing model files.
    /// [default: platform cache dir / ut/models]
    #[arg(long, env = "MODELS_DIR")]
    models_dir: Option<PathBuf>,

    /// Decode strategy: greedy (fastest) or beam2 (width-2 beam search, reserved for Phase 2).
    #[arg(long = "decode-mode", env = "DECODE_MODE", default_value = "greedy")]
    decode_mode: DecodeModeArg,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let models_dir = args.models_dir.unwrap_or_else(default_models_dir);
    let decode_mode: DecodeMode = args.decode_mode.into();
    let addr = format!("0.0.0.0:{}", args.port);

    tracing::info!(?models_dir, ?decode_mode, "Loading translation engine");

    let engine = TranslationEngine::new(&models_dir, decode_mode);
    let state = AppState { engine };

    let app = Router::new()
        .route("/translate", post(routes::translate::translate))
        .route("/languages", get(routes::languages::languages))
        .route("/health", get(|| async { "OK" }))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
