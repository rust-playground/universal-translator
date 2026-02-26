use std::path::PathBuf;

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use translator_core::engine::TranslationEngine;

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
#[command(name = "translator-api")]
struct Args {
    /// Directory containing model files.
    /// [default: ~/.cache/ut/models  (macOS: ~/Library/Caches/ut/models)]
    #[arg(long, env = "MODELS_DIR")]
    models_dir: Option<PathBuf>,

    /// Beam width for decoding. 0 = greedy, default 4.
    #[arg(long, default_value = "4", env = "BEAM_WIDTH")]
    beam: u8,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let models_dir = args.models_dir.unwrap_or_else(default_models_dir);
    tracing::info!(?models_dir, beam = args.beam, "Loading translation engine");

    let engine = TranslationEngine::new(&models_dir, args.beam);
    let state = AppState { engine };

    let app = Router::new()
        .route("/translate", post(routes::translate::translate))
        .route("/languages", get(routes::languages::languages))
        .route("/health", get(|| async { "OK" }))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:3000";
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
