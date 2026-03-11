use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use translator_core::EngineConfig;

mod commands;

fn default_models_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("ut/models")
}

#[derive(Parser)]
#[command(name = "ut", about = "Universal translation CLI")]
struct Cli {
    /// Directory containing language-pair model directories.
    /// [default: ~/.cache/ut/models  (macOS: ~/Library/Caches/ut/models)]
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// GGUF model file name within the model directory (e.g. "model-q8_0.gguf").
    /// Overrides MODEL_FILE env var and auto-detection.
    #[arg(long)]
    model_file: Option<String>,

    /// Number of concurrent decode slots.
    #[arg(long, env = "MAX_DECODE_SLOTS")]
    n_slots: Option<usize>,

    /// Maximum tokens per translation request.
    #[arg(long, env = "KV_BUDGET_PER_SLOT")]
    max_tokens: Option<u32>,

    /// Bounded queue capacity for pending translation requests.
    #[arg(long, env = "QUEUE_CAPACITY")]
    queue_capacity: Option<usize>,

    /// Prefill accumulation delay in milliseconds.
    #[arg(long, env = "PREFILL_ACCUMULATION_MS")]
    prefill_delay_ms: Option<u64>,

    #[command(subcommand)]
    command: commands::Commands,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let config = EngineConfig {
        models_dir: cli.models_dir.unwrap_or_else(default_models_dir),
        model_file: cli.model_file,
        n_slots: cli.n_slots,
        max_tokens: cli.max_tokens,
        queue_capacity: cli.queue_capacity,
        prefill_delay_ms: cli.prefill_delay_ms,
    };
    cli.command.run(config)
}
