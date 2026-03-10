use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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

    #[command(subcommand)]
    command: commands::Commands,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let models_dir = cli.models_dir.unwrap_or_else(default_models_dir);
    cli.command.run(&models_dir, cli.model_file.as_deref())
}
