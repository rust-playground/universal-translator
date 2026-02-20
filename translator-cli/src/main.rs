use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use translator_core::engine::TranslationEngine;

mod commands;

#[derive(Parser)]
#[command(name = "translator", about = "Universal translation CLI")]
struct Cli {
    /// Directory containing language-pair model directories.
    #[arg(long, default_value = "./models")]
    models_dir: PathBuf,

    #[command(subcommand)]
    command: commands::Commands,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let engine = TranslationEngine::new(&cli.models_dir);
    cli.command.run(engine).await
}
