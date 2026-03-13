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

fn default_model_path() -> PathBuf {
    default_models_dir().join("translategemma-4b/model-q8_0.gguf")
}

#[derive(Parser)]
#[command(name = "ut", about = "Universal translation CLI")]
struct Cli {
    /// Path to the GGUF model file.
    /// [default: <cache>/ut/models/translategemma-4b/model-q8_0.gguf]
    #[arg(long, env = "MODEL_PATH")]
    model_path: Option<PathBuf>,

    /// Number of concurrent decode slots.
    #[arg(long, env = "MAX_DECODE_SLOTS")]
    n_slots: Option<usize>,

    /// Maximum tokens per translation request.
    #[arg(long, env = "KV_BUDGET_PER_SLOT")]
    max_tokens: Option<u32>,

    /// Prefill accumulation delay in milliseconds.
    #[arg(long, env = "PREFILL_ACCUMULATION_MS")]
    prefill_delay_ms: Option<u64>,

    /// Hard ceiling (in characters) for text chunks sent to the model.
    /// Defaults to (KV budget − 100) × 4.
    #[arg(long, env = "MAX_CHUNK_CHARS")]
    max_chunk_chars: Option<usize>,

    /// Target size (in characters) for paragraph-level chunk packing.
    /// Shorter chunks improve translation quality. Defaults to ~60% of max-chunk-chars.
    #[arg(long, env = "PARAGRAPH_TARGET_CHARS")]
    paragraph_target_chars: Option<usize>,

    /// Bounded queue capacity for pending translation requests.
    #[arg(long, env = "QUEUE_CAPACITY")]
    queue_capacity: Option<usize>,

    /// Timeout (in seconds) when sending work items to the scheduler queue.
    /// Allows requests to wait for capacity instead of failing instantly. Default: 30.
    #[arg(long, env = "QUEUE_SEND_TIMEOUT_SECS")]
    queue_send_timeout_secs: Option<u64>,

    #[command(subcommand)]
    command: commands::Commands,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let model_path = cli.model_path.unwrap_or_else(default_model_path);
    let config = EngineConfig {
        model_path: model_path.clone(),
        n_slots: cli.n_slots,
        max_tokens: cli.max_tokens,
        queue_capacity: cli.queue_capacity,
        prefill_delay_ms: cli.prefill_delay_ms,
        max_chunk_chars: cli.max_chunk_chars,
        paragraph_target_chars: cli.paragraph_target_chars,
        queue_send_timeout_secs: cli.queue_send_timeout_secs,
    };
    cli.command.run(config, &default_models_dir())
}
