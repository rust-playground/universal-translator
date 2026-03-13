use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;

const DEFAULT_GGUF_URL: &str =
    "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q8_0.gguf";

#[derive(Args)]
pub struct SetupArgs {
    /// URL to download the GGUF model file from.
    #[arg(long, default_value = DEFAULT_GGUF_URL)]
    pub url: String,

    /// Output file path for the downloaded model.
    /// [default: <cache>/ut/models/translategemma-4b/model-q8_0.gguf]
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Re-download even if the file already exists.
    #[arg(long)]
    pub force: bool,
}

impl SetupArgs {
    pub fn run(self, default_models_dir: &Path) -> Result<()> {
        let output = self
            .output
            .unwrap_or_else(|| default_models_dir.join("translategemma-4b/model-q8_0.gguf"));

        if output.exists() && !self.force {
            println!("Model already exists: {}", output.display());
            println!("Use --force to re-download.");
            return Ok(());
        }

        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(download(&self.url, &output))?;

        let size_mb = std::fs::metadata(&output)?.len() as f64 / (1024.0 * 1024.0);
        println!();
        println!("Model saved: {}", output.display());
        println!("Size: {size_mb:.1} MB");
        println!();
        println!("Next steps:");
        println!("  cargo run -p translator-cli -- translate -t \"Hello\" -l fr");

        Ok(())
    }
}

async fn download(url: &str, output: &Path) -> Result<()> {
    use futures::StreamExt;
    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::Write;

    println!("Downloading model...");
    println!("  URL:    {url}");
    println!("  Output: {}", output.display());
    println!();

    let client = reqwest::Client::new();
    let resp = client.get(url).send().await?.error_for_status()?;

    let total_size = resp.content_length().unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
                .progress_chars("#>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})")?,
        );
        pb
    };

    let tmp_path = output.with_extension("gguf.tmp");
    let mut file = std::fs::File::create(&tmp_path)?;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        pb.inc(chunk.len() as u64);
    }

    file.flush()?;
    drop(file);
    pb.finish_with_message("download complete");

    std::fs::rename(&tmp_path, output)?;

    Ok(())
}
