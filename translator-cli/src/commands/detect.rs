use std::path::Path;

use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::engine::TranslationEngine;

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Pretty,
    Json,
}

#[derive(Args)]
pub struct DetectArgs {
    /// Text whose language to detect.
    pub text: String,

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl DetectArgs {
    pub fn run(self, models_dir: &Path) -> Result<()> {
        let engine = TranslationEngine::new(models_dir);
        let detected = engine.detect_language(&self.text)?;

        match self.output {
            OutputFormat::Json => println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "text": self.text,
                    "detected_language": detected,
                }))?
            ),
            OutputFormat::Pretty => println!("Detected language: {detected}"),
        }

        Ok(())
    }
}
