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
pub struct DetectLanguageArgs {
    /// Text whose language to detect.
    pub text: String,

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl DetectLanguageArgs {
    pub async fn run(self, models_dir: &Path) -> Result<()> {
        let engine = TranslationEngine::new(models_dir);
        let result = engine.detect_language_full(&self.text).await?;

        match self.output {
            OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
            OutputFormat::Pretty => {
                let supported = if result.translation_supported { "yes" } else { "no" };
                println!(
                    "Language: {} ({}) — confidence: {:.1}% — translation supported: {}",
                    result.language,
                    result.language_code,
                    result.confidence * 100.0,
                    supported,
                );
            }
        }
        Ok(())
    }
}
