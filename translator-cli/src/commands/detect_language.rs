use std::path::Path;

use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::detector::Detector;
use translator_core::engine::supported_target_languages;

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
    pub fn run(self, _models_dir: &Path) -> Result<()> {
        let detector = Detector::new();
        let (code, language_name, confidence) = detector.detect_with_confidence(&self.text)?;
        let translation_supported = supported_target_languages().contains(&code.as_str());

        match self.output {
            OutputFormat::Json => println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "language_code": code,
                    "language": language_name,
                    "confidence": confidence,
                    "translation_supported": translation_supported,
                }))?
            ),
            OutputFormat::Pretty => {
                let supported = if translation_supported { "yes" } else { "no" };
                println!(
                    "Language: {} ({}) — confidence: {:.1}% — translation supported: {}",
                    language_name,
                    code,
                    confidence * 100.0,
                    supported,
                );
            }
        }
        Ok(())
    }
}
