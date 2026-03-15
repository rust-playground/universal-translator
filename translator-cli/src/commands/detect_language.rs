use std::path::Path;

use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::detector::Detector;
use translator_core::Language;

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
        let lang = code.parse::<Language>().ok();

        match self.output {
            OutputFormat::Json => println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "language": lang,
                    "confidence": confidence,
                    "translation_supported": lang.is_some(),
                }))?
            ),
            OutputFormat::Pretty => {
                let supported = if lang.is_some() { "yes" } else { "no" };
                let lang_display = lang.map(|l| l.code()).unwrap_or(&code);
                println!(
                    "Language: {} ({}) — confidence: {:.1}% — translation supported: {}",
                    language_name,
                    lang_display,
                    confidence * 100.0,
                    supported,
                );
            }
        }
        Ok(())
    }
}
