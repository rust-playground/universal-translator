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
        let translate_language = code.parse::<Language>().ok();

        match self.output {
            OutputFormat::Json => println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "language": code,
                    "translate_language": translate_language,
                    "confidence": confidence,
                }))?
            ),
            OutputFormat::Pretty => match translate_language {
                Some(lang) if lang.code() != code => {
                    // Detector emitted an alias (e.g. "nb"); show the mapping.
                    println!(
                        "Language: {} ({}) — translate as: {} — confidence: {:.1}%",
                        language_name,
                        code,
                        lang.code(),
                        confidence * 100.0,
                    );
                }
                Some(_) => {
                    println!(
                        "Language: {} ({}) — confidence: {:.1}%",
                        language_name,
                        code,
                        confidence * 100.0,
                    );
                }
                None => {
                    println!(
                        "Language: {} ({}) — confidence: {:.1}% — translation supported: no",
                        language_name,
                        code,
                        confidence * 100.0,
                    );
                }
            },
        }
        Ok(())
    }
}
