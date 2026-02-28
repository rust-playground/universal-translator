use std::path::Path;

use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::{
    engine::{DecodeMode, TranslationEngine},
    types::TranslationBatch,
};

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Pretty,
    Json,
}

#[derive(Clone, ValueEnum)]
enum DecodeModeArg {
    /// Greedy decoding — maximum throughput.
    Greedy,
    /// Beam search with width 2 (reserved for Phase 2 custom decoder).
    Beam2,
}

impl From<DecodeModeArg> for DecodeMode {
    fn from(m: DecodeModeArg) -> Self {
        match m {
            DecodeModeArg::Greedy => DecodeMode::Greedy,
            DecodeModeArg::Beam2 => DecodeMode::Beam2,
        }
    }
}

#[derive(Args)]
pub struct TranslateArgs {
    /// Text to translate (repeat for multiple).
    #[arg(short = 't', long = "text", required = true)]
    pub texts: Vec<String>,

    /// Target language ISO 639-1 code (repeat or comma-separate: -l fr,de or -l fr -l de).
    #[arg(short = 'l', long = "language", required = true, value_delimiter = ',')]
    pub languages: Vec<String>,

    /// Source language ISO 639-1 code. When provided, skips auto-detection.
    /// All texts in the batch are assumed to be in this language.
    #[arg(short = 's', long = "source")]
    pub source_language: Option<String>,

    /// Decode strategy: greedy (fastest) or beam2 (width-2 beam search, reserved for Phase 2).
    #[arg(long = "decode-mode", env = "DECODE_MODE", default_value = "greedy")]
    decode_mode: DecodeModeArg,

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl TranslateArgs {
    pub async fn run(self, models_dir: &Path) -> Result<()> {
        let engine = TranslationEngine::new(models_dir, self.decode_mode.into());
        let batch = TranslationBatch {
            texts: self.texts,
            target_languages: self.languages,
            source_language: self.source_language,
        };

        let result = engine.translate_batch(batch).await?;

        match self.output {
            OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
            OutputFormat::Pretty => {
                for r in &result.results {
                    println!("Source [{}]: {}", r.detected_language, r.source_text);
                    for (lang, translation) in &r.translations {
                        println!("  [{lang}] {translation}");
                    }
                    for (lang, err) in &r.errors {
                        println!("  [{lang}] ERROR: {err}");
                    }
                    println!();
                }
            }
        }

        Ok(())
    }
}
