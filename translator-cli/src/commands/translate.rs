use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::{engine::TranslationEngine, types::TranslationBatch};

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Pretty,
    Json,
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

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl TranslateArgs {
    pub async fn run(self, engine: TranslationEngine) -> Result<()> {
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
