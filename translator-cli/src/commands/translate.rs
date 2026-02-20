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

    /// Target language ISO 639-1 code (repeat for multiple).
    #[arg(short = 'l', long = "language", required = true)]
    pub languages: Vec<String>,

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl TranslateArgs {
    pub async fn run(self, engine: TranslationEngine) -> Result<()> {
        let batch = TranslationBatch {
            texts: self.texts,
            target_languages: self.languages,
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
