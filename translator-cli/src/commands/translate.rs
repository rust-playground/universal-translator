use anyhow::Result;
use clap::Args;
use translator_core::{Language, engine::TranslationEngine, types::TranslationBatch, EngineConfig};

#[derive(Clone, clap::ValueEnum)]
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
    pub fn run(self, config: EngineConfig) -> Result<()> {
        let engine = TranslationEngine::from_config(config)?;

        let target_languages = if self.languages == ["all"] {
            Language::all().to_vec()
        } else {
            self.languages
                .iter()
                .map(|s| s.parse::<Language>())
                .collect::<Result<Vec<_>, _>>()?
        };

        let source_language = self
            .source_language
            .as_deref()
            .map(|s| s.parse::<Language>())
            .transpose()?;

        let batch = TranslationBatch {
            texts: self.texts,
            target_languages,
            source_language,
        };

        let result = engine.translate_batch_chunked(batch)?;

        match self.output {
            OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
            OutputFormat::Pretty => {
                for r in &result.results {
                    let src = r
                        .detected_language
                        .map(|l| l.code())
                        .unwrap_or("unknown");
                    println!("Source [{src}]: {}", r.source_text);
                    for (lang, translation) in &r.translations {
                        println!("  [{}] {translation}", lang.code());
                    }
                    for (lang, err) in &r.errors {
                        println!("  [{}] ERROR: {err}", lang.code());
                    }
                    println!();
                }
            }
        }

        Ok(())
    }
}
