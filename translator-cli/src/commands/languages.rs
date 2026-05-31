use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::detector::detect_supported_codes;
use translator_core::Language;

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Pretty,
    Json,
}

/// Which language list to display.
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum For {
    /// Translate-supported codes (the `Language` enum).
    Translate,
    /// Detect-supported codes (broader: lingua + script + heuristic refinements).
    Detect,
}

#[derive(Args)]
pub struct LanguagesArgs {
    /// Filter output to entries containing this string.
    #[arg(long)]
    pub filter: Option<String>,

    /// Which use case to list.
    #[arg(long = "for", value_enum, default_value = "translate")]
    pub r#for: For,

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl LanguagesArgs {
    pub fn run(self) -> Result<()> {
        let filter = self.filter.as_deref().unwrap_or("").to_lowercase();

        let entries: Vec<(&str, &str)> = match self.r#for {
            For::Translate => Language::all()
                .iter()
                .map(|lang| (lang.code(), lang.full_name()))
                .filter(|(code, name)| {
                    filter.is_empty()
                        || code.to_lowercase().contains(filter.as_str())
                        || name.to_lowercase().contains(filter.as_str())
                })
                .collect(),
            For::Detect => detect_supported_codes()
                .iter()
                .copied()
                .filter(|(code, name)| {
                    filter.is_empty()
                        || code.to_lowercase().contains(filter.as_str())
                        || name.to_lowercase().contains(filter.as_str())
                })
                .collect(),
        };

        match self.output {
            OutputFormat::Pretty => {
                for (code, name) in &entries {
                    println!("{code:<10} {name}");
                }
                println!("\n{} language(s)", entries.len());
            }
            OutputFormat::Json => {
                let json: Vec<_> = entries
                    .iter()
                    .map(|(code, name)| serde_json::json!({"code": code, "name": name}))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
        }
        Ok(())
    }
}
