use anyhow::Result;
use clap::{Args, ValueEnum};
use translator_core::Language;

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Pretty,
    Json,
}

#[derive(Args)]
pub struct LanguagesArgs {
    /// Filter output to entries containing this string.
    #[arg(long)]
    pub filter: Option<String>,

    #[arg(long, value_enum, default_value = "pretty")]
    pub output: OutputFormat,
}

impl LanguagesArgs {
    pub fn run(self) -> Result<()> {
        let filter = self.filter.as_deref().unwrap_or("").to_lowercase();

        let entries: Vec<(&str, &str)> = Language::all()
            .iter()
            .map(|lang| (lang.code(), lang.full_name()))
            .filter(|(code, name)| {
                filter.is_empty()
                    || code.contains(filter.as_str())
                    || name.to_lowercase().contains(filter.as_str())
            })
            .collect();

        match self.output {
            OutputFormat::Pretty => {
                for (code, name) in &entries {
                    println!("{code:<6} {name}");
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
