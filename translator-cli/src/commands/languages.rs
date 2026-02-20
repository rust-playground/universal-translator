use anyhow::Result;
use clap::{Args, ValueEnum};
use lingua::Language;
use std::collections::HashMap;
use translator_core::engine::supported_languages;

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

        // Build code → display-name from lingua (covers all intersection members).
        let name_map: HashMap<String, String> = Language::all()
            .into_iter()
            .map(|l| {
                let code = format!("{:?}", l.iso_code_639_1()).to_lowercase();
                let name = format!("{l:?}").to_lowercase();
                (code, name)
            })
            .collect();

        let entries: Vec<(&str, String)> = supported_languages()
            .into_iter()
            .map(|code| {
                let name = name_map.get(code).cloned().unwrap_or_else(|| code.to_string());
                (code, name)
            })
            .filter(|(code, name)| {
                filter.is_empty() || code.contains(filter.as_str()) || name.contains(filter.as_str())
            })
            .collect();
        // already sorted by supported_languages()

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
