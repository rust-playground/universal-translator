use anyhow::Result;
use clap::{Args, ValueEnum};
use lingua::Language;
use std::collections::HashMap;
use translator_core::engine::supported_target_languages;

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

        // Build code → display-name from lingua, then fill in any gaps for languages
        // that are valid translation targets but not in lingua's detectable set.
        // Malayalam (ml) is detectable via script analysis but not named by lingua.
        let mut name_map: HashMap<String, String> = Language::all()
            .into_iter()
            .map(|l| {
                let code = format!("{:?}", l.iso_code_639_1()).to_lowercase();
                let name = format!("{l:?}").to_lowercase();
                (code, name)
            })
            .collect();
        name_map.entry("en".into()).or_insert_with(|| "english".into());
        name_map.entry("ml".into()).or_insert_with(|| "malayalam".into());

        let entries: Vec<(&str, String)> = supported_target_languages()
            .iter()
            .copied()
            .map(|code| {
                let name = name_map.get(code).cloned().unwrap_or_else(|| code.to_string());
                (code, name)
            })
            .filter(|(code, name)| {
                filter.is_empty() || code.contains(filter.as_str()) || name.contains(filter.as_str())
            })
            .collect();
        // already sorted by supported_target_languages()

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
