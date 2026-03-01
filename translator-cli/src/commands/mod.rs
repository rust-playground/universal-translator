pub mod detect;
pub mod detect_language;
pub mod languages;
pub mod translate;

use std::path::Path;

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    Translate(translate::TranslateArgs),
    Detect(detect::DetectArgs),
    DetectLanguage(detect_language::DetectLanguageArgs),
    Languages(languages::LanguagesArgs),
}

impl Commands {
    pub async fn run(self, models_dir: &Path) -> Result<()> {
        match self {
            Commands::Translate(args) => args.run(models_dir).await,
            Commands::Detect(args) => args.run(models_dir).await,
            Commands::DetectLanguage(args) => args.run(models_dir).await,
            Commands::Languages(args) => args.run(),
        }
    }
}
