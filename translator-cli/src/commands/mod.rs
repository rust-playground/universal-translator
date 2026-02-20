pub mod detect;
pub mod languages;
pub mod translate;

use anyhow::Result;
use clap::Subcommand;
use translator_core::engine::TranslationEngine;

#[derive(Subcommand)]
pub enum Commands {
    Translate(translate::TranslateArgs),
    Detect(detect::DetectArgs),
    Languages(languages::LanguagesArgs),
}

impl Commands {
    pub async fn run(self, engine: TranslationEngine) -> Result<()> {
        match self {
            Commands::Translate(args) => args.run(engine).await,
            Commands::Detect(args) => args.run(engine).await,
            Commands::Languages(args) => args.run(),
        }
    }
}
