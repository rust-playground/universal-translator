pub mod detect;
pub mod detect_language;
pub mod languages;
pub mod translate;

use anyhow::Result;
use clap::Subcommand;
use translator_core::EngineConfig;

#[derive(Subcommand)]
pub enum Commands {
    Translate(translate::TranslateArgs),
    Detect(detect::DetectArgs),
    DetectLanguage(detect_language::DetectLanguageArgs),
    Languages(languages::LanguagesArgs),
}

impl Commands {
    pub fn run(self, config: EngineConfig) -> Result<()> {
        match self {
            Commands::Translate(args) => args.run(config),
            Commands::Detect(args) => args.run(&config.models_dir),
            Commands::DetectLanguage(args) => args.run(&config.models_dir),
            Commands::Languages(args) => args.run(),
        }
    }
}
