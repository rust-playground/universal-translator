pub mod detect;
pub mod detect_language;
pub mod languages;
pub mod setup;
pub mod translate;

use std::path::Path;

use anyhow::Result;
use clap::Subcommand;
use translator_core::EngineConfig;

#[derive(Subcommand)]
pub enum Commands {
    Translate(translate::TranslateArgs),
    Detect(detect::DetectArgs),
    DetectLanguage(detect_language::DetectLanguageArgs),
    Languages(languages::LanguagesArgs),
    /// Download model weights from HuggingFace.
    Setup(setup::SetupArgs),
}

impl Commands {
    pub fn run(self, config: EngineConfig, default_models_dir: &Path) -> Result<()> {
        match self {
            Commands::Translate(args) => args.run(config),
            Commands::Detect(args) => args.run(default_models_dir),
            Commands::DetectLanguage(args) => args.run(default_models_dir),
            Commands::Languages(args) => args.run(),
            Commands::Setup(args) => args.run(default_models_dir),
        }
    }
}
