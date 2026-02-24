use thiserror::Error;

#[derive(Debug, Error)]
pub enum TranslatorError {
    #[error("Model not found for language pair: {0}")]
    ModelNotFound(String),

    #[error("Language detection failed: {0}")]
    DetectionFailed(String),

    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Translation failed: {0}")]
    TranslationFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),
}
