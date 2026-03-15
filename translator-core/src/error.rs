use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Per-language error within a translation result.
#[derive(Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "message")]
pub enum TranslationItemError {
    #[error("detection failed: {0}")]
    DetectionFailed(String),
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("translation failed: {0}")]
    TranslationFailed(String),
}

impl From<&TranslatorError> for TranslationItemError {
    fn from(e: &TranslatorError) -> Self {
        match e {
            TranslatorError::DetectionFailed(msg) => Self::DetectionFailed(msg.clone()),
            TranslatorError::UnsupportedLanguage(msg) => Self::UnsupportedLanguage(msg.clone()),
            _ => Self::TranslationFailed(e.to_string()),
        }
    }
}

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

    #[error("Service overloaded: {0}")]
    ServiceUnavailable(String),

    #[error("Input too long: {0}")]
    InputTooLong(String),
}
