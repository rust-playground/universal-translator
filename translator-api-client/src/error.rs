use thiserror::Error;

/// Errors from the translator API client.
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Server error ({status}): {message}")]
    Server { status: u16, message: String },

    #[error("Failed to parse response: {0}")]
    Parse(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Retries exhausted after {attempts} attempts: {last_error}")]
    RetriesExhausted { attempts: u32, last_error: String },
}

impl ClientError {
    /// Whether this error is worth retrying.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Request(e) => e.is_connect() || e.is_timeout() || e.is_request(),
            Self::Server { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 504),
            _ => false,
        }
    }
}
