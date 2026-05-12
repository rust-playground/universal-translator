use std::future::Future;
use std::time::Duration;

use futures::Stream;
use translator_core::types::{
    LanguageDetectionResult, TranslationRequest, TranslationResult, TranslationResultSet,
};
use translator_core::Language;

use crate::error::ClientError;
use crate::retry::RetryConfig;
use crate::stream::parse_sse_stream;

/// HTTP client for the Universal Translator API.
#[derive(Clone)]
pub struct TranslatorClient {
    http: reqwest::Client,
    base_url: String,
    retry: RetryConfig,
}

/// Builder for [`TranslatorClient`].
pub struct TranslatorClientBuilder {
    base_url: String,
    timeout: Duration,
    connect_timeout: Duration,
    retry: RetryConfig,
}

impl Default for TranslatorClientBuilder {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:3000".to_string(),
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(10),
            retry: RetryConfig::default(),
        }
    }
}

impl TranslatorClientBuilder {
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.retry.max_retries = n;
        self
    }

    pub fn base_backoff(mut self, d: Duration) -> Self {
        self.retry.base_backoff = d;
        self
    }

    pub fn max_backoff(mut self, d: Duration) -> Self {
        self.retry.max_backoff = d;
        self
    }

    pub fn build(self) -> Result<TranslatorClient, ClientError> {
        let http = reqwest::Client::builder()
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout)
            .pool_max_idle_per_host(4)
            .build()?;

        Ok(TranslatorClient {
            http,
            base_url: self.base_url.trim_end_matches('/').to_string(),
            retry: self.retry,
        })
    }
}

impl TranslatorClient {
    /// Create a builder with sensible defaults.
    pub fn builder() -> TranslatorClientBuilder {
        TranslatorClientBuilder::default()
    }

    /// POST `/translate` — batch translation with retry.
    pub async fn translate(
        &self,
        req: &TranslationRequest,
    ) -> Result<TranslationResultSet, ClientError> {
        self.with_retry(|| async {
            let resp = self
                .http
                .post(format!("{}/translate", self.base_url))
                .json(req)
                .send()
                .await?;
            handle_response(resp).await
        })
        .await
    }

    /// POST `/translate/stream` — SSE streaming translation.
    ///
    /// The initial HTTP connection is retried; once streaming begins, no retry is attempted
    /// (partial results may have already been yielded to the caller).
    pub async fn translate_stream(
        &self,
        req: &TranslationRequest,
    ) -> Result<impl Stream<Item = Result<TranslationResult, ClientError>>, ClientError> {
        let resp = self
            .with_retry(|| async {
                let resp = self
                    .http
                    .post(format!("{}/translate/stream", self.base_url))
                    .json(req)
                    .send()
                    .await?;
                check_status(resp).await
            })
            .await?;

        Ok(parse_sse_stream(resp))
    }

    /// GET `/languages` — list translate-supported languages.
    ///
    /// Returns typed `Language` values. Use `.code()` and `.full_name()` on
    /// each entry; no need for a separate name field on the wire.
    pub async fn languages(&self) -> Result<Vec<Language>, ClientError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            languages: Vec<Language>,
        }

        self.with_retry(|| async {
            let resp = self
                .http
                .get(format!("{}/languages?for=translate", self.base_url))
                .send()
                .await?;
            let resp: Resp = handle_response(resp).await?;
            Ok(resp.languages)
        })
        .await
    }

    /// GET `/languages?for=detect` — list detect-supported codes (broader
    /// than translate; includes lingua's full coverage plus script and
    /// heuristic refinements).
    ///
    /// Returns raw BCP 47 code strings since the detect universe includes
    /// codes outside the translate `Language` enum (e.g. `cy`, `eu`, `ka`).
    /// Parse with `code.parse::<Language>().ok()` if you need the enum form.
    pub async fn languages_detect(&self) -> Result<Vec<String>, ClientError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            languages: Vec<String>,
        }

        self.with_retry(|| async {
            let resp = self
                .http
                .get(format!("{}/languages?for=detect", self.base_url))
                .send()
                .await?;
            let resp: Resp = handle_response(resp).await?;
            Ok(resp.languages)
        })
        .await
    }

    /// GET `/health` — health check with retry.
    pub async fn health(&self) -> Result<(), ClientError> {
        self.with_retry(|| async {
            let resp = self
                .http
                .get(format!("{}/health", self.base_url))
                .send()
                .await?;
            check_status(resp).await?;
            Ok(())
        })
        .await
    }

    /// POST `/detect-language` — language detection with retry.
    pub async fn detect_language(
        &self,
        text: &str,
    ) -> Result<LanguageDetectionResult, ClientError> {
        #[derive(serde::Serialize)]
        struct Req<'a> {
            text: &'a str,
        }

        self.with_retry(|| async {
            let resp = self
                .http
                .post(format!("{}/detect-language", self.base_url))
                .json(&Req { text })
                .send()
                .await?;
            handle_response(resp).await
        })
        .await
    }

    /// Execute `f` with retry logic according to [`RetryConfig`].
    async fn with_retry<F, Fut, T>(&self, f: F) -> Result<T, ClientError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, ClientError>>,
    {
        let mut last_error: Option<ClientError> = None;

        for attempt in 0..=self.retry.max_retries {
            match f().await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if !e.is_retryable() || attempt == self.retry.max_retries {
                        if attempt > 0 {
                            return Err(ClientError::RetriesExhausted {
                                attempts: attempt + 1,
                                last_error: e.to_string(),
                            });
                        }
                        return Err(e);
                    }
                    let backoff = self.retry.backoff_duration(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max = self.retry.max_retries,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "retrying request"
                    );
                    last_error = Some(e);
                    tokio::time::sleep(backoff).await;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ClientError::RetriesExhausted {
            attempts: self.retry.max_retries + 1,
            last_error: "unknown".to_string(),
        }))
    }
}

/// Check HTTP status and return the response body deserialized as JSON.
async fn handle_response<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
) -> Result<T, ClientError> {
    let resp = check_status(resp).await?;
    resp.json::<T>()
        .await
        .map_err(|e| ClientError::Parse(format!("JSON decode: {e}")))
}

/// Check HTTP status code, returning the response on success or a `ClientError::Server` on failure.
async fn check_status(resp: reqwest::Response) -> Result<reqwest::Response, ClientError> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status().as_u16();
    let message = resp.text().await.unwrap_or_default();
    Err(ClientError::Server { status, message })
}
