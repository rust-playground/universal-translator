use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use translator_core::error::TranslatorError;

pub struct ApiError(pub TranslatorError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self.0 {
            TranslatorError::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.0.to_string()),
            TranslatorError::DetectionFailed(_) => {
                (StatusCode::UNPROCESSABLE_ENTITY, self.0.to_string())
            }
            TranslatorError::UnsupportedLanguage(_) => {
                (StatusCode::BAD_REQUEST, self.0.to_string())
            }
            TranslatorError::ServiceUnavailable(_) => {
                (StatusCode::TOO_MANY_REQUESTS, self.0.to_string())
            }
            TranslatorError::InputTooLong(_) => {
                (StatusCode::PAYLOAD_TOO_LARGE, self.0.to_string())
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };

        if status.is_server_error() {
            tracing::error!(error = %message, %status, "request failed");
        }

        (status, Json(json!({ "error": message }))).into_response()
    }
}

impl From<TranslatorError> for ApiError {
    fn from(e: TranslatorError) -> Self {
        ApiError(e)
    }
}
