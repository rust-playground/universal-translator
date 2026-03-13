use translator_core::engine::TranslationEngine;

/// Axum shared state — cheap to clone (all internals are Arc-wrapped).
#[derive(Clone)]
pub struct AppState {
    pub engine: TranslationEngine,
    pub max_texts_per_request: usize,
    pub max_work_items_per_request: usize,
    #[cfg(feature = "opentelemetry")]
    pub error_ctr: opentelemetry::metrics::Counter<u64>,
}
