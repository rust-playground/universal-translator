use translator_core::engine::TranslationEngine;

/// Axum shared state — cheap to clone (all internals are Arc-wrapped).
#[derive(Clone)]
pub struct AppState {
    pub engine: TranslationEngine,
}
