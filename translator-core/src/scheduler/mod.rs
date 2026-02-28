//! Continuous-batching scheduler — Phase 2.
//!
//! [`decoder`] contains [`decoder::CustomT5Decoder`], a T5 decoder with
//! externalized KV state that allows per-slot cache management.

pub mod decoder;
pub use decoder::{CustomT5Decoder, DecoderKvCache};
