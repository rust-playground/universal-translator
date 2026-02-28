//! Continuous-batching scheduler — Phase 2.
//!
//! [`decoder`] contains [`decoder::CustomT5Decoder`], a T5 decoder with
//! externalized KV state that allows per-slot cache management.
//!
//! [`continuous`] contains [`continuous::ContinuousScheduler`], the Phase 2d-B
//! slot-pool scheduler that replaces the epoch-aligned batch worker.

pub mod continuous;
pub mod decoder;
pub mod sampling;

pub use continuous::{ContinuousScheduler, InferRequest};
pub use decoder::{CustomT5Decoder, DecoderKvCache};
pub use sampling::{
    apply_decoding_filters, apply_length_bias, force_eos_on_tail_repeat,
    EOS_LOGIT_BIAS, LENGTH_PENALTY_START, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY,
    TAIL_REPEAT_CHECK_LEN, TAIL_REPEAT_NGRAM,
};
