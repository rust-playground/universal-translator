//! Continuous-batching scheduler — Gemma phase.
//!
//! [`decoder`] contains [`decoder::GemmaSlotDecoder`], a per-slot wrapper
//! around a Gemma model clone with its own growing KV cache.
//!
//! [`continuous`] contains [`continuous::ContinuousScheduler`], the slot-pool
//! scheduler that drives concurrent decode across all active slots.

pub mod continuous;
pub mod decoder;
pub mod sampling;

pub use continuous::{ContinuousScheduler, InferRequest, SLOT_CAPACITY};
pub use decoder::GemmaSlotDecoder;
pub use sampling::{
    apply_decoding_filters, apply_length_bias, force_eos_on_tail_repeat, sample_token,
    EOS_LOGIT_BIAS, LENGTH_PENALTY_START, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY,
    TAIL_REPEAT_CHECK_LEN, TAIL_REPEAT_NGRAM, TEMPERATURE, TOP_K, TOP_P,
};
