//! Continuous-batching scheduler — Phase 2d-B.
//!
//! Maintains a fixed pool of [`N_SLOTS`] decode slots.  When a slot's
//! sequence emits EOS it is retired immediately and the freed slot is filled
//! from the incoming work queue — removing the "batch-complete barrier" of the
//! old epoch-aligned worker.
//!
//! Each slot is decoded independently (strategy i — serial per-slot), which
//! avoids the heterogeneous-KV-depth problem without requiring attention
//! masking.  Future work (2d-A) can batch compatible slots for higher GPU
//! utilisation.

use std::sync::Arc;
use std::sync::mpsc;

use candle_core::{D, Tensor};
use tokio::sync::oneshot;

use crate::error::TranslatorError;
use crate::model::LoadedModel;

/// Max output tokens buffered per slot.  Pre-allocates the KV cache at this
/// depth — keep conservatively sized to limit per-slot memory.
///
/// Per slot memory (B=1, 32 layers, 8 heads, d_kv=64, fp32):
///   32 × 2 × 1 × 8 × SLOT_CAPACITY × 64 × 4 bytes
///   = SLOT_CAPACITY × 131_072 bytes
/// At 256: ~32 MB per slot.
pub const SLOT_CAPACITY: usize = 256;

/// Number of parallel decode slots in the pool.
pub const N_SLOTS: usize = 32;

// ── Public request type ───────────────────────────────────────────────────────

/// A single translation request dispatched to the continuous scheduler.
///
/// `text` must already carry the MADLAD language token prefix (e.g. `"<2fr> …"`).
pub struct InferRequest {
    pub text: String,
    pub reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
}

// ── Internal slot ─────────────────────────────────────────────────────────────

struct Slot {
    cache: crate::scheduler::decoder::DecoderKvCache,
    /// Predicted natural translation endpoint — EOS bias reference.
    /// Decoupled from `cache.capacity` (KV memory ceiling) so the bias
    /// saturates at the right step rather than at the hard limit.
    expected_len: usize,
    current_token: u32,
    output_ids: Vec<u32>,
    reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Continuous-batching decode scheduler.
///
/// Spawn via [`ContinuousScheduler::run`] — it owns the model after the first
/// call and never returns until the work channel is closed.
pub struct ContinuousScheduler {
    model: Arc<LoadedModel>,
    work_rx: mpsc::Receiver<InferRequest>,
}

impl ContinuousScheduler {
    pub fn new(model: Arc<LoadedModel>, work_rx: mpsc::Receiver<InferRequest>) -> Self {
        Self { model, work_rx }
    }

    /// Drive the scheduler to completion.
    ///
    /// Spawns onto a blocking thread (Metal/CUDA GPU requires non-async context)
    /// and returns when the work channel closes.
    pub async fn run(self) {
        let model = self.model;
        let mut work_rx = self.work_rx;

        tokio::task::spawn_blocking(move || {
            run_loop(&model, &mut work_rx);
        })
        .await
        .ok();
    }
}

// ── Per-slot decode helper ────────────────────────────────────────────────────

/// Run one decode step for a slot.  Returns `Ok(tok)` on success; the caller
/// decides whether `tok` causes retirement (EOS or capacity check).
fn decode_one_step(
    slot: &mut Slot,
    decoder: &crate::scheduler::decoder::CustomT5Decoder,
    device: &candle_core::Device,
    eos_token_id: u32,
) -> Result<u32, TranslatorError> {
    let dec = Tensor::from_vec(vec![slot.current_token], (1usize, 1usize), device)
        .map_err(|e| TranslatorError::Model(e.to_string()))?;
    let logits_t = decoder.decode_step(&dec, &mut slot.cache)?; // [1, vocab_size]
    let vocab_size = logits_t
        .dim(1)
        .map_err(|e| TranslatorError::Model(e.to_string()))?;
    let mut logits: Vec<f32> = logits_t
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| TranslatorError::Model(e.to_string()))?;
    crate::scheduler::sampling::apply_decoding_filters(&mut logits, &slot.output_ids);
    crate::scheduler::sampling::apply_length_bias(
        &mut logits,
        eos_token_id,
        slot.output_ids.len(),
        slot.expected_len,
    );
    crate::scheduler::sampling::force_eos_on_tail_repeat(
        &mut logits,
        eos_token_id,
        &slot.output_ids,
    );
    // Force EOS at the capacity boundary so termination is always EOS-triggered,
    // never a hard mid-word cutoff.
    if slot.output_ids.len() + 1 >= slot.cache.capacity {
        for (i, v) in logits.iter_mut().enumerate() {
            if i != eos_token_id as usize {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(logits, (1usize, vocab_size), device)
        .map_err(|e| TranslatorError::Model(e.to_string()))?
        .argmax(D::Minus1)
        .and_then(|t| t.to_vec1::<u32>())
        .map(|v| v[0])
        .map_err(|e| TranslatorError::Model(e.to_string()))
}

// ── Scheduler loop (runs in spawn_blocking) ───────────────────────────────────

fn run_loop(
    model: &LoadedModel,
    work_rx: &mut mpsc::Receiver<InferRequest>,
) {
    let decoder = model.custom_decoder();
    let eos_id = model.eos_token_id();
    let start_id = model.decoder_start_token_id();

    let mut slots: Vec<Option<Slot>> = (0..N_SLOTS).map(|_| None).collect();

    'scheduler: loop {
        // ── Fill empty slots from the work queue (non-blocking) ──────────────
        for slot in slots.iter_mut() {
            if slot.is_some() {
                continue;
            }
            match work_rx.try_recv() {
                Ok(InferRequest { text, reply_tx }) => {
                    // Error is handled inside fill_slot (sends to reply_tx).
                    if let Ok(s) = fill_slot(model, decoder, text, reply_tx, start_id) {
                        *slot = Some(s);
                    }
                }
                Err(_) => break, // no queued work — stop scanning
            }
        }

        // ── Count active slots ───────────────────────────────────────────────
        let active = slots.iter().filter(|s| s.is_some()).count();

        if active == 0 {
            // No active work — block until a new request arrives.
            // std::sync::mpsc::recv() blocks the OS thread directly, no runtime involvement.
            match work_rx.recv() {
                Err(_) => break 'scheduler, // channel closed → clean shutdown
                Ok(InferRequest { text, reply_tx }) => {
                    // Fill the first empty slot.
                    for slot in slots.iter_mut() {
                        if slot.is_none() {
                            if let Ok(s) = fill_slot(model, decoder, text, reply_tx, start_id) {
                                *slot = Some(s);
                            }
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // ── Decode one step for each active slot (serial, strategy i) ────────
        for slot_opt in slots.iter_mut() {
            // decode_one_step borrows slot_opt.as_mut() transiently; the borrow
            // ends before the match arms below touch slot_opt again.
            let tok_result = match slot_opt.as_mut() {
                None => continue,
                Some(s) => decode_one_step(s, decoder, model.device(), eos_id),
            };

            match tok_result {
                Err(e) => {
                    let finished = slot_opt.take().unwrap();
                    let _ = finished.reply_tx.send(Err(e));
                }
                Ok(tok) => {
                    let at_capacity = {
                        let s = slot_opt.as_ref().unwrap();
                        // output_ids.len() + 1 because we'd push tok next.
                        s.output_ids.len() + 1 >= s.cache.capacity
                    };
                    if tok == eos_id || at_capacity {
                        // decode_one_step guarantees tok == eos_id when at_capacity;
                        // never push eos_id or a partial capacity-boundary token.
                        let finished = slot_opt.take().unwrap();
                        let text = model.decode_output_ids(&finished.output_ids);
                        let _ = finished.reply_tx.send(text);
                    } else {
                        let s = slot_opt.as_mut().unwrap();
                        s.output_ids.push(tok);
                        s.current_token = tok;
                    }
                }
            }
        }
    }
}

// ── Slot init ─────────────────────────────────────────────────────────────────

/// Try to encode `text` and prepare a decode slot.
///
/// Returns `Ok(Slot)` on success.  On any error, the error is sent through
/// `reply_tx` before returning `Err(())` — the caller can ignore the `Err`.
fn fill_slot(
    model: &LoadedModel,
    decoder: &crate::scheduler::decoder::CustomT5Decoder,
    text: String,
    reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
    decoder_start_token_id: u32,
) -> Result<Slot, ()> {
    let (encoder_output, seq_len) = match model.encode_only(&[text]) {
        Ok(v) => v,
        Err(e) => {
            let _ = reply_tx.send(Err(e));
            return Err(());
        }
    };

    // Adaptive capacity: bound by the expected output length but no more than
    // SLOT_CAPACITY to keep per-slot memory predictable.
    let adaptive = (seq_len as f32 * 1.40 + 10.0) as usize;
    let capacity = SLOT_CAPACITY.min(crate::model::MAX_NEW_TOKENS.min(adaptive));
    // expected_len: predicted natural translation endpoint used as the EOS bias
    // reference — decoupled from capacity (KV ceiling) so the bias peaks at the
    // right step rather than at the hard limit.
    let expected_len = (seq_len as f32 * 1.35 + 10.0) as usize;

    let mut cache = match decoder.new_kv_cache(1, capacity) {
        Ok(c) => c,
        Err(e) => {
            let _ = reply_tx.send(Err(e));
            return Err(());
        }
    };

    if let Err(e) = decoder.compute_cross_kv(&encoder_output, &mut cache) {
        let _ = reply_tx.send(Err(e));
        return Err(());
    }

    Ok(Slot {
        cache,
        expected_len,
        current_token: decoder_start_token_id,
        output_ids: Vec::new(),
        reply_tx,
    })
}
