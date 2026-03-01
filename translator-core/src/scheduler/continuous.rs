//! Continuous-batching scheduler for TranslateGemma.
//!
//! Maintains a fixed pool of [`N_SLOTS`] decode slots.  When a slot's
//! sequence emits EOS (or `<end_of_turn>`) it is retired immediately and the
//! freed slot is filled from the incoming work queue.
//!
//! Each slot prefills its prompt in one forward pass, then participates in
//! batched decode via [`LoadedGemmaModel::forward_batched`] — one call per
//! round processes all active slots simultaneously instead of serially.

use std::sync::Arc;
use std::sync::mpsc;

use candle_core::{DType, Tensor};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tokio::sync::oneshot;

use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::model_batched::SlotKvCache;
use crate::scheduler::decoder::GemmaSlotDecoder;
use crate::scheduler::sampling::{
    apply_decoding_filters, apply_length_bias, force_eos_on_tail_repeat, sample_token,
};

fn cerr(e: candle_core::Error) -> TranslatorError {
    TranslatorError::Model(e.to_string())
}

/// Maximum output tokens per slot (prompt tokens + generated tokens combined).
pub const SLOT_CAPACITY: usize = 4096;

/// Number of parallel decode slots in the pool.
pub const N_SLOTS: usize = 24;

// ── Public request type ───────────────────────────────────────────────────────

/// A single translation request dispatched to the continuous scheduler.
///
/// `text` must already be formatted as a complete Gemma instruct prompt
/// (e.g. the output of `translate_gemma_prompt()`).
pub struct InferRequest {
    pub text: String,
    pub reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
}

// ── Internal slot ─────────────────────────────────────────────────────────────

struct Slot {
    decoder: GemmaSlotDecoder,
    /// Current token to feed as input on the next decode step.
    current_token: u32,
    /// All output token IDs confirmed so far.
    output_ids: Vec<u32>,
    /// Predicted natural endpoint for EOS bias (decoupled from SLOT_CAPACITY).
    expected_len: usize,
    reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Continuous-batching decode scheduler for TranslateGemma.
///
/// Spawn via [`ContinuousScheduler::run`] — it drives the decode loop until the
/// work channel closes.
pub struct ContinuousScheduler {
    model: Arc<LoadedGemmaModel>,
    work_rx: mpsc::Receiver<InferRequest>,
}

impl ContinuousScheduler {
    pub fn new(model: Arc<LoadedGemmaModel>, work_rx: mpsc::Receiver<InferRequest>) -> Self {
        Self { model, work_rx }
    }

    /// Drive the scheduler to completion.
    ///
    /// Spawns onto a blocking thread and returns when the work channel closes.
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

// ── Scheduler loop ────────────────────────────────────────────────────────────

fn run_loop(model: &LoadedGemmaModel, work_rx: &mut mpsc::Receiver<InferRequest>) {
    let eos_id = model.eos_token_id();
    let mut slots: Vec<Option<Slot>> = (0..N_SLOTS).map(|_| None).collect();
    let mut rng = SmallRng::from_entropy();

    'scheduler: loop {
        // ── Fill empty slots from the work queue (non-blocking) ───────────
        for slot in slots.iter_mut() {
            if slot.is_some() {
                continue;
            }
            match work_rx.try_recv() {
                Ok(InferRequest { text, reply_tx }) => {
                    if let Ok(s) = prefill_slot(model, text, reply_tx, eos_id, &mut rng) {
                        *slot = Some(s);
                    }
                }
                Err(_) => break, // no more queued work
            }
        }

        // ── Collect active slot indices ───────────────────────────────────
        let active_indices: Vec<usize> =
            slots.iter().enumerate().filter_map(|(i, s)| s.as_ref().map(|_| i)).collect();

        if active_indices.is_empty() {
            // No active work — block until a new request arrives.
            match work_rx.recv() {
                Err(_) => break 'scheduler, // channel closed → clean shutdown
                Ok(InferRequest { text, reply_tx }) => {
                    for slot in slots.iter_mut() {
                        if slot.is_none() {
                            if let Ok(s) = prefill_slot(model, text, reply_tx, eos_id, &mut rng)
                            {
                                *slot = Some(s);
                            }
                            break;
                        }
                    }
                }
            }
            continue;
        }

        let n = active_indices.len();
        tracing::debug!(active_slots = n, "batched decode pass");

        // ── Build [N, 1] token tensor ─────────────────────────────────────
        let tokens_vec: Vec<u32> = active_indices
            .iter()
            .map(|&i| slots[i].as_ref().unwrap().current_token)
            .collect();

        let tokens_t = match Tensor::from_slice(&tokens_vec, (n, 1), model.device())
            .map_err(cerr)
        {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                for &si in &active_indices {
                    let finished = slots[si].take().unwrap();
                    let _ = finished
                        .reply_tx
                        .send(Err(TranslatorError::Model(msg.clone())));
                }
                continue;
            }
        };

        // ── Temporarily move KV caches out so we can pass &mut [SlotKvCache]
        //    while still holding the rest of each Slot by index ─────────────
        let mut batch_kv: Vec<SlotKvCache> = active_indices
            .iter()
            .map(|&i| {
                std::mem::replace(
                    &mut slots[i].as_mut().unwrap().decoder.kv_cache,
                    SlotKvCache { layers: Vec::new(), seq_len: 0 },
                )
            })
            .collect();

        let all_logits_t = match model.forward_batched(&tokens_t, &mut batch_kv) {
            Ok(t) => t,
            Err(e) => {
                // Restore KV caches before retiring slots
                for (bi, &si) in active_indices.iter().enumerate() {
                    slots[si].as_mut().unwrap().decoder.kv_cache =
                        std::mem::replace(&mut batch_kv[bi], SlotKvCache { layers: Vec::new(), seq_len: 0 });
                }
                let msg = e.to_string();
                for &si in &active_indices {
                    let finished = slots[si].take().unwrap();
                    let _ = finished
                        .reply_tx
                        .send(Err(TranslatorError::Model(msg.clone())));
                }
                continue;
            }
        };

        // ── Restore KV caches ─────────────────────────────────────────────
        for (bi, &si) in active_indices.iter().enumerate() {
            slots[si].as_mut().unwrap().decoder.kv_cache =
                std::mem::replace(&mut batch_kv[bi], SlotKvCache { layers: Vec::new(), seq_len: 0 });
        }

        // ── Single GPU→CPU transfer: all slot logits at once ──────────────
        let all_logits_cpu: Vec<Vec<f32>> = match all_logits_t
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2::<f32>())
            .map_err(cerr)
        {
            Ok(v) => v,
            Err(e) => {
                let msg = e.to_string();
                for &si in &active_indices {
                    if let Some(finished) = slots[si].take() {
                        let _ = finished.reply_tx.send(Err(TranslatorError::Model(msg.clone())));
                    }
                }
                continue;
            }
        };

        // ── Per-slot: apply filters, sample, retire if EOS ────────────────
        for (batch_idx, &slot_idx) in active_indices.iter().enumerate() {
            let mut logits = all_logits_cpu[batch_idx].clone();

            let slot = slots[slot_idx].as_mut().unwrap();

            apply_decoding_filters(&mut logits, &slot.output_ids);
            apply_length_bias(&mut logits, eos_id, slot.output_ids.len(), slot.expected_len);
            force_eos_on_tail_repeat(&mut logits, eos_id, &slot.output_ids);

            // Hard ceiling: force EOS when approaching capacity
            if slot.output_ids.len() + 1 >= SLOT_CAPACITY {
                for (i, v) in logits.iter_mut().enumerate() {
                    if i != eos_id as usize {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }

            let tok = sample_token(&mut logits, &mut rng);
            let at_capacity = slot.output_ids.len() + 1 >= SLOT_CAPACITY;

            if tok == eos_id || at_capacity {
                let finished = slots[slot_idx].take().unwrap();
                let text = model.decode_output_ids(&finished.output_ids);
                let _ = finished.reply_tx.send(text);
            } else {
                slot.output_ids.push(tok);
                slot.current_token = tok;
            }
        }
    }
}

// ── Slot initialisation ───────────────────────────────────────────────────────

/// Prefill the prompt, sample the first output token, and construct a [`Slot`].
///
/// On any error the error is sent through `reply_tx` before returning `Err(())`.
fn prefill_slot(
    model: &LoadedGemmaModel,
    text: String,
    reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
    eos_token_id: u32,
    rng: &mut SmallRng,
) -> Result<Slot, ()> {
    let token_ids = match model.tokenize(&text) {
        Ok(ids) => ids,
        Err(e) => {
            let _ = reply_tx.send(Err(e));
            return Err(());
        }
    };

    let prompt_len = token_ids.len();
    let expected_len = ((prompt_len as f32 * 0.55 + 30.0) as usize).clamp(48, SLOT_CAPACITY);

    let mut decoder = model.new_slot_decoder();

    let mut logits = match decoder.prefill(model, &token_ids) {
        Ok(l) => l,
        Err(e) => {
            let _ = reply_tx.send(Err(e));
            return Err(());
        }
    };

    apply_decoding_filters(&mut logits, &[]);
    apply_length_bias(&mut logits, eos_token_id, 0, expected_len);

    let first_token = sample_token(&mut logits, rng);

    if first_token == eos_token_id {
        let _ = reply_tx.send(Ok(String::new()));
        return Err(());
    }

    Ok(Slot {
        decoder,
        current_token: first_token,
        output_ids: vec![first_token],
        expected_len,
        reply_tx,
    })
}
