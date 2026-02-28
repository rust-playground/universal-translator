//! Continuous-batching scheduler for TranslateGemma.
//!
//! Maintains a fixed pool of [`N_SLOTS`] decode slots.  When a slot's
//! sequence emits EOS (or `<end_of_turn>`) it is retired immediately and the
//! freed slot is filled from the incoming work queue.
//!
//! Each slot prefills its prompt in one forward pass, then decodes token-by-
//! token using temperature sampling.  The scheduler runs in a dedicated
//! blocking thread (Metal/CUDA require non-async context).

use std::sync::Arc;
use std::sync::mpsc;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use tokio::sync::oneshot;

use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::scheduler::decoder::GemmaSlotDecoder;
use crate::scheduler::sampling::{
    apply_decoding_filters, apply_length_bias, force_eos_on_tail_repeat, sample_token,
};

/// Maximum output tokens per slot (prompt tokens + generated tokens combined).
pub const SLOT_CAPACITY: usize = 512;

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
    /// All output token IDs confirmed so far (does not include current_token
    /// until the step AFTER it is generated).
    output_ids: Vec<u32>,
    /// Predicted natural endpoint for EOS bias (decoupled from SLOT_CAPACITY).
    expected_len: usize,
    reply_tx: oneshot::Sender<Result<String, TranslatorError>>,
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Continuous-batching decode scheduler for TranslateGemma.
///
/// Spawn via [`ContinuousScheduler::run`] — it owns the model channel receiver
/// and never returns until the work channel is closed.
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

// ── Decode helper ─────────────────────────────────────────────────────────────

/// Run one decode step for a slot.
///
/// Returns the sampled next token.  Applies all logit filters (repetition
/// penalty, n-gram, length bias, tail-repeat) then samples with temperature.
fn decode_one_step(
    slot: &mut Slot,
    eos_token_id: u32,
    rng: &mut SmallRng,
) -> Result<u32, TranslatorError> {
    let mut logits = slot.decoder.decode_step(slot.current_token)?;

    apply_decoding_filters(&mut logits, &slot.output_ids);
    apply_length_bias(&mut logits, eos_token_id, slot.output_ids.len(), slot.expected_len);
    force_eos_on_tail_repeat(&mut logits, eos_token_id, &slot.output_ids);

    // Hard ceiling: force EOS when approaching capacity so we always terminate
    // with the EOS token rather than a mid-word cutoff.
    if slot.output_ids.len() + 1 >= SLOT_CAPACITY {
        for (i, v) in logits.iter_mut().enumerate() {
            if i != eos_token_id as usize {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    Ok(sample_token(&mut logits, rng))
}

// ── Scheduler loop ────────────────────────────────────────────────────────────

fn run_loop(model: &LoadedGemmaModel, work_rx: &mut mpsc::Receiver<InferRequest>) {
    let eos_id = model.eos_token_id();
    let mut slots: Vec<Option<Slot>> = (0..N_SLOTS).map(|_| None).collect();
    let mut rng = SmallRng::from_entropy();

    'scheduler: loop {
        // ── Layer 1: fill empty slots from the work queue (non-blocking) ─────
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

        // ── Count active slots ───────────────────────────────────────────────
        let active = slots.iter().filter(|s| s.is_some()).count();

        if active == 0 {
            // No active work — block until a new request arrives.
            match work_rx.recv() {
                Err(_) => break 'scheduler, // channel closed → clean shutdown
                Ok(InferRequest { text, reply_tx }) => {
                    for slot in slots.iter_mut() {
                        if slot.is_none() {
                            if let Ok(s) = prefill_slot(model, text, reply_tx, eos_id, &mut rng) {
                                *slot = Some(s);
                            }
                            break;
                        }
                    }
                }
            }
            continue;
        }

        tracing::debug!(active_slots = active, "decode pass");

        // ── Layer 3: one decode step per active slot ──────────────────────────
        for slot_opt in slots.iter_mut() {
            let tok_result = match slot_opt.as_mut() {
                None => continue,
                Some(s) => decode_one_step(s, eos_id, &mut rng),
            };

            match tok_result {
                Err(e) => {
                    let finished = slot_opt.take().unwrap();
                    let _ = finished.reply_tx.send(Err(e));
                }
                Ok(tok) => {
                    let at_capacity = {
                        let s = slot_opt.as_ref().unwrap();
                        s.output_ids.len() + 1 >= SLOT_CAPACITY
                    };
                    if tok == eos_id || at_capacity {
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

// ── Slot initialisation ───────────────────────────────────────────────────────

/// Prefill the prompt, sample the first output token, and construct a [`Slot`].
///
/// On any error, the error is sent through `reply_tx` before returning `Err(())`.
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
    // expected_len: predicted output length used for EOS bias, decoupled from
    // SLOT_CAPACITY.  Template overhead is ~30 tokens; we use a fraction of
    // the total prompt length as a rough translation-length estimate.
    let expected_len = ((prompt_len as f32 * 0.55 + 30.0) as usize).clamp(48, SLOT_CAPACITY);

    let mut decoder = model.new_slot_decoder();

    let mut logits = match decoder.prefill(&token_ids) {
        Ok(l) => l,
        Err(e) => {
            let _ = reply_tx.send(Err(e));
            return Err(());
        }
    };

    // Apply filters and sample the first output token.
    apply_decoding_filters(&mut logits, &[]);
    apply_length_bias(&mut logits, eos_token_id, 0, expected_len);

    let first_token = sample_token(&mut logits, rng);

    if first_token == eos_token_id {
        // Immediate EOS — return empty translation.
        let _ = reply_tx.send(Ok(String::new()));
        return Err(());
    }

    // The first output token is held in `current_token`.  It will be fed as
    // input on the first decode step, and pushed to `output_ids` only when
    // the NEXT token is confirmed not to be EOS (in the decode loop).
    Ok(Slot {
        decoder,
        current_token: first_token,
        output_ids: vec![first_token],
        expected_len,
        reply_tx,
    })
}
