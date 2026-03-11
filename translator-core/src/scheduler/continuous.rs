//! Continuous-batching scheduler for TranslateGemma backed by llama.cpp.
//!
//! Maintains a configurable pool of decode slots.  When a slot's sequence
//! emits an end-of-generation token it is retired immediately and the freed
//! slot is filled from the incoming work queue.
//!
//! **Batched prefill**: each scheduler loop iteration collects all immediately-
//! available requests and prefills them in a single `ctx.decode` call.
//!
//! **Batched decode**: every active slot participates in one `ctx.decode`
//! call per step — one call per round regardless of batch size.

use std::sync::mpsc;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::LlamaToken;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::scheduler::sampling::{
    apply_decoding_filters, apply_length_bias, force_eos_on_tail_repeat, sample_token,
};

struct Metrics {
    #[cfg(feature = "opentelemetry")]
    active_slots: opentelemetry::metrics::Gauge<u64>,
    #[cfg(feature = "opentelemetry")]
    slot_utilisation_pct: opentelemetry::metrics::Gauge<u64>,
    #[cfg(feature = "opentelemetry")]
    decode_forward_ms: opentelemetry::metrics::Histogram<f64>,
    #[cfg(feature = "opentelemetry")]
    prefill_ms: opentelemetry::metrics::Histogram<f64>,
    #[cfg(feature = "opentelemetry")]
    prompt_tokens: opentelemetry::metrics::Histogram<u64>,
    #[cfg(feature = "opentelemetry")]
    slots_completed: opentelemetry::metrics::Counter<u64>,
    #[cfg(feature = "opentelemetry")]
    tokens_generated: opentelemetry::metrics::Counter<u64>,
}

impl Metrics {
    fn new() -> Self {
        #[cfg(feature = "opentelemetry")]
        let meter = opentelemetry::global::meter("translator");
        Self {
            #[cfg(feature = "opentelemetry")]
            active_slots: meter.u64_gauge("translator.scheduler.active_slots").build(),
            #[cfg(feature = "opentelemetry")]
            slot_utilisation_pct: meter
                .u64_gauge("translator.scheduler.slot_utilisation_pct")
                .with_description("Active slot utilisation as integer percentage (0-100)")
                .build(),
            #[cfg(feature = "opentelemetry")]
            decode_forward_ms: meter
                .f64_histogram("translator.scheduler.decode_forward_ms")
                .with_boundaries(vec![
                    1., 5., 10., 25., 50., 100., 250., 500., 1000., 2500., 5000.,
                ])
                .build(),
            #[cfg(feature = "opentelemetry")]
            prefill_ms: meter
                .f64_histogram("translator.scheduler.prefill_ms")
                .with_boundaries(vec![
                    50., 100., 200., 500., 1000., 2000., 5000., 10000., 30000.,
                ])
                .build(),
            #[cfg(feature = "opentelemetry")]
            prompt_tokens: meter
                .u64_histogram("translator.scheduler.prompt_tokens")
                .with_boundaries(vec![10., 20., 50., 100., 200., 400., 600., 1024., 2048.])
                .build(),
            #[cfg(feature = "opentelemetry")]
            slots_completed: meter
                .u64_counter("translator.scheduler.slots_completed")
                .build(),
            #[cfg(feature = "opentelemetry")]
            tokens_generated: meter
                .u64_counter("translator.scheduler.tokens_generated")
                .build(),
        }
    }
}

/// Maximum output tokens per slot (prompt tokens + generated tokens combined).
pub const SLOT_CAPACITY: usize = 4096;

/// Token headroom reserved for output when checking prompt length.
const INPUT_HEADROOM: usize = 64;

// ── Public request type ───────────────────────────────────────────────────────

/// A single translation request dispatched to the continuous scheduler.
///
/// `text` is the full Gemma instruct-format prompt string.
/// Tokenization is performed on the scheduler thread.
pub struct InferRequest {
    pub text: String,
    /// Expected number of output tokens, used to calibrate EOS bias.
    pub expected_output_len: usize,
    /// Position of this request in the caller's work list — echoed back in the reply.
    pub index: usize,
    pub reply_tx: mpsc::Sender<(usize, Result<String, TranslatorError>)>,
}

// ── Internal types ────────────────────────────────────────────────────────────

/// Tokenized request waiting to be prefilled.
struct PendingPrefill {
    token_ids: Vec<u32>,
    expected_len: usize,
    index: usize,
    reply_tx: mpsc::Sender<(usize, Result<String, TranslatorError>)>,
}

struct Slot {
    /// llama.cpp sequence ID (0..n_slots). Returned to the pool on retirement.
    seq_id: i32,
    /// Current token to feed as input on the next decode step.
    current_token: u32,
    /// All output token IDs confirmed so far.
    output_ids: Vec<u32>,
    /// Predicted natural endpoint for EOS bias (decoupled from SLOT_CAPACITY).
    expected_len: usize,
    /// Next position in the KV cache for this sequence.
    pos: i32,
    index: usize,
    reply_tx: mpsc::Sender<(usize, Result<String, TranslatorError>)>,
    /// When this slot was assigned (after prefill). Used to measure decode latency.
    assigned_at: std::time::Instant,
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Continuous-batching decode scheduler for TranslateGemma.
///
/// Call [`ContinuousScheduler::run`] on a dedicated OS thread — it drives the
/// decode loop until the work channel closes.
pub struct ContinuousScheduler {
    model: std::sync::Arc<LoadedGemmaModel>,
    work_rx: crossbeam_channel::Receiver<InferRequest>,
    n_slots: usize,
    kv_budget_per_slot: u32,
    prefill_delay_ms: u64,
    metrics: Metrics,
}

impl ContinuousScheduler {
    pub fn new(
        model: std::sync::Arc<LoadedGemmaModel>,
        work_rx: crossbeam_channel::Receiver<InferRequest>,
        n_slots: usize,
        kv_budget_per_slot: u32,
        prefill_delay_ms: u64,
    ) -> Self {
        Self {
            model,
            work_rx,
            n_slots,
            kv_budget_per_slot,
            prefill_delay_ms,
            metrics: Metrics::new(),
        }
    }

    /// Drive the scheduler to completion (blocking).
    ///
    /// Call from a dedicated `std::thread::spawn` thread.
    /// Returns when the work channel closes.
    pub fn run(self) {
        run_loop(
            &self.model,
            &self.work_rx,
            self.n_slots,
            self.kv_budget_per_slot,
            self.prefill_delay_ms,
            &self.metrics,
        );
    }
}

// ── Scheduler loop ────────────────────────────────────────────────────────────

fn run_loop(
    model: &LoadedGemmaModel,
    work_rx: &crossbeam_channel::Receiver<InferRequest>,
    n_slots: usize,
    kv_budget_per_slot: u32,
    prefill_delay_ms: u64,
    metrics: &Metrics,
) {
    tracing::info!(n_slots, kv_budget_per_slot, prefill_delay_ms, "ContinuousScheduler started");

    let n_ctx = n_slots as u32 * kv_budget_per_slot;
    tracing::info!(n_ctx, kv_budget_per_slot, "creating llama context");
    let max_prompt_tokens = (kv_budget_per_slot as usize).saturating_sub(INPUT_HEADROOM);
    let prefill_accumulation_delay = std::time::Duration::from_millis(prefill_delay_ms);

    let mut ctx = match model.create_context(n_ctx, n_slots as u32) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to create llama context: {e}");
            return;
        }
    };

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);

    let eos_id = model.eos_token_id();
    let mut slots: Vec<Option<Slot>> = (0..n_slots).map(|_| None).collect();
    let mut rng = SmallRng::from_entropy();

    // Sequence ID pool: free IDs available for assignment.
    // Stored as a stack (LIFO) — order doesn't matter.
    let mut free_seq_ids: Vec<i32> = (0..n_slots as i32).rev().collect();

    // Items that couldn't fit in the last prefill batch.
    let mut carry_over: Vec<PendingPrefill> = Vec::new();

    'scheduler: loop {
        // ── Fill empty slots via batched prefill ──────────────────────────
        let n_empty = slots.iter().filter(|s| s.is_none()).count();
        if n_empty > 0 {
            let from_carry = carry_over.len().min(n_empty);
            let mut pending: Vec<PendingPrefill> = carry_over.drain(..from_carry).collect();

            let remaining_capacity = n_empty - pending.len();
            if remaining_capacity > 0 {
                pending.extend(collect_pending(model, work_rx, remaining_capacity, max_prompt_tokens, metrics));
            }

            if !pending.is_empty() {
                batch_prefill_and_assign(
                    model,
                    &mut ctx,
                    &mut batch,
                    &mut pending,
                    &mut slots,
                    &mut free_seq_ids,
                    eos_id,
                    &mut rng,
                    metrics,
                );
            }
        }

        // ── Collect active slot indices ──────────────────────────────────
        let active_indices: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();

        let n_active = active_indices.len();

        #[cfg(feature = "opentelemetry")]
        {
            metrics.active_slots.record(n_active as u64, &[]);
            let utilisation = (n_active as f64 / n_slots as f64 * 100.0) as u64;
            metrics.slot_utilisation_pct.record(utilisation, &[]);
        }

        if active_indices.is_empty() {
            // No active work — block until a request arrives (or channel closes).
            match work_rx.recv() {
                Err(_) => break 'scheduler, // all senders dropped → clean shutdown
                Ok(req) => {
                    let mut pending = Vec::new();
                    if let Some(pp) = tokenize_into_pending(model, req, max_prompt_tokens) {
                        #[cfg(feature = "opentelemetry")]
                        metrics
                            .prompt_tokens
                            .record(pp.token_ids.len() as u64, &[]);
                        pending.push(pp);
                    }
                    // Accumulation window: collect additional items arriving within
                    // PREFILL_ACCUMULATION_DELAY so concurrent requests aren't split
                    // across separate prefill batches.
                    let t_accum = std::time::Instant::now();
                    let deadline = t_accum + prefill_accumulation_delay;
                    loop {
                        let remaining =
                            deadline.saturating_duration_since(std::time::Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match work_rx.recv_timeout(remaining) {
                            Ok(r) => {
                                if let Some(pp) = tokenize_into_pending(model, r, max_prompt_tokens) {
                                    #[cfg(feature = "opentelemetry")]
                                    metrics
                                        .prompt_tokens
                                        .record(pp.token_ids.len() as u64, &[]);
                                    pending.push(pp);
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    tracing::debug!(
                        items = pending.len(),
                        waited_ms = t_accum.elapsed().as_millis(),
                        "idle-path accumulation closed"
                    );
                    if pending.len() > slots.len() {
                        carry_over = pending.drain(slots.len()..).collect();
                        tracing::debug!(
                            deferred = carry_over.len(),
                            "idle-path overflow → carry_over"
                        );
                    }
                    if !pending.is_empty() {
                        batch_prefill_and_assign(
                            model,
                            &mut ctx,
                            &mut batch,
                            &mut pending,
                            &mut slots,
                            &mut free_seq_ids,
                            eos_id,
                            &mut rng,
                            metrics,
                        );
                    }
                }
            }
            continue;
        }

        // ── Batched decode step ──────────────────────────────────────────
        batch.clear();
        for &slot_idx in &active_indices {
            let slot = slots[slot_idx].as_ref().unwrap();
            batch
                .add(
                    LlamaToken(slot.current_token as i32),
                    slot.pos,
                    &[slot.seq_id],
                    true,
                )
                .expect("batch capacity exceeded in decode step");
        }

        #[cfg(feature = "opentelemetry")]
        let _t_fw = std::time::Instant::now();

        if let Err(e) = ctx.decode(&mut batch) {
            #[cfg(feature = "opentelemetry")]
            metrics
                .decode_forward_ms
                .record(_t_fw.elapsed().as_micros() as f64 / 1000.0, &[]);

            let msg = format!("decode: {e}");
            for &si in &active_indices {
                let finished = slots[si].take().unwrap();
                free_seq_ids.push(finished.seq_id);
                let _ = ctx.clear_kv_cache_seq(Some(finished.seq_id as u32), None, None);
                let _ = finished
                    .reply_tx
                    .send((finished.index, Err(TranslatorError::Model(msg.clone()))));
            }
            continue;
        }

        #[cfg(feature = "opentelemetry")]
        metrics
            .decode_forward_ms
            .record(_t_fw.elapsed().as_micros() as f64 / 1000.0, &[]);

        // Per-slot logit extraction and sampling on CPU.
        let mut tok_ids: Vec<u32> = Vec::with_capacity(n_active);
        for (bi, &slot_idx) in active_indices.iter().enumerate() {
            let slot = slots[slot_idx].as_ref().unwrap();
            let mut logits = ctx.get_logits_ith(bi as i32).to_vec();

            apply_decoding_filters(&mut logits, &slot.output_ids);
            apply_length_bias(&mut logits, eos_id, slot.output_ids.len(), slot.expected_len);
            force_eos_on_tail_repeat(&mut logits, eos_id, &slot.output_ids);

            // Hard ceiling at SLOT_CAPACITY.
            if slot.output_ids.len() + 1 >= SLOT_CAPACITY {
                for (j, v) in logits.iter_mut().enumerate() {
                    if j != eos_id as usize {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }

            tok_ids.push(sample_token(&mut logits, &mut rng));
        }

        // ── Retire slots that emitted EOG; update the rest ───────────────
        for (i, tok) in tok_ids.into_iter().enumerate() {
            let slot_idx = active_indices[i];
            if model.is_eog_token(tok) {
                let finished = slots[slot_idx].take().unwrap();
                let cause = if finished.output_ids.len() + 1 >= SLOT_CAPACITY {
                    "capacity"
                } else {
                    "eos"
                };
                let slot_ms = finished.assigned_at.elapsed().as_millis();
                tracing::debug!(tokens = finished.output_ids.len(), cause, slot_ms, "slot retired");
                #[cfg(feature = "opentelemetry")]
                {
                    use opentelemetry::KeyValue;
                    metrics
                        .slots_completed
                        .add(1, &[KeyValue::new("cause", cause)]);
                    metrics
                        .tokens_generated
                        .add(finished.output_ids.len() as u64, &[]);
                }
                free_seq_ids.push(finished.seq_id);
                let _ = ctx.clear_kv_cache_seq(Some(finished.seq_id as u32), None, None);
                let text = model.decode_output_ids(&finished.output_ids);
                let _ = finished.reply_tx.send((finished.index, text));
            } else {
                let slot = slots[slot_idx].as_mut().unwrap();
                slot.output_ids.push(tok);
                slot.current_token = tok;
                slot.pos += 1;
            }
        }

        // ── Overlap: fill freed slots immediately ────────────────────────
        let n_empty_after = slots.iter().filter(|s| s.is_none()).count();
        if n_empty_after > 0 {
            let from_carry = carry_over.len().min(n_empty_after);
            let mut pending: Vec<PendingPrefill> = carry_over.drain(..from_carry).collect();
            let remaining = n_empty_after - pending.len();
            if remaining > 0 {
                pending.extend(collect_pending(model, work_rx, remaining, max_prompt_tokens, metrics));
            }
            if !pending.is_empty() {
                batch_prefill_and_assign(
                    model,
                    &mut ctx,
                    &mut batch,
                    &mut pending,
                    &mut slots,
                    &mut free_seq_ids,
                    eos_id,
                    &mut rng,
                    metrics,
                );
            }
        }
    }
}

// ── Helper: collect pending prefill items ────────────────────────────────────

/// Tokenize an [`InferRequest`] into a [`PendingPrefill`].
///
/// Returns `None` (and sends an error reply) if tokenization fails or the
/// prompt exceeds `max_prompt_tokens`.
fn tokenize_into_pending(
    model: &LoadedGemmaModel,
    req: InferRequest,
    max_prompt_tokens: usize,
) -> Option<PendingPrefill> {
    match model.tokenize(&req.text) {
        Ok(token_ids) => {
            if token_ids.len() > max_prompt_tokens {
                let _ = req.reply_tx.send((
                    req.index,
                    Err(TranslatorError::InputTooLong(format!(
                        "prompt is {} tokens, max is {max_prompt_tokens}",
                        token_ids.len()
                    ))),
                ));
                return None;
            }
            Some(PendingPrefill {
                token_ids,
                expected_len: req.expected_output_len,
                index: req.index,
                reply_tx: req.reply_tx,
            })
        }
        Err(e) => {
            let _ = req.reply_tx.send((req.index, Err(e)));
            None
        }
    }
}

/// Non-blocking drain of up to `limit` items from the work queue.
#[cfg_attr(not(feature = "opentelemetry"), allow(unused_variables))]
fn collect_pending(
    model: &LoadedGemmaModel,
    work_rx: &crossbeam_channel::Receiver<InferRequest>,
    limit: usize,
    max_prompt_tokens: usize,
    metrics: &Metrics,
) -> Vec<PendingPrefill> {
    let mut pending = Vec::new();
    for _ in 0..limit {
        match work_rx.try_recv() {
            Ok(req) => {
                if let Some(pp) = tokenize_into_pending(model, req, max_prompt_tokens) {
                    #[cfg(feature = "opentelemetry")]
                    metrics
                        .prompt_tokens
                        .record(pp.token_ids.len() as u64, &[]);
                    pending.push(pp);
                }
            }
            Err(_) => break,
        }
    }
    pending
}

// ── Batched prefill ───────────────────────────────────────────────────────────

/// Prefill all `pending` requests in one `ctx.decode` call, sample first tokens,
/// and assign the resulting [`Slot`]s into empty entries of `slots`.
#[cfg_attr(not(feature = "opentelemetry"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
fn batch_prefill_and_assign(
    model: &LoadedGemmaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    pending: &mut Vec<PendingPrefill>,
    slots: &mut [Option<Slot>],
    free_seq_ids: &mut Vec<i32>,
    eos_id: u32,
    rng: &mut SmallRng,
    metrics: &Metrics,
) {
    let empty_slot_count = slots.iter().filter(|s| s.is_none()).count();
    tracing::debug!(
        batch_size = pending.len(),
        empty_slots = empty_slot_count,
        "prefill batch"
    );

    batch.clear();

    // Intermediate struct to hold prefill state after batch building.
    struct PrefillEntry {
        seq_id: i32,
        n_prompt_tokens: usize,
        logits_batch_idx: i32,
        expected_len: usize,
        index: usize,
        reply_tx: mpsc::Sender<(usize, Result<String, TranslatorError>)>,
    }
    let mut entries: Vec<PrefillEntry> = Vec::with_capacity(pending.len());
    let mut batch_token_count: i32 = 0;

    for p in pending.drain(..) {
        let seq_id = match free_seq_ids.pop() {
            Some(id) => id,
            None => {
                // Should not happen (caller limits pending to empty slot count),
                // but handle gracefully.
                let _ = p
                    .reply_tx
                    .send((p.index, Err(TranslatorError::Model("no free seq_id".into()))));
                continue;
            }
        };

        let n_tokens = p.token_ids.len();
        for (ti, &tok) in p.token_ids.iter().enumerate() {
            batch
                .add(
                    LlamaToken(tok as i32),
                    ti as i32, // position within this sequence
                    &[seq_id],
                    ti == n_tokens - 1, // only request logits for last token
                )
                .expect("prefill batch capacity exceeded — increase KV_BUDGET_PER_SLOT");
            batch_token_count += 1;
        }

        entries.push(PrefillEntry {
            seq_id,
            n_prompt_tokens: n_tokens,
            logits_batch_idx: batch_token_count - 1,
            expected_len: p.expected_len,
            index: p.index,
            reply_tx: p.reply_tx,
        });
    }

    if entries.is_empty() {
        return;
    }

    #[cfg(feature = "opentelemetry")]
    let _pf = std::time::Instant::now();

    if let Err(e) = ctx.decode(batch) {
        let msg = format!("prefill decode: {e}");
        for entry in entries {
            free_seq_ids.push(entry.seq_id);
            let _ = ctx.clear_kv_cache_seq(Some(entry.seq_id as u32), None, None);
            let _ = entry
                .reply_tx
                .send((entry.index, Err(TranslatorError::Model(msg.clone()))));
        }
        return;
    }

    #[cfg(feature = "opentelemetry")]
    metrics
        .prefill_ms
        .record(_pf.elapsed().as_millis() as f64, &[]);

    // Sample first token for each prefilled sequence.
    for entry in entries {
        let mut logits = ctx.get_logits_ith(entry.logits_batch_idx).to_vec();

        apply_decoding_filters(&mut logits, &[]);
        apply_length_bias(&mut logits, eos_id, 0, entry.expected_len);
        let first_token = sample_token(&mut logits, rng);

        if model.is_eog_token(first_token) {
            free_seq_ids.push(entry.seq_id);
            let _ = ctx.clear_kv_cache_seq(Some(entry.seq_id as u32), None, None);
            let _ = entry.reply_tx.send((entry.index, Ok(String::new())));
            continue;
        }

        // Assign to the first empty slot.
        let empty_slot = slots.iter_mut().find(|s| s.is_none());
        match empty_slot {
            Some(slot) => {
                *slot = Some(Slot {
                    seq_id: entry.seq_id,
                    current_token: first_token,
                    output_ids: vec![first_token],
                    expected_len: entry.expected_len,
                    pos: entry.n_prompt_tokens as i32, // next decode position
                    index: entry.index,
                    reply_tx: entry.reply_tx,
                    assigned_at: std::time::Instant::now(),
                });
            }
            None => {
                // No empty slot — shouldn't happen, but handle gracefully.
                free_seq_ids.push(entry.seq_id);
                let _ = ctx.clear_kv_cache_seq(Some(entry.seq_id as u32), None, None);
                let _ = entry.reply_tx.send((
                    entry.index,
                    Err(TranslatorError::Model("no empty slot available".into())),
                ));
            }
        }
    }
}
