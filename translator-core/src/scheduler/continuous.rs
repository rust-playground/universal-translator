//! Continuous-batching scheduler for TranslateGemma.
//!
//! Maintains a configurable pool of decode slots.  When a slot's sequence
//! emits EOS (or `<end_of_turn>`) it is retired immediately and the freed
//! slot is filled from the incoming work queue.
//!
//! **Batched prefill**: each scheduler loop iteration collects all immediately-
//! available requests and prefills them in a single `forward_prefill_batched`
//! call instead of N serial single-slot forward passes.
//!
//! **Batched decode**: every active slot participates in one `forward_batched`
//! call per step — one call per round regardless of batch size.

use std::sync::{Arc, mpsc};

use candle_core::{DType, Tensor};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::model_batched::SlotKvCache;
use crate::scheduler::decoder::GemmaSlotDecoder;
use crate::scheduler::sampling::{
    apply_decoding_filters, apply_length_bias, force_eos_on_tail_repeat, sample_token, TOP_K,
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
                .with_boundaries(vec![1., 5., 10., 25., 50., 100., 250., 500., 1000., 2500., 5000.])
                .build(),
            #[cfg(feature = "opentelemetry")]
            prefill_ms: meter
                .f64_histogram("translator.scheduler.prefill_ms")
                .with_boundaries(vec![50., 100., 200., 500., 1000., 2000., 5000., 10000., 30000.])
                .build(),
            #[cfg(feature = "opentelemetry")]
            prompt_tokens: meter
                .u64_histogram("translator.scheduler.prompt_tokens")
                .with_boundaries(vec![10., 20., 50., 100., 200., 400., 600., 1024., 2048.])
                .build(),
            #[cfg(feature = "opentelemetry")]
            slots_completed: meter.u64_counter("translator.scheduler.slots_completed").build(),
            #[cfg(feature = "opentelemetry")]
            tokens_generated: meter.u64_counter("translator.scheduler.tokens_generated").build(),
        }
    }
}

fn cerr(e: candle_core::Error) -> TranslatorError {
    TranslatorError::Model(e.to_string())
}

/// Maximum output tokens per slot (prompt tokens + generated tokens combined).
pub const SLOT_CAPACITY: usize = 4096;

// ── Public request type ───────────────────────────────────────────────────────

/// A single translation request dispatched to the continuous scheduler.
///
/// `text` must already be formatted as a complete Gemma instruct prompt
/// (e.g. the output of `translate_gemma_prompt()`).
pub struct InferRequest {
    pub text: String,
    /// Expected number of output tokens, used to calibrate EOS bias.
    /// Computed by the engine from the original text length (not the prompt length).
    pub expected_output_len: usize,
    pub reply_tx: mpsc::Sender<Result<String, TranslatorError>>,
}

// ── Internal types ────────────────────────────────────────────────────────────

/// Tokenized request waiting to be prefilled.
struct PendingPrefill {
    token_ids: Vec<u32>,
    expected_len: usize,
    reply_tx: mpsc::Sender<Result<String, TranslatorError>>,
}

struct Slot {
    decoder: GemmaSlotDecoder,
    /// Current token to feed as input on the next decode step.
    current_token: u32,
    /// All output token IDs confirmed so far.
    output_ids: Vec<u32>,
    /// Predicted natural endpoint for EOS bias (decoupled from SLOT_CAPACITY).
    expected_len: usize,
    reply_tx: mpsc::Sender<Result<String, TranslatorError>>,
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Continuous-batching decode scheduler for TranslateGemma.
///
/// Call [`ContinuousScheduler::run`] on a dedicated OS thread — it drives the
/// decode loop until the work channel closes.
pub struct ContinuousScheduler {
    model: Arc<LoadedGemmaModel>,
    work_rx: crossbeam_channel::Receiver<InferRequest>,
    n_slots: usize,
    metrics: Metrics,
}

impl ContinuousScheduler {
    pub fn new(
        model: Arc<LoadedGemmaModel>,
        work_rx: crossbeam_channel::Receiver<InferRequest>,
        n_slots: usize,
    ) -> Self {
        Self { model, work_rx, n_slots, metrics: Metrics::new() }
    }

    /// Drive the scheduler to completion (blocking).
    ///
    /// Call from a dedicated `std::thread::spawn` thread.
    /// Returns when the work channel closes.
    pub fn run(self) {
        run_loop(&self.model, &self.work_rx, self.n_slots, &self.metrics);
    }
}

// ── Scheduler loop ────────────────────────────────────────────────────────────

fn run_loop(
    model: &LoadedGemmaModel,
    work_rx: &crossbeam_channel::Receiver<InferRequest>,
    n_slots: usize,
    metrics: &Metrics,
) {
    tracing::info!(n_slots, "ContinuousScheduler started");

    let eos_id = model.eos_token_id();
    let mut slots: Vec<Option<Slot>> = (0..n_slots).map(|_| None).collect();
    let mut rng = SmallRng::from_entropy();

    // Pre-allocated arena buffers — reused every iteration to avoid hot-path allocs.
    let mut active_indices: Vec<usize> = Vec::with_capacity(n_slots);
    let mut tokens_vec: Vec<u32> = Vec::with_capacity(n_slots);
    // Scratch buffer for sample_token — avoids one Vec alloc per token generated.
    let mut sample_scratch: Vec<(u32, f32)> = Vec::with_capacity(TOP_K + 1);

    'scheduler: loop {
        // ── Fill empty slots via batched prefill ──────────────────────────
        let n_empty = slots.iter().filter(|s| s.is_none()).count();
        if n_empty > 0 {
            let mut pending = collect_pending(model, work_rx, n_empty, metrics);
            if !pending.is_empty() {
                batch_prefill_and_assign(
                    model, &mut pending, &mut slots, eos_id, &mut rng, metrics, &mut sample_scratch,
                );
            }
        }

        // ── Collect active slot indices (reuse pre-allocated vec) ─────────
        active_indices.clear();
        for (i, s) in slots.iter().enumerate() {
            if s.is_some() {
                active_indices.push(i);
            }
        }

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
                    tokenize_into_pending(model, req, &mut pending, metrics);
                    while let Ok(r) = work_rx.try_recv() {
                        tokenize_into_pending(model, r, &mut pending, metrics);
                    }
                    if !pending.is_empty() {
                        batch_prefill_and_assign(
                            model,
                            &mut pending,
                            &mut slots,
                            eos_id,
                            &mut rng,
                            metrics,
                            &mut sample_scratch,
                        );
                    }
                }
            }
            continue;
        }

        tracing::debug!(active_slots = n_active, "batched decode pass");

        // ── Build [N, 1] token tensor (reuse pre-allocated vec) ──────────
        tokens_vec.clear();
        for &i in &active_indices {
            tokens_vec.push(slots[i].as_ref().unwrap().current_token);
        }

        let tokens_t = match Tensor::from_slice(&tokens_vec, (n_active, 1), model.device())
            .map_err(cerr)
        {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                for &si in &active_indices {
                    let finished = slots[si].take().unwrap();
                    let _ = finished.reply_tx.send(Err(TranslatorError::Model(msg.clone())));
                }
                continue;
            }
        };

        // ── Temporarily move KV caches out ───────────────────────────────
        let mut batch_kv: Vec<SlotKvCache> = active_indices
            .iter()
            .map(|&i| {
                std::mem::replace(
                    &mut slots[i].as_mut().unwrap().decoder.kv_cache,
                    SlotKvCache { layers: Vec::new(), seq_len: 0 },
                )
            })
            .collect();

        #[cfg(feature = "opentelemetry")]
        let _fw = std::time::Instant::now();
        let forward_result = model.forward_batched(&tokens_t, &mut batch_kv);
        #[cfg(feature = "opentelemetry")]
        metrics.decode_forward_ms.record(_fw.elapsed().as_millis() as f64, &[]);

        let all_logits_t = match forward_result {
            Ok(t) => t,
            Err(e) => {
                for (bi, &si) in active_indices.iter().enumerate() {
                    slots[si].as_mut().unwrap().decoder.kv_cache =
                        std::mem::replace(&mut batch_kv[bi], SlotKvCache { layers: Vec::new(), seq_len: 0 });
                }
                let msg = e.to_string();
                for &si in &active_indices {
                    let finished = slots[si].take().unwrap();
                    let _ = finished.reply_tx.send(Err(TranslatorError::Model(msg.clone())));
                }
                continue;
            }
        };

        // ── Restore KV caches ─────────────────────────────────────────────
        for (bi, &si) in active_indices.iter().enumerate() {
            slots[si].as_mut().unwrap().decoder.kv_cache =
                std::mem::replace(&mut batch_kv[bi], SlotKvCache { layers: Vec::new(), seq_len: 0 });
        }

        // ── GPU→CPU transfer ─────────────────────────────────────────────
        // Temperature is applied CPU-side after top-K selection (only ≤40
        // candidates), avoiding a full-vocab GPU kernel every step.
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
        // `all_logits_cpu.into_iter()` takes ownership — no per-slot Vec clone.
        for (mut logits, &slot_idx) in all_logits_cpu.into_iter().zip(active_indices.iter()) {
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

            let tok = sample_token(&mut logits, &mut rng, &mut sample_scratch);
            let at_capacity = slot.output_ids.len() + 1 >= SLOT_CAPACITY;

            if tok == eos_id || at_capacity {
                let finished = slots[slot_idx].take().unwrap();
                #[cfg(feature = "opentelemetry")]
                {
                    use opentelemetry::KeyValue;
                    let cause = if tok == eos_id { "eos" } else { "capacity" };
                    metrics.slots_completed.add(1, &[KeyValue::new("cause", cause)]);
                    metrics.tokens_generated.add(finished.output_ids.len() as u64, &[]);
                }
                let text = model.decode_output_ids(&finished.output_ids);
                let _ = finished.reply_tx.send(text);
            } else {
                slot.output_ids.push(tok);
                slot.current_token = tok;
            }
        }
    }
}

// ── Helper: collect pending prefill items ────────────────────────────────────

/// Non-blocking drain of up to `limit` items from the work queue.
/// Tokenizes each request; on tokenization failure, sends error and skips.
fn collect_pending(
    model: &LoadedGemmaModel,
    work_rx: &crossbeam_channel::Receiver<InferRequest>,
    limit: usize,
    metrics: &Metrics,
) -> Vec<PendingPrefill> {
    let mut pending = Vec::new();
    for _ in 0..limit {
        match work_rx.try_recv() {
            Ok(req) => tokenize_into_pending(model, req, &mut pending, metrics),
            Err(_) => break,
        }
    }
    pending
}

/// Tokenize a single request and push into `pending`, or send error on failure.
#[cfg_attr(not(feature = "opentelemetry"), allow(unused_variables))]
fn tokenize_into_pending(
    model: &LoadedGemmaModel,
    req: InferRequest,
    pending: &mut Vec<PendingPrefill>,
    metrics: &Metrics,
) {
    match model.tokenize(&req.text) {
        Ok(token_ids) => {
            #[cfg(feature = "opentelemetry")]
            metrics.prompt_tokens.record(token_ids.len() as u64, &[]);
            pending.push(PendingPrefill {
                token_ids,
                expected_len: req.expected_output_len,
                reply_tx: req.reply_tx,
            });
        }
        Err(e) => {
            let _ = req.reply_tx.send(Err(e));
        }
    }
}

// ── Batched prefill ───────────────────────────────────────────────────────────

/// Prefill all `pending` requests in one GPU call, sample first tokens, and
/// assign the resulting [`Slot`]s into empty entries of `slots`.
#[cfg_attr(not(feature = "opentelemetry"), allow(unused_variables))]
fn batch_prefill_and_assign(
    model: &LoadedGemmaModel,
    pending: &mut Vec<PendingPrefill>,
    slots: &mut [Option<Slot>],
    eos_id: u32,
    rng: &mut SmallRng,
    metrics: &Metrics,
    sample_scratch: &mut Vec<(u32, f32)>,
) {
    tracing::debug!(batch_size = pending.len(), "prefill batch");
    let seqs: Vec<Vec<u32>> = pending.iter().map(|p| p.token_ids.clone()).collect();
    let mut kv_caches: Vec<SlotKvCache> = (0..seqs.len())
        .map(|_| SlotKvCache::new(model.n_layers()))
        .collect();

    #[cfg(feature = "opentelemetry")]
    let _pf = std::time::Instant::now();

    let all_logits_t = match model.forward_prefill_batched(&seqs, &mut kv_caches) {
        Ok(t) => t,
        Err(e) => {
            let msg = e.to_string();
            for pb in pending.drain(..) {
                let _ = pb.reply_tx.send(Err(TranslatorError::Model(msg.clone())));
            }
            return;
        }
    };

    #[cfg(feature = "opentelemetry")]
    metrics.prefill_ms.record(_pf.elapsed().as_millis() as f64, &[]);

    // Transfer all logits to CPU. Temperature is applied CPU-side after top-K.
    let all_logits_cpu = match all_logits_t
        .to_dtype(DType::F32)
        .and_then(|t| t.to_vec2::<f32>())
    {
        Ok(v) => v,
        Err(e) => {
            let msg = e.to_string();
            for pb in pending.drain(..) {
                let _ = pb.reply_tx.send(Err(TranslatorError::Model(msg.clone())));
            }
            return;
        }
    };

    let mut kv_iter = kv_caches.into_iter();
    for (pb, mut logits) in pending.drain(..).zip(all_logits_cpu) {
        let kv = kv_iter.next().unwrap();

        apply_decoding_filters(&mut logits, &[]);
        apply_length_bias(&mut logits, eos_id, 0, pb.expected_len);
        let first_token = sample_token(&mut logits, rng, sample_scratch);

        if first_token == eos_id {
            let _ = pb.reply_tx.send(Ok(String::new()));
            continue;
        }

        let decoder = GemmaSlotDecoder::new(kv, model.device().clone());

        // Assign to the first empty slot.
        for slot in slots.iter_mut() {
            if slot.is_none() {
                *slot = Some(Slot {
                    decoder,
                    current_token: first_token,
                    output_ids: vec![first_token],
                    expected_len: pb.expected_len,
                    reply_tx: pb.reply_tx,
                });
                break;
            }
        }
    }
}
