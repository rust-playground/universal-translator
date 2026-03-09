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
use rayon::prelude::*;

use crate::error::TranslatorError;
use crate::model::LoadedGemmaModel;
use crate::model_batched::SlotKvCache;
use crate::scheduler::decoder::GemmaSlotDecoder;
use crate::scheduler::sampling::{
    apply_decoding_filters, apply_length_bias, check_tail_repeat, compute_ban_indices,
    compute_length_bias, sample_token, REPETITION_PENALTY, TOP_K,
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

/// How long to wait for additional requests after the first one arrives when
/// the scheduler is idle.  This lets staggered `spawn_blocking` threads (which
/// start with ~1–5ms jitter) all reach the channel before prefill fires.
/// 10ms adds at most 10ms to first-batch latency but can multiply throughput
/// when many requests arrive "simultaneously".
const PREFILL_ACCUMULATION_DELAY: std::time::Duration = std::time::Duration::from_millis(10);

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
    decoder: GemmaSlotDecoder,
    /// Current token to feed as input on the next decode step.
    current_token: u32,
    /// All output token IDs confirmed so far.
    output_ids: Vec<u32>,
    /// Predicted natural endpoint for EOS bias (decoupled from SLOT_CAPACITY).
    expected_len: usize,
    index: usize,
    reply_tx: mpsc::Sender<(usize, Result<String, TranslatorError>)>,
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
    // Scratch buffer for sample_token (prefill path only).
    let mut sample_scratch: Vec<(u32, f32)> = Vec::with_capacity(TOP_K + 1);

    // ── GPU filter pipeline constants (Phase 2) ───────────────────────────
    // log(PENALTY) subtracted from penalised-token logits — equivalent to
    // dividing probability by PENALTY in probability space.
    let log_rep_delta: f32 = -(REPETITION_PENALTY.ln());
    // Maximum n-gram ban tokens per slot (padded with token 0 for unused slots).
    const MAX_BAN: usize = 32;
    // Pre-allocated flat CPU buffers for GPU scatter tensors — resized each step.
    let mut add_idx_flat: Vec<u32> = Vec::new();
    let mut add_val_flat: Vec<f32> = Vec::new();
    let mut ban_idx_flat: Vec<u32> = Vec::new();
    let mut force_eos_flags: Vec<u8> = Vec::new();
    // Lazily initialised [1, vocab_size] tensor: -inf everywhere, 0 at eos_id.
    // Built on the first decode step when vocab_size is known from logit shape.
    let mut eos_only_logits: Option<Tensor> = None;

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
                    // Accumulation window: collect additional items arriving within
                    // PREFILL_ACCUMULATION_DELAY so concurrent requests aren't split
                    // across separate prefill batches due to spawn_blocking thread
                    // startup jitter (~1–5ms between threads).
                    let deadline = std::time::Instant::now() + PREFILL_ACCUMULATION_DELAY;
                    loop {
                        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match work_rx.recv_timeout(remaining) {
                            Ok(r) => tokenize_into_pending(model, r, &mut pending, metrics),
                            Err(_) => break,
                        }
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
                    let _ = finished.reply_tx.send((finished.index, Err(TranslatorError::Model(msg.clone()))));
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
                    let _ = finished.reply_tx.send((finished.index, Err(TranslatorError::Model(msg.clone()))));
                }
                continue;
            }
        };

        // ── Restore KV caches ─────────────────────────────────────────────
        for (bi, &si) in active_indices.iter().enumerate() {
            slots[si].as_mut().unwrap().decoder.kv_cache =
                std::mem::replace(&mut batch_kv[bi], SlotKvCache { layers: Vec::new(), seq_len: 0 });
        }

        // ── GPU filter pipeline (Phase 2) ────────────────────────────────
        // Cast logits to F32 (may be BF16 on some backends). Creates a new
        // contiguous tensor suitable for in-place scatter mutation.
        let all_logits_t = match all_logits_t.to_dtype(DType::F32).map_err(cerr) {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                for &si in &active_indices {
                    if let Some(finished) = slots[si].take() {
                        let _ = finished.reply_tx.send((finished.index, Err(TranslatorError::Model(msg.clone()))));
                    }
                }
                continue;
            }
        };

        // Derive vocab_size from logit tensor shape; initialise eos_only_logits once.
        let vocab_size = match all_logits_t.dim(1).map_err(cerr) {
            Ok(v) => v,
            Err(e) => {
                let msg = e.to_string();
                for &si in &active_indices {
                    if let Some(finished) = slots[si].take() {
                        let _ = finished.reply_tx.send((finished.index, Err(TranslatorError::Model(msg.clone()))));
                    }
                }
                continue;
            }
        };
        let eos_only = match eos_only_logits {
            Some(ref t) => t,
            None => {
                let mut v = vec![f32::NEG_INFINITY; vocab_size];
                v[eos_id as usize] = 0.0;
                match Tensor::from_slice(&v, (1, vocab_size), model.device()).map_err(cerr) {
                    Ok(t) => eos_only_logits.insert(t),
                    Err(e) => {
                        let msg = e.to_string();
                        for &si in &active_indices {
                            if let Some(finished) = slots[si].take() {
                                let _ = finished.reply_tx.send((
                                    finished.index,
                                    Err(TranslatorError::Model(msg.clone())),
                                ));
                            }
                        }
                        continue;
                    }
                }
            }
        };

        // Build per-slot filter data in parallel (CPU work while GPU is idle).
        // Each element: (slot_idx, output_ids, eos_bias, ban_ids, force_eos_flag)
        type SlotFilterData = (usize, Vec<u32>, f32, Vec<u32>, bool);
        let per_slot: Vec<SlotFilterData> = active_indices
            .par_iter()
            .map(|&slot_idx| {
                let slot = slots[slot_idx].as_ref().unwrap();
                let output_ids = slot.output_ids.clone();
                let eos_bias = compute_length_bias(output_ids.len(), slot.expected_len);
                let ban_ids = compute_ban_indices(&output_ids);
                let force_eos =
                    check_tail_repeat(&output_ids) || output_ids.len() + 1 >= SLOT_CAPACITY;
                (slot_idx, output_ids, eos_bias, ban_ids, force_eos)
            })
            .collect();

        // Flatten into rectangular CPU tensors:
        //   add_idx/val [N, max_seq_len + 1] — rep penalty + EOS bias (one scatter call)
        //   ban_idx     [N, MAX_BAN]          — n-gram banned tokens
        //   force_eos   [N]                   — force-EOS flag
        // Extra "+1" column holds the EOS index with its per-slot length-bias value.
        let max_seq_len = per_slot.iter().map(|(_, ids, _, _, _)| ids.len()).max().unwrap_or(0);
        let col_width = max_seq_len + 1; // last column = EOS length bias

        add_idx_flat.clear();
        add_val_flat.clear();
        ban_idx_flat.clear();
        force_eos_flags.clear();

        for (_, output_ids, eos_bias, ban_ids, force_eos) in &per_slot {
            // Rep penalty columns (pad short seqs with token 0 and value 0.0).
            for &tok in output_ids.iter() {
                add_idx_flat.push(tok);
                add_val_flat.push(log_rep_delta);
            }
            for _ in output_ids.len()..max_seq_len {
                add_idx_flat.push(0);
                add_val_flat.push(0.0);
            }
            // EOS length bias (last column).
            add_idx_flat.push(eos_id);
            add_val_flat.push(*eos_bias);

            // N-gram ban columns (pad with token 0 and -inf for unused entries).
            let n_bans = ban_ids.len().min(MAX_BAN);
            for &tok in ban_ids.iter().take(n_bans) {
                ban_idx_flat.push(tok);
            }
            ban_idx_flat.extend(std::iter::repeat_n(0u32, MAX_BAN - n_bans));

            force_eos_flags.push(*force_eos as u8);
        }

        // Upload small index/value tensors to GPU and apply filters in-place.
        let gpu_result = (|| -> Result<Vec<u32>, TranslatorError> {
            let add_idx_t =
                Tensor::from_slice(&add_idx_flat, (n_active, col_width), model.device())
                    .map_err(cerr)?;
            let add_val_t =
                Tensor::from_slice(&add_val_flat, (n_active, col_width), model.device())
                    .map_err(cerr)?;
            let ban_idx_t =
                Tensor::from_slice(&ban_idx_flat, (n_active, MAX_BAN), model.device())
                    .map_err(cerr)?;
            let neg_inf_t =
                Tensor::full(f32::NEG_INFINITY, (n_active, MAX_BAN), model.device())
                    .map_err(cerr)?;

            // In-place: rep penalty + EOS length bias.
            all_logits_t.scatter_add_set(&add_idx_t, &add_val_t, 1).map_err(cerr)?;
            // In-place: n-gram bans (set banned positions to -inf).
            all_logits_t.scatter_set(&ban_idx_t, &neg_inf_t, 1).map_err(cerr)?;

            // Force EOS for tail-repeat / capacity-exceeded slots.
            let flags_t =
                Tensor::from_slice(&force_eos_flags, (n_active, 1usize), model.device())
                    .map_err(cerr)?
                    .broadcast_as((n_active, vocab_size))
                    .map_err(cerr)?;
            let eos_only_broadcast =
                eos_only.broadcast_as((n_active, vocab_size)).map_err(cerr)?;
            let filtered_logits_t =
                flags_t.where_cond(&eos_only_broadcast, &all_logits_t).map_err(cerr)?;

            // GPU argmax → [N] u32 token IDs (only N×4 bytes transferred).
            filtered_logits_t.argmax(1).map_err(cerr)?.to_vec1::<u32>().map_err(cerr)
        })();

        let tok_ids = match gpu_result {
            Ok(v) => v,
            Err(e) => {
                let msg = e.to_string();
                for &si in &active_indices {
                    if let Some(finished) = slots[si].take() {
                        let _ = finished.reply_tx.send((finished.index, Err(TranslatorError::Model(msg.clone()))));
                    }
                }
                continue;
            }
        };

        // ── Serial retire/update — mutates slots[], cheap ─────────────────
        for (i, tok) in tok_ids.into_iter().enumerate() {
            let slot_idx = per_slot[i].0;
            let at_capacity = slots[slot_idx].as_ref().unwrap().output_ids.len() + 1 >= SLOT_CAPACITY;
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
                let _ = finished.reply_tx.send((finished.index, text));
            } else {
                let slot = slots[slot_idx].as_mut().unwrap();
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
                index: req.index,
                reply_tx: req.reply_tx,
            });
        }
        Err(e) => {
            let _ = req.reply_tx.send((req.index, Err(e)));
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
                let _ = pb.reply_tx.send((pb.index, Err(TranslatorError::Model(msg.clone()))));
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
                let _ = pb.reply_tx.send((pb.index, Err(TranslatorError::Model(msg.clone()))));
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
            let _ = pb.reply_tx.send((pb.index, Ok(String::new())));
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
                    index: pb.index,
                    reply_tx: pb.reply_tx,
                });
                break;
            }
        }
    }
}
