//! Batched Gemma 3 decoder — stateless weights, external per-slot KV cache.
//!
//! Forked from `candle_transformers::models::quantized_gemma3` with these changes:
//!  - KV cache moved out of `LayerWeights` into external [`SlotKvCache`] structs
//!  - [`ModelWeights::forward`] takes `&self` + `&mut SlotKvCache` (stateless weights)
//!  - [`ModelWeights::forward_batched`] runs one forward pass across N active slots
//!
//! This lets the continuous scheduler replace N serial single-slot `forward` calls
//! with one `forward_batched` call, amortising the weight-read cost across all slots.

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::D;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

pub const MAX_SEQ_LEN: usize = 131072;
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

// ── External KV cache ─────────────────────────────────────────────────────────

/// Per-slot KV cache owned by the scheduler, passed to `ModelWeights::forward`.
///
/// `layers[i]` accumulates `(key, value)` tensors for layer `i`.  Initially
/// all `None`; grows on each forward call.
/// `seq_len` is the total number of tokens consumed so far (= `index_pos` for
/// the next `forward` call).
pub struct SlotKvCache {
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// Tokens consumed so far — `index_pos` for the next forward call.
    pub seq_len: usize,
}

impl SlotKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self { layers: vec![None; n_layers], seq_len: 0 }
    }
}

// ── Local repeat_kv ───────────────────────────────────────────────────────────

/// Repeat key/value heads `n_rep` times to expand GQA groups to full head count.
///
/// Input `xs`: `[b_sz, n_kv_head, seq_len, head_dim]`
/// Output:     `[b_sz, n_kv_head * n_rep, seq_len, head_dim]`
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    // Cat n_rep copies along the seq_len dim then reshape — equivalent to head repetition.
    Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
}

// ── QMatMul wrapper ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

// ── Rotary embeddings ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    /// Standard single-position RoPE for prefill / single-slot decode.
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Batched RoPE for N slots each at a different position (decode only, seq_len=1).
    ///
    /// `q`: `[N, n_heads, 1, head_dim]`, `k`: `[N, n_kv_heads, 1, head_dim]`
    /// Returns tensors of the same shape.
    fn apply_rotary_emb_batched(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let half = q.dim(D::Minus1)? / 2;

        // Build per-position cos/sin rows → each [1, 1, half], stacked to [N, 1, 1, half]
        let cos_rows: Vec<Tensor> = positions
            .iter()
            .map(|&p| -> Result<Tensor> { self.cos.i(p)?.reshape((1, 1, half)) })
            .collect::<Result<_>>()?;
        let sin_rows: Vec<Tensor> = positions
            .iter()
            .map(|&p| -> Result<Tensor> { self.sin.i(p)?.reshape((1, 1, half)) })
            .collect::<Result<_>>()?;

        let cos = Tensor::stack(&cos_rows, 0)?; // [N, 1, 1, half]
        let sin = Tensor::stack(&sin_rows, 0)?;

        // Manual rotate-half: broadcast cos/sin [N,1,1,half] → [N, n_heads, 1, half]
        let rope = |t: &Tensor| -> Result<Tensor> {
            let t1 = t.narrow(D::Minus1, 0, half)?;
            let t2 = t.narrow(D::Minus1, half, half)?;
            let new_t1 = t1.broadcast_mul(&cos)?.sub(&t2.broadcast_mul(&sin)?)?;
            let new_t2 = t1.broadcast_mul(&sin)?.add(&t2.broadcast_mul(&cos)?)?;
            Tensor::cat(&[new_t1, new_t2], D::Minus1)
        };
        Ok((rope(q)?, rope(k)?))
    }
}

// ── Layer weights ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerWeights {
    // Attention projections
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    // Per-head norms
    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,

    // Layer norms
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,

    mlp: Mlp,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,

    sliding_window_size: Option<usize>,
    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,

    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    /// Causal + optional sliding-window attention mask for prefill.
    /// Shape: `[b_sz, 1, seq_len, seq_len + index_pos]`.
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i { 0u32 } else { 1u32 }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?.to_dtype(dtype)
    }

    /// Single-slot attention — takes external `kv` cache instead of an internal field.
    ///
    /// Used by [`ModelWeights::forward`] for both prefill and single-slot decode.
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self.rotary_embedding.apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match kv.as_ref() {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        *kv = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output =
            attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.q_dim))?;
        self.attention_wo.forward(&attn_output)
    }

    /// Batched decode-step attention for N active slots.
    ///
    /// All slots contribute exactly one new token (`seq_len = 1` per slot).
    /// KV caches are updated in-place.  Padded KV and per-slot masks handle
    /// the fact that slots are at different positions.
    ///
    /// `x`:          `[N, 1, dim]`
    /// `positions`:  `kv_cache.seq_len` for each slot (position of the new token)
    /// `max_kv_len`: `max(positions)` — KV pad target width
    ///
    /// Returns `[N, 1, dim]`.
    fn forward_attn_batched(
        &self,
        x: &Tensor,
        kv_caches: &mut [SlotKvCache],
        layer_idx: usize,
        positions: &[usize],
        max_kv_len: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let n = positions.len();
        let (_, seq_len, _) = x.dims3()?; // seq_len == 1

        // Project Q/K/V: [N, 1, dim] → [N, n_head/n_kv_head, 1, head_dim]
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.reshape((n, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((n, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((n, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self.rotary_embedding.apply_rotary_emb_batched(&q, &k, positions)?;

        // Build per-slot padded KV tensors and attention masks
        let total_kv_len = max_kv_len + 1;
        let device = x.device();
        let mut k_list: Vec<Tensor> = Vec::with_capacity(n);
        let mut v_list: Vec<Tensor> = Vec::with_capacity(n);
        let mut mask_list: Vec<Tensor> = Vec::with_capacity(n);

        for i in 0..n {
            let pos_i = positions[i]; // = seq_len BEFORE this step

            // Extract new K/V for slot i: [1, n_kv_head, 1, head_dim]
            let k_new_i = k.i(i)?.unsqueeze(0)?;
            let v_new_i = v.i(i)?.unsqueeze(0)?;

            // Cat with existing cache
            let (k_updated, v_updated) = match &kv_caches[i].layers[layer_idx] {
                None => (k_new_i, v_new_i),
                Some((k_cache, v_cache)) => {
                    let k_c = Tensor::cat(&[k_cache, &k_new_i], 2)?;
                    let v_c = Tensor::cat(&[v_cache, &v_new_i], 2)?;
                    (k_c, v_c)
                }
            };
            // k_updated: [1, n_kv_head, pos_i+1, head_dim]

            // Pad to total_kv_len so we can stack across the batch
            let cur_kv_len = pos_i + 1;
            let pad_len = total_kv_len - cur_kv_len;
            let (k_padded, v_padded) = if pad_len > 0 {
                let k_pad = Tensor::zeros(
                    (1, self.n_kv_head, pad_len, self.head_dim),
                    k_updated.dtype(),
                    k_updated.device(),
                )?;
                let v_pad = Tensor::zeros(
                    (1, self.n_kv_head, pad_len, self.head_dim),
                    v_updated.dtype(),
                    v_updated.device(),
                )?;
                (
                    Tensor::cat(&[&k_updated, &k_pad], 2)?,
                    Tensor::cat(&[&v_updated, &v_pad], 2)?,
                )
            } else {
                (k_updated.clone(), v_updated.clone())
            };

            // Store unpadded KV in the cache
            kv_caches[i].layers[layer_idx] = Some((k_updated, v_updated));

            k_list.push(k_padded);
            v_list.push(v_padded);

            // Per-slot mask: attend to [sliding_start..=pos_i], mask everything else
            let sliding_start = match self.sliding_window_size {
                Some(w) => pos_i.saturating_sub(w.saturating_sub(1)),
                None => 0,
            };
            let mask_vals: Vec<u32> = (0..total_kv_len)
                .map(|j| if j >= sliding_start && j <= pos_i { 1u32 } else { 0u32 })
                .collect();
            let mask_i =
                Tensor::from_slice(&mask_vals, (1usize, 1usize, 1usize, total_kv_len), device)?;
            mask_list.push(mask_i);
        }

        // Stack across batch
        let k_all = Tensor::cat(&k_list, 0)?; // [N, n_kv_head, total_kv_len, head_dim]
        let v_all = Tensor::cat(&v_list, 0)?;
        let mask_all = Tensor::cat(&mask_list, 0)?; // [N, 1, 1, total_kv_len]

        // GQA: expand kv heads to match query head count
        let k_all = repeat_kv(k_all, self.n_head / self.n_kv_head)?;
        let v_all = repeat_kv(v_all, self.n_head / self.n_kv_head)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k_all.transpose(2, 3)?)? * scale)?;
        // attn_weights: [N, n_head, 1, total_kv_len]

        let mask_bc = mask_all.broadcast_as(attn_weights.shape())?;
        let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
        attn_weights = mask_bc.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_all)?;
        // attn_output: [N, n_head, 1, head_dim]

        let attn_output =
            attn_output.transpose(1, 2)?.reshape((n, seq_len, self.q_dim))?;
        self.attention_wo.forward(&attn_output)
    }
}

// ── Model weights ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| ct.metadata.contains_key(&format!("{}.attention.head_count", p)))
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let _rope_freq_scaling_factor = md_get("rope.scaling.factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR);

        let q_dim = head_count * key_length;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq =
                ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk =
                ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv =
                ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up =
                ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };

            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let sliding_window_size = is_sliding.then_some(sliding_window_size);
            let layer_rope_frequency =
                if is_sliding { rope_freq_base_sliding } else { rope_freq_base };

            let rotary_embedding =
                RotaryEmbedding::new(key_length, layer_rope_frequency, device)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            // Note: no kv_cache field — KV is held externally in SlotKvCache.
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                span_attn,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            span,
            span_output,
        })
    }

    /// Return the number of transformer layers (= number of KV cache entries per slot).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Single-slot forward pass (prefill or single-slot decode).
    ///
    /// - `x`:         `[1, seq_len]` token IDs (U32)
    /// - `index_pos`: position of the first token in the full sequence (for RoPE)
    /// - `kv_cache`:  per-slot KV state — updated in-place
    ///
    /// Returns `[1, vocab_size]` logits for the last token position.
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        kv_cache: &mut SlotKvCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let _enter = self.span.enter();

        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(
                &x,
                attention_mask.as_ref(),
                index_pos,
                &mut kv_cache.layers[layer_idx],
            )?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;
            drop(_enter);

            layer_in = x;
        }

        kv_cache.seq_len = index_pos + seq_len;

        let _enter = self.span_output.enter();
        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;
        Ok(output)
    }

    /// N-slot batched decode step.
    ///
    /// - `tokens`:    `[N, 1]` — one current token per active slot
    /// - `kv_caches`: per-slot KV state — updated in-place
    ///
    /// Returns `[N, vocab_size]` logits (one row per slot).
    pub fn forward_batched(
        &self,
        tokens: &Tensor,
        kv_caches: &mut [SlotKvCache],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let positions: Vec<usize> = kv_caches.iter().map(|kv| kv.seq_len).collect();
        let max_kv_len = positions.iter().copied().max().unwrap_or(0);

        let mut layer_in = self.tok_embeddings.forward(tokens)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
        // layer_in: [N, 1, dim]

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn_batched(
                &x,
                kv_caches,
                layer_idx,
                &positions,
                max_kv_len,
            )?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;
            drop(_enter);

            layer_in = x;
        }

        // Advance all slot seq_lens by 1
        for kv in kv_caches.iter_mut() {
            kv.seq_len += 1;
        }

        let _enter = self.span_output.enter();
        // layer_in: [N, 1, dim] → take the single output position → [N, dim]
        let x = layer_in.i((.., 0usize, ..))?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;
        Ok(output)
    }
}
