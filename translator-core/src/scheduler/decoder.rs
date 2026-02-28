//! Custom T5 decoder with externalized KV cache.
//!
//! Replicates `candle_transformers::quantized_t5`'s decoder stack but exposes
//! the KV state in [`DecoderKvCache`] so a scheduler can manage it per-slot
//! without cloning the whole model. The encoder remains in
//! `LoadedModel::model_template` and is not reproduced here.
//!
//! **Phase 2a** — validates numerical equivalence with `translate_greedy_batched`:
//! `translate_with_custom_decoder` must produce byte-identical output.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Activation;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;

use crate::error::TranslatorError;

// ── Config parsing ─────────────────────────────────────────────────────────────
// qt5::Config fields are private, so we parse config.json ourselves.

struct ParsedConfig {
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    n_heads: usize,
    num_decoder_layers: usize,
    vocab_size: usize,
    num_buckets: u32,
    max_distance: u32,
    tie_word_embeddings: bool,
    layer_norm_eps: f64,
    ff_gated: bool,
    ff_act: Activation,
}

fn parse_config(config_str: &str) -> Result<ParsedConfig, TranslatorError> {
    let v: serde_json::Value = serde_json::from_str(config_str)
        .map_err(|e| TranslatorError::Model(format!("config parse: {e}")))?;

    let get_usize = |key: &str| -> Result<usize, TranslatorError> {
        v[key]
            .as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| TranslatorError::Model(format!("config missing '{key}'")))
    };

    let d_model = get_usize("d_model")?;
    let d_kv = get_usize("d_kv")?;
    let d_ff = get_usize("d_ff")?;
    let n_heads = get_usize("num_heads")?;
    let num_layers = get_usize("num_layers")?;
    let num_decoder_layers = v["num_decoder_layers"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(num_layers);
    let vocab_size = get_usize("vocab_size")?;
    let num_buckets = v["relative_attention_num_buckets"]
        .as_u64()
        .unwrap_or(32) as u32;
    let max_distance = v["relative_attention_max_distance"]
        .as_u64()
        .unwrap_or(128) as u32;
    let tie_word_embeddings = v["tie_word_embeddings"].as_bool().unwrap_or(false);
    let layer_norm_eps = v["layer_norm_epsilon"]
        .as_f64()
        .or_else(|| v["layer_norm_eps"].as_f64())
        .unwrap_or(1e-6);

    let ff_proj = v["feed_forward_proj"].as_str().unwrap_or("relu");
    let ff_gated = ff_proj.starts_with("gated-");
    let ff_act = if ff_proj.contains("gelu") {
        Activation::NewGelu
    } else {
        Activation::Relu
    };

    Ok(ParsedConfig {
        d_model,
        d_kv,
        d_ff,
        n_heads,
        num_decoder_layers,
        vocab_size,
        num_buckets,
        max_distance,
        tie_word_embeddings,
        layer_norm_eps,
        ff_gated,
        ff_act,
    })
}

// ── Safety ────────────────────────────────────────────────────────────────────
// All weight tensors (Tensor, QMatMul) are Arc-backed; CustomT5Decoder is a
// read-only template after load — mutations occur only in the caller's
// DecoderKvCache.  Same reasoning as LoadedModel's unsafe impls.
unsafe impl Send for CustomT5Decoder {}
unsafe impl Sync for CustomT5Decoder {}

// ── Error helper ──────────────────────────────────────────────────────────────
fn cerr(e: candle_core::Error) -> TranslatorError {
    TranslatorError::Model(e.to_string())
}

// ── T5 RMS layer-norm ─────────────────────────────────────────────────────────
/// Matches `T5LayerNorm::forward` exactly: variance is computed in f32 for
/// numerical stability; division uses the original-dtype tensor; output is cast
/// back and scaled by the learned weight.
fn t5_rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let dtype = xs.dtype();
    let variance = xs.to_dtype(DType::F32)?.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let xs = xs.broadcast_div(&(variance + eps)?.sqrt()?)?;
    xs.to_dtype(dtype)?.broadcast_mul(weight)
}

// ── KV cache ──────────────────────────────────────────────────────────────────

/// Per-batch decode-state KV cache.
///
/// * `self_k[i]` / `self_v[i]` — accumulated self-attention K/V for layer `i`;
///   shape `[B, n_heads, t, d_kv]`, grows by 1 each [`CustomT5Decoder::decode_step`].
/// * `cross_k[i]` / `cross_v[i]` — cross-attention K/V (encoder-derived, constant);
///   shape `[B, n_heads, enc_seq, d_kv]`, populated once by
///   [`CustomT5Decoder::compute_cross_kv`].
pub struct DecoderKvCache {
    pub self_k: Vec<Tensor>,
    pub self_v: Vec<Tensor>,
    pub cross_k: Vec<Tensor>,
    pub cross_v: Vec<Tensor>,
}

// ── Per-layer weight bundle ───────────────────────────────────────────────────

struct LayerWeights {
    // Self-attention
    sa_q: QMatMul,
    sa_k: QMatMul,
    sa_v: QMatMul,
    sa_o: QMatMul,
    sa_norm: Tensor, // [d_model] dequantized

    // Cross-attention
    ca_q: QMatMul,
    ca_k: QMatMul,
    ca_v: QMatMul,
    ca_o: QMatMul,
    ca_norm: Tensor, // [d_model] dequantized

    // Feed-forward
    ff_wi_0: Option<QMatMul>, // gated T5v1.1: gate branch (wi_0)
    ff_wi_1: Option<QMatMul>, // gated T5v1.1: linear branch (wi_1)
    ff_wi:   Option<QMatMul>, // non-gated T5v1.0
    ff_wo:   QMatMul,
    ff_norm: Tensor, // [d_model] dequantized
    ff_gated: bool,
    ff_act:   Activation,
}

// ── CustomT5Decoder ───────────────────────────────────────────────────────────

/// Custom T5 decoder with externalized KV cache.
///
/// Weight tensors share the underlying `Arc<QTensor>` storage with
/// `model_template` — loading adds only reference-count increments, no extra
/// VRAM/RAM.
pub struct CustomT5Decoder {
    layers: Vec<LayerWeights>,
    final_layer_norm: Tensor,  // [d_model] dequantized
    embed_tokens: Tensor,      // [vocab_size, d_model] dequantized
    lm_head: Option<QMatMul>,  // None when tie_word_embeddings = true
    /// [num_buckets, n_heads] — from decoder layer 0; reused across all layers.
    rel_attn_bias: Tensor,
    n_heads: usize,
    d_kv: usize,
    d_model: usize,
    inner_dim: usize, // n_heads * d_kv
    tie_word_embeddings: bool,
    relative_attention_num_buckets: u32,
    relative_attention_max_distance: u32,
    layer_norm_eps: f64,
    device: Device,
}

impl CustomT5Decoder {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Load decoder weights from the same `VarBuilder` used by `LoadedModel`.
    ///
    /// `VarBuilder` is `Clone`-cheap (Arc-backed), so pass a clone of the vb
    /// used for `T5ForConditionalGeneration::load`.
    ///
    /// `config_str` — raw contents of `config.json` from the model directory.
    /// We parse required fields ourselves because `qt5::Config` fields are private.
    pub fn load(vb: QVarBuilder, config_str: &str) -> Result<Self, TranslatorError> {
        let cfg = parse_config(config_str)?;
        let inner_dim = cfg.n_heads * cfg.d_kv;
        let device = vb.device().clone();

        // Shared input/output embeddings (dequantized to f32).
        // Mirror T5ForConditionalGeneration::load's shared_vb logic.
        let shared_vb = if vb.contains_key("shared.weight") {
            vb.pp("shared")
        } else {
            vb.pp("decoder").pp("embed_tokens")
        };
        let embed_tokens = shared_vb
            .get((cfg.vocab_size, cfg.d_model), "weight")
            .and_then(|t| t.dequantize(&device))
            .map_err(|e| TranslatorError::Model(format!("embed_tokens: {e}")))?;

        // Helpers
        let qmm = |vb: QVarBuilder, out: usize, inp: usize| {
            QMatMul::new(out, inp, vb).map_err(|e| TranslatorError::Model(e.to_string()))
        };
        let norm_w = |vb: QVarBuilder, dim: usize| {
            vb.get(dim, "weight")
                .and_then(|t| t.dequantize(&device))
                .map_err(|e: candle_core::Error| TranslatorError::Model(e.to_string()))
        };

        let dec_vb = vb.pp("decoder");
        let mut layers = Vec::with_capacity(cfg.num_decoder_layers);
        let mut rel_attn_bias_opt: Option<Tensor> = None;

        for i in 0..cfg.num_decoder_layers {
            // T5Block::load uses vb.pp("block.{i}").pp("layer") then "0", "1", "2"
            let block = dec_vb.pp(format!("block.{i}")).pp("layer");

            // ── Layer 0: self-attention ────────────────────────────────────────
            let sa0 = block.pp("0");
            let sa_attn = sa0.pp("SelfAttention");
            let sa_q = qmm(sa_attn.pp("q"), cfg.d_model, inner_dim)?;
            let sa_k = qmm(sa_attn.pp("k"), cfg.d_model, inner_dim)?;
            let sa_v = qmm(sa_attn.pp("v"), cfg.d_model, inner_dim)?;
            let sa_o = qmm(sa_attn.pp("o"), inner_dim, cfg.d_model)?;
            let sa_norm = norm_w(sa0.pp("layer_norm"), cfg.d_model)?;

            // Relative position bias lives only in layer 0; reused for all layers.
            if i == 0 {
                let bias = sa_attn
                    .pp("relative_attention_bias")
                    .get((cfg.num_buckets as usize, cfg.n_heads), "weight")
                    .and_then(|t| t.dequantize(&device))
                    .map_err(|e: candle_core::Error| TranslatorError::Model(e.to_string()))?;
                rel_attn_bias_opt = Some(bias);
            }

            // ── Layer 1: cross-attention ───────────────────────────────────────
            let ca1 = block.pp("1");
            let ca_attn = ca1.pp("EncDecAttention");
            let ca_q = qmm(ca_attn.pp("q"), cfg.d_model, inner_dim)?;
            let ca_k = qmm(ca_attn.pp("k"), cfg.d_model, inner_dim)?;
            let ca_v = qmm(ca_attn.pp("v"), cfg.d_model, inner_dim)?;
            let ca_o = qmm(ca_attn.pp("o"), inner_dim, cfg.d_model)?;
            let ca_norm = norm_w(ca1.pp("layer_norm"), cfg.d_model)?;

            // ── Layer 2: feed-forward ──────────────────────────────────────────
            let ff2 = block.pp("2");
            let ff_dense = ff2.pp("DenseReluDense");
            let ff_norm = norm_w(ff2.pp("layer_norm"), cfg.d_model)?;
            let (ff_wi_0, ff_wi_1, ff_wi) = if cfg.ff_gated {
                (
                    Some(qmm(ff_dense.pp("wi_0"), cfg.d_model, cfg.d_ff)?),
                    Some(qmm(ff_dense.pp("wi_1"), cfg.d_model, cfg.d_ff)?),
                    None,
                )
            } else {
                (None, None, Some(qmm(ff_dense.pp("wi"), cfg.d_model, cfg.d_ff)?))
            };
            let ff_wo = qmm(ff_dense.pp("wo"), cfg.d_ff, cfg.d_model)?;

            layers.push(LayerWeights {
                sa_q, sa_k, sa_v, sa_o, sa_norm,
                ca_q, ca_k, ca_v, ca_o, ca_norm,
                ff_wi_0, ff_wi_1, ff_wi, ff_wo, ff_norm,
                ff_gated: cfg.ff_gated,
                ff_act: cfg.ff_act,
            });
        }

        let rel_attn_bias = rel_attn_bias_opt
            .expect("decoder must have at least one layer");

        let final_layer_norm = norm_w(dec_vb.pp("final_layer_norm"), cfg.d_model)?;

        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(
                QMatMul::new(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
            )
        };

        Ok(Self {
            layers,
            final_layer_norm,
            embed_tokens,
            lm_head,
            rel_attn_bias,
            n_heads: cfg.n_heads,
            d_kv: cfg.d_kv,
            d_model: cfg.d_model,
            inner_dim,
            tie_word_embeddings: cfg.tie_word_embeddings,
            relative_attention_num_buckets: cfg.num_buckets,
            relative_attention_max_distance: cfg.max_distance,
            layer_norm_eps: cfg.layer_norm_eps,
            device,
        })
    }

    // ── KV cache helpers ──────────────────────────────────────────────────────

    /// Create an empty [`DecoderKvCache`] for `batch_size` sequences.
    ///
    /// `self_k[i]` / `self_v[i]` start as `[B, n_heads, 0, d_kv]` and grow by
    /// 1 each [`decode_step`].  `cross_k` / `cross_v` are left empty until
    /// [`compute_cross_kv`] is called.
    pub fn new_kv_cache(&self, batch_size: usize) -> Result<DecoderKvCache, TranslatorError> {
        let nl = self.layers.len();
        let empty = || -> candle_core::Result<Vec<Tensor>> {
            (0..nl)
                .map(|_| {
                    Tensor::zeros(
                        (batch_size, self.n_heads, 0usize, self.d_kv),
                        DType::F32,
                        &self.device,
                    )
                })
                .collect()
        };
        Ok(DecoderKvCache {
            self_k: empty().map_err(cerr)?,
            self_v: empty().map_err(cerr)?,
            cross_k: vec![],
            cross_v: vec![],
        })
    }

    /// Pre-compute and store cross-attention K/V from encoder hidden states.
    ///
    /// `encoder_hidden` — `[B, enc_seq, d_model]`.  Call once per batch after
    /// encoding; the resulting tensors are reused unchanged across all decode
    /// steps (the built-in decoder recomputes them per step — precomputing is
    /// numerically identical and avoids the redundant work).
    pub fn compute_cross_kv(
        &self,
        encoder_hidden: &Tensor,
        cache: &mut DecoderKvCache,
    ) -> Result<(), TranslatorError> {
        let (b, enc_seq, _) = encoder_hidden.dims3().map_err(cerr)?;
        let mut cross_k = Vec::with_capacity(self.layers.len());
        let mut cross_v = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let k = layer
                .ca_k
                .forward(encoder_hidden)
                .map_err(cerr)?
                .reshape((b, enc_seq, self.n_heads, self.d_kv))
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?
                .contiguous()
                .map_err(cerr)?;
            let v = layer
                .ca_v
                .forward(encoder_hidden)
                .map_err(cerr)?
                .reshape((b, enc_seq, self.n_heads, self.d_kv))
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?
                .contiguous()
                .map_err(cerr)?;
            cross_k.push(k);
            cross_v.push(v);
        }

        cache.cross_k = cross_k;
        cache.cross_v = cross_v;
        Ok(())
    }

    // ── Decode step ───────────────────────────────────────────────────────────

    /// Run one greedy decode step for a batch.
    ///
    /// `input_ids` — `[B, 1]`, the current token for each sequence.
    /// Extends `cache.self_k[i]` / `cache.self_v[i]` in place.
    /// Returns logits `[B, vocab_size]`.
    ///
    /// Numerical output is identical to `T5ForConditionalGeneration::decode`
    /// for the same step (verified in Phase 2a).
    pub fn decode_step(
        &self,
        input_ids: &Tensor,      // [B, 1]
        cache: &mut DecoderKvCache,
    ) -> Result<Tensor, TranslatorError> {
        let (b, _q_len) = input_ids.dims2().map_err(cerr)?;

        // Embed current tokens: [B, d_model] → [B, 1, d_model]
        let flat = input_ids.flatten_all().map_err(cerr)?;
        let mut hidden = self
            .embed_tokens
            .index_select(&flat, 0)
            .map_err(cerr)?
            .unsqueeze(1)
            .map_err(cerr)?;

        // Query position index = tokens already in cache (before this step).
        let step = cache.self_k[0].dim(2).map_err(cerr)? as u32;
        let new_kv_len = step + 1;

        // T5 relative position bias: [1, n_heads, 1, new_kv_len].
        // Matches T5Attention::forward use_cache=true branch: q_start=step, q_end=step+1.
        let pos_bias = self.relative_position_bias(step, new_kv_len)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // ── Self-attention (pre-norm + residual) ──────────────────────────
            let normed = t5_rms_norm(&hidden, &layer.sa_norm, self.layer_norm_eps)
                .map_err(cerr)?;

            let q = layer
                .sa_q
                .forward(&normed)
                .map_err(cerr)?
                .reshape((b, 1, self.n_heads, self.d_kv))
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?
                .contiguous()
                .map_err(cerr)?; // [B, n_heads, 1, d_kv]

            let k_new = layer
                .sa_k
                .forward(&normed)
                .map_err(cerr)?
                .reshape((b, 1, self.n_heads, self.d_kv))
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?; // [B, n_heads, 1, d_kv]

            let v_new = layer
                .sa_v
                .forward(&normed)
                .map_err(cerr)?
                .reshape((b, 1, self.n_heads, self.d_kv))
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?; // [B, n_heads, 1, d_kv]

            // Grow KV cache: [B, n_heads, t, d_kv] → [B, n_heads, t+1, d_kv]
            let k = Tensor::cat(&[&cache.self_k[i], &k_new], 2)
                .map_err(cerr)?
                .contiguous()
                .map_err(cerr)?;
            let v = Tensor::cat(&[&cache.self_v[i], &v_new], 2)
                .map_err(cerr)?
                .contiguous()
                .map_err(cerr)?;
            cache.self_k[i] = k.clone();
            cache.self_v[i] = v.clone();

            // Q @ K^T: [B, n_heads, 1, new_kv_len] + position bias → softmax → @ V
            let scores = q
                .matmul(&k.t().map_err(cerr)?)
                .map_err(cerr)?
                .broadcast_add(&pos_bias)
                .map_err(cerr)?;
            let attn_w = candle_nn::ops::softmax_last_dim(&scores).map_err(cerr)?;
            let sa_out = attn_w
                .matmul(&v)
                .map_err(cerr)? // [B, n_heads, 1, d_kv]
                .transpose(1, 2)
                .map_err(cerr)? // [B, 1, n_heads, d_kv]
                .reshape((b, 1, self.inner_dim))
                .map_err(cerr)?;
            let sa_out = layer.sa_o.forward(&sa_out).map_err(cerr)?;
            hidden = (hidden + sa_out).map_err(cerr)?;

            // ── Cross-attention (precomputed K/V, pre-norm + residual) ─────────
            let normed = t5_rms_norm(&hidden, &layer.ca_norm, self.layer_norm_eps)
                .map_err(cerr)?;

            let q = layer
                .ca_q
                .forward(&normed)
                .map_err(cerr)?
                .reshape((b, 1, self.n_heads, self.d_kv))
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?
                .contiguous()
                .map_err(cerr)?;

            let scores = q
                .matmul(&cache.cross_k[i].t().map_err(cerr)?)
                .map_err(cerr)?;
            let attn_w = candle_nn::ops::softmax_last_dim(&scores).map_err(cerr)?;
            let ca_out = attn_w
                .matmul(&cache.cross_v[i])
                .map_err(cerr)?
                .transpose(1, 2)
                .map_err(cerr)?
                .reshape((b, 1, self.inner_dim))
                .map_err(cerr)?;
            let ca_out = layer.ca_o.forward(&ca_out).map_err(cerr)?;
            hidden = (hidden + ca_out).map_err(cerr)?;

            // ── Feed-forward (pre-norm + residual) ───────────────────────────
            let normed = t5_rms_norm(&hidden, &layer.ff_norm, self.layer_norm_eps)
                .map_err(cerr)?;

            let ff_out = if layer.ff_gated {
                // T5v1.1 gated: act(wi_0(x)) * wi_1(x) → wo
                let gate = layer
                    .ff_act
                    .forward(&layer.ff_wi_0.as_ref().unwrap().forward(&normed).map_err(cerr)?)
                    .map_err(cerr)?;
                let lin = layer
                    .ff_wi_1
                    .as_ref()
                    .unwrap()
                    .forward(&normed)
                    .map_err(cerr)?;
                layer.ff_wo.forward(&gate.broadcast_mul(&lin).map_err(cerr)?).map_err(cerr)?
            } else {
                // T5v1.0: act(wi(x)) → wo
                let h = layer
                    .ff_act
                    .forward(&layer.ff_wi.as_ref().unwrap().forward(&normed).map_err(cerr)?)
                    .map_err(cerr)?;
                layer.ff_wo.forward(&h).map_err(cerr)?
            };
            hidden = (hidden + ff_out).map_err(cerr)?;
        }

        // Final layer norm → [B, d_model]
        let hidden = t5_rms_norm(&hidden, &self.final_layer_norm, self.layer_norm_eps)
            .map_err(cerr)?
            .squeeze(1)
            .map_err(cerr)?;

        // LM head — with tied-embedding rescaling if applicable.
        // Mirrors T5ForConditionalGeneration::decode's scaling_factor logic.
        let logits = if self.tie_word_embeddings {
            (hidden * (self.d_model as f64).sqrt())
                .map_err(cerr)?
                .matmul(&self.embed_tokens.t().map_err(cerr)?)
                .map_err(cerr)?
        } else {
            self.lm_head.as_ref().unwrap().forward(&hidden).map_err(cerr)?
        };

        Ok(logits)
    }

    // ── Position bias ─────────────────────────────────────────────────────────

    /// T5 relative position bias for a single query at `q_pos` attending to
    /// `kv_len` keys at positions `0..kv_len`.
    ///
    /// Mirrors `T5Attention::forward` with `use_cache=true`:
    ///   `q_start = kv_len - 1 = q_pos`, `q_end = kv_len`.
    /// Returns `[1, n_heads, 1, kv_len]`.
    fn relative_position_bias(
        &self,
        q_pos: u32,
        kv_len: u32,
    ) -> Result<Tensor, TranslatorError> {
        let num_buckets = self.relative_attention_num_buckets / 2; // 16 for 32-bucket config
        let max_exact = num_buckets / 2;                           // 8
        let max_dist  = self.relative_attention_max_distance;

        // Compute the bucket index for each key position j ∈ 0..kv_len.
        // Exactly matches the candle T5Attention bucket formula.
        let buckets: Vec<u32> = (0..kv_len)
            .map(|j| {
                let i = q_pos;
                if i < j {
                    // Future key (decoder self-attn shouldn't see this, but keep formula intact)
                    if j - i < max_exact {
                        j - i + num_buckets
                    } else {
                        let b = f32::log(
                            (j - i) as f32 / max_exact as f32,
                            max_dist as f32 / max_exact as f32,
                        ) * (num_buckets - max_exact) as f32;
                        u32::min(
                            max_exact + num_buckets + b as u32,
                            self.relative_attention_num_buckets - 1,
                        )
                    }
                } else if i - j < max_exact {
                    i - j // [0, max_exact)
                } else {
                    let b = f32::log(
                        (i - j) as f32 / max_exact as f32,
                        max_dist as f32 / max_exact as f32,
                    ) * (num_buckets - max_exact) as f32;
                    max_exact + b as u32 // [max_exact, num_buckets]
                }
            })
            .collect();

        // index_select on rel_attn_bias [num_buckets, n_heads] with 1-D U32 idx [kv_len]
        // → [kv_len, n_heads] → unsqueeze(0) → [1, kv_len, n_heads]
        // → permute(2,0,1) → [n_heads, 1, kv_len] → unsqueeze(0) → [1, n_heads, 1, kv_len]
        let idx = Tensor::new(buckets.as_slice(), &self.device).map_err(cerr)?;
        self.rel_attn_bias
            .index_select(&idx, 0)
            .map_err(cerr)? // [kv_len, n_heads]
            .unsqueeze(0)
            .map_err(cerr)? // [1, kv_len, n_heads]
            .permute((2, 0, 1))
            .map_err(cerr)? // [n_heads, 1, kv_len]
            .unsqueeze(0)
            .map_err(cerr) // [1, n_heads, 1, kv_len]
    }
}
