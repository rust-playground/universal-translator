use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::{s, Array2, Array3, Array4};
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use crate::error::TranslatorError;

const MAX_INPUT_TOKENS: usize = 512;
const MAX_NEW_TOKENS: usize = 1024;
/// Standard T5 length-normalization exponent for beam scoring.
const BEAM_LENGTH_PENALTY: f32 = 0.6;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct T5Config {
    num_layers: usize,
    eos_token_id: u32,
    #[serde(default)]
    decoder_start_token_id: Option<u32>,
}

// ─── Execution Providers ─────────────────────────────────────────────────────

#[allow(unreachable_code)]
fn execution_providers(_model_dir: &Path) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(feature = "cuda")]
    return vec![
        ort::ep::TensorRT::default().build(),
        ort::ep::CUDA::default().build(),
    ];

    #[cfg(feature = "coreml")]
    {
        use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
        let cache_dir = _model_dir.join("coreml-cache");
        return vec![
            ort::ep::CoreML::default()
                .with_compute_units(ComputeUnits::All)
                .with_model_format(ModelFormat::MLProgram)
                .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                .with_model_cache_dir(cache_dir.to_string_lossy())
                .build(),
        ];
    }

    vec![]
}

// ─── Session builder ─────────────────────────────────────────────────────────

fn build_session(path: &Path) -> Result<Session, TranslatorError> {
    let model_dir = path.parent().unwrap_or(path);
    Session::builder()
        .map_err(|e| TranslatorError::Model(format!("session builder: {e}")))?
        .with_log_level(ort::logging::LogLevel::Error)
        .map_err(|e| TranslatorError::Model(format!("log level: {e}")))?
        .with_execution_providers(execution_providers(model_dir))
        .map_err(|e| TranslatorError::Model(format!("execution providers: {e}")))?
        .commit_from_file(path)
        .map_err(|e| TranslatorError::Model(format!("load {}: {e}", path.display())))
}

// ─── Output extraction helpers ───────────────────────────────────────────────

fn extract_kv4(outputs: &SessionOutputs<'_>, name: &str) -> Result<Array4<f32>, TranslatorError> {
    outputs[name]
        .try_extract_array::<f32>()
        .map_err(|e| TranslatorError::Model(format!("extract '{name}': {e}")))?
        .into_dimensionality::<ndarray::Ix4>()
        .map(|v| v.to_owned())
        .map_err(|e| TranslatorError::Model(format!("reshape '{name}': {e}")))
}

fn extract_logits3(outputs: &SessionOutputs<'_>) -> Result<Array3<f32>, TranslatorError> {
    outputs["logits"]
        .try_extract_array::<f32>()
        .map_err(|e| TranslatorError::Model(format!("extract logits: {e}")))?
        .into_dimensionality::<ndarray::Ix3>()
        .map(|v| v.to_owned())
        .map_err(|e| TranslatorError::Model(format!("logits shape: {e}")))
}

// ─── Numerics ────────────────────────────────────────────────────────────────

/// Greedy argmax over vocab dim of a `[B, 1, vocab]` logits tensor.
fn argmax3(logits: &Array3<f32>) -> Vec<u32> {
    let b = logits.shape()[0];
    (0..b)
        .map(|i| {
            logits
                .slice(s![i, 0, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        })
        .collect()
}

/// Numerically stable log-softmax over an iterator of logit values.
fn log_softmax(logits: impl Iterator<Item = f32> + Clone) -> Vec<f32> {
    let max_l = logits.clone().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = max_l + logits.clone().map(|x| (x - max_l).exp()).sum::<f32>().ln();
    logits.map(|x| x - log_sum_exp).collect()
}

// ─── KV Cache ────────────────────────────────────────────────────────────────

/// KV cache for one sequence (batched slot) or one beam.
///
/// Decoder self-attention KV (`dec_keys`/`dec_vals`) grows by one position at
/// each step — ORT's `decoder_with_past_model.onnx` concatenates internally and
/// emits the full accumulated sequence in `present.*`, so we replace
/// the stored tensor with the output each step.
///
/// Encoder cross-attention KV (`enc_keys`/`enc_vals`) is constant once computed
/// in the first decoder step and is wrapped in `Arc` so beam-search clones share
/// the same allocation without copying.
#[derive(Clone)]
struct KvCache {
    dec_keys: Vec<Array4<f32>>,
    dec_vals: Vec<Array4<f32>>,
    enc_keys: Arc<Vec<Array4<f32>>>,
    enc_vals: Arc<Vec<Array4<f32>>>,
}

impl KvCache {
    /// Extract all KV tensors from the first decoder step's outputs.
    fn from_init_outputs(
        outputs: &SessionOutputs<'_>,
        num_layers: usize,
    ) -> Result<Self, TranslatorError> {
        let mut dec_keys = Vec::with_capacity(num_layers);
        let mut dec_vals = Vec::with_capacity(num_layers);
        let mut enc_keys = Vec::with_capacity(num_layers);
        let mut enc_vals = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            dec_keys.push(extract_kv4(outputs, &format!("present.{layer}.decoder.key"))?);
            dec_vals.push(extract_kv4(outputs, &format!("present.{layer}.decoder.value"))?);
            enc_keys.push(extract_kv4(outputs, &format!("present.{layer}.encoder.key"))?);
            enc_vals.push(extract_kv4(outputs, &format!("present.{layer}.encoder.value"))?);
        }
        Ok(Self {
            dec_keys,
            dec_vals,
            enc_keys: Arc::new(enc_keys),
            enc_vals: Arc::new(enc_vals),
        })
    }
}

// ─── LoadedModel ─────────────────────────────────────────────────────────────

/// A loaded MADLAD-400-3B-MT model backed by three ONNX Runtime sessions.
///
/// Required files in the model directory:
///   `encoder_model.onnx`           — input_ids + attention_mask → encoder hidden states
///   `decoder_model.onnx`           — first decode step (no past KV inputs)
///   `decoder_with_past_model.onnx` — subsequent steps (96 past_key_value inputs)
///   `config.json`                  — T5 model config
///   `tokenizer.json`               — HuggingFace fast tokenizer
///
/// ORT sessions are thread-safe; the `Mutex` wrappers satisfy Rust's `&mut self`
/// requirement for `Session::run` while allowing `LoadedModel` to be `Send + Sync`.
/// Since inference is serialised through a single background worker there is no
/// actual lock contention in practice.
pub struct LoadedModel {
    encoder_session:      Mutex<Session>,
    decoder_init_session: Mutex<Session>,
    decoder_step_session: Mutex<Session>,
    tokenizer:            Tokenizer,
    num_layers:           usize,
    eos_token_id:         u32,
    decoder_start_token_id: u32,
}

impl LoadedModel {
    pub fn load(model_dir: &Path, _num_threads: usize) -> Result<Self, TranslatorError> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(TranslatorError::Io)?;
        let config: T5Config = serde_json::from_str(&config_str)
            .map_err(|e| TranslatorError::Model(format!("config parse: {e}")))?;

        let encoder_path = model_dir.join("encoder_model.onnx");
        if !encoder_path.exists() {
            return Err(TranslatorError::ModelNotFound(format!(
                "{} not found — run models/download.sh",
                encoder_path.display()
            )));
        }

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| TranslatorError::Model(format!("tokenizer load: {e}")))?;

        tracing::info!(
            num_layers = config.num_layers,
            "MADLAD-400 ORT model loaded from {}",
            model_dir.display()
        );

        Ok(Self {
            encoder_session:      Mutex::new(build_session(&encoder_path)?),
            decoder_init_session: Mutex::new(build_session(&model_dir.join("decoder_model.onnx"))?),
            decoder_step_session: Mutex::new(build_session(&model_dir.join("decoder_with_past_model.onnx"))?),
            tokenizer,
            num_layers:             config.num_layers,
            eos_token_id:           config.eos_token_id,
            decoder_start_token_id: config.decoder_start_token_id.unwrap_or(0),
        })
    }

    /// Translate a batch of strings.  Always call from `spawn_blocking`.
    pub fn translate_batch(
        &self,
        texts: &[String],
        beam_width: u8,
    ) -> Result<Vec<String>, TranslatorError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if beam_width <= 1 {
            self.translate_greedy_batched(texts)
        } else {
            texts.iter().map(|t| self.translate_beam(t, beam_width as usize)).collect()
        }
    }

    // ─── Greedy batched decode ────────────────────────────────────────────────

    fn translate_greedy_batched(&self, texts: &[String]) -> Result<Vec<String>, TranslatorError> {
        let b = texts.len();

        // ── Tokenize & pad ──────────────────────────────────────────────────
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(t.as_str(), true)
                    .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))
            })
            .collect::<Result<_, _>>()?;

        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_INPUT_TOKENS);

        if seq_len == 0 {
            return Ok(vec![String::new(); b]);
        }

        let mut ids_flat  = vec![0i64; b * seq_len];
        let mut mask_flat = vec![0i64; b * seq_len];
        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().take(seq_len).enumerate() {
                ids_flat[i * seq_len + j]  = id as i64;
                mask_flat[i * seq_len + j] = 1;
            }
        }
        let input_ids = Array2::from_shape_vec((b, seq_len), ids_flat)
            .map_err(|e| TranslatorError::Model(format!("input_ids shape: {e}")))?;
        let attn_mask = Array2::from_shape_vec((b, seq_len), mask_flat)
            .map_err(|e| TranslatorError::Model(format!("attn_mask shape: {e}")))?;

        // ── Encode ──────────────────────────────────────────────────────────
        let enc_hidden: Array3<f32> = {
            let enc_inputs = ort::inputs![
                "input_ids"      => TensorRef::from_array_view(&input_ids)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "attention_mask" => TensorRef::from_array_view(&attn_mask)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
            ];
            let mut guard = self.encoder_session.lock()
                .map_err(|_| TranslatorError::Model("encoder lock poisoned".into()))?;
            let outputs = guard.run(enc_inputs)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| TranslatorError::Model(format!("enc hidden: {e}")))?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TranslatorError::Model(format!("enc hidden shape: {e}")))?
                .to_owned()
        };

        // ── First decoder step ───────────────────────────────────────────────
        let decoder_start = Array2::<i64>::from_elem((b, 1), self.decoder_start_token_id as i64);
        let (first_tokens, mut kv) = {
            let init_inputs = ort::inputs![
                "input_ids"              => TensorRef::from_array_view(&decoder_start)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "encoder_hidden_states"  => TensorRef::from_array_view(&enc_hidden)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "encoder_attention_mask" => TensorRef::from_array_view(&attn_mask)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
            ];
            let mut guard = self.decoder_init_session.lock()
                .map_err(|_| TranslatorError::Model("decoder_init lock poisoned".into()))?;
            let outputs = guard.run(init_inputs)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            let logits = extract_logits3(&outputs)?;
            let kv = KvCache::from_init_outputs(&outputs, self.num_layers)?;
            (argmax3(&logits), kv)
        };

        let mut output_ids: Vec<Vec<u32>> = vec![vec![]; b];
        let mut finished = vec![false; b];
        let mut current_tokens = first_tokens;

        for (i, &tok) in current_tokens.iter().enumerate() {
            if tok == self.eos_token_id { finished[i] = true; }
            else { output_ids[i].push(tok); }
        }

        // ── Subsequent decode steps ──────────────────────────────────────────
        for _ in 1..MAX_NEW_TOKENS {
            if finished.iter().all(|&f| f) { break; }

            // Feed EOS for finished sequences to keep the KV cache consistent.
            let dec_input = Array2::<i64>::from_shape_fn((b, 1), |(i, _)| {
                if finished[i] { self.eos_token_id as i64 } else { current_tokens[i] as i64 }
            });

            // Build the 2 + 4×layers named inputs.
            // Note: decoder_with_past_model takes encoder_attention_mask but NOT
            // encoder_hidden_states — the hidden states are already in the KV cache.
            let mut step_inputs = ort::inputs![
                "input_ids"              => TensorRef::from_array_view(&dec_input)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "encoder_attention_mask" => TensorRef::from_array_view(&attn_mask)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
            ];
            for layer in 0..self.num_layers {
                step_inputs.push((
                    Cow::Owned(format!("past_key_values.{layer}.decoder.key")),
                    SessionInputValue::from(
                        TensorRef::from_array_view(&kv.dec_keys[layer])
                            .map_err(|e| TranslatorError::Model(e.to_string()))?,
                    ),
                ));
                step_inputs.push((
                    Cow::Owned(format!("past_key_values.{layer}.decoder.value")),
                    SessionInputValue::from(
                        TensorRef::from_array_view(&kv.dec_vals[layer])
                            .map_err(|e| TranslatorError::Model(e.to_string()))?,
                    ),
                ));
                step_inputs.push((
                    Cow::Owned(format!("past_key_values.{layer}.encoder.key")),
                    SessionInputValue::from(
                        TensorRef::from_array_view(&kv.enc_keys[layer])
                            .map_err(|e| TranslatorError::Model(e.to_string()))?,
                    ),
                ));
                step_inputs.push((
                    Cow::Owned(format!("past_key_values.{layer}.encoder.value")),
                    SessionInputValue::from(
                        TensorRef::from_array_view(&kv.enc_vals[layer])
                            .map_err(|e| TranslatorError::Model(e.to_string()))?,
                    ),
                ));
            }

            // Run and extract all outputs while holding the lock.
            let (logits, new_dec_keys, new_dec_vals) = {
                let mut guard = self.decoder_step_session.lock()
                    .map_err(|_| TranslatorError::Model("decoder_step lock poisoned".into()))?;
                let outputs = guard.run(step_inputs)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?;
                let logits = extract_logits3(&outputs)?;
                let mut dk = Vec::with_capacity(self.num_layers);
                let mut dv = Vec::with_capacity(self.num_layers);
                for layer in 0..self.num_layers {
                    dk.push(extract_kv4(
                        &outputs,
                        &format!("present.{layer}.decoder.key"),
                    )?);
                    dv.push(extract_kv4(
                        &outputs,
                        &format!("present.{layer}.decoder.value"),
                    )?);
                }
                (logits, dk, dv)
            };
            // Update decoder KV (encoder KV is constant).
            kv.dec_keys = new_dec_keys;
            kv.dec_vals = new_dec_vals;

            let next_tokens = argmax3(&logits);
            for (i, &tok) in next_tokens.iter().enumerate() {
                if !finished[i] {
                    if tok == self.eos_token_id { finished[i] = true; }
                    else { output_ids[i].push(tok); }
                }
            }
            current_tokens = next_tokens;
        }

        output_ids
            .iter()
            .map(|ids| {
                self.tokenizer
                    .decode(ids, true)
                    .map_err(|e| TranslatorError::Model(format!("decode: {e}")))
            })
            .collect()
    }

    // ─── Beam search ─────────────────────────────────────────────────────────

    /// Beam search decode for a single text.
    ///
    /// Each beam holds an independent `KvCache`; the encoder cross-attention KV
    /// is shared across beams via `Arc` (computed once, never modified).
    fn translate_beam(&self, text: &str, beam_width: usize) -> Result<String, TranslatorError> {
        // ── Tokenize ────────────────────────────────────────────────────────
        let enc = self.tokenizer
            .encode(text, true)
            .map_err(|e| TranslatorError::Model(format!("tokenize: {e}")))?;
        let ids_vec: Vec<i64> = enc
            .get_ids()
            .iter()
            .take(MAX_INPUT_TOKENS)
            .map(|&id| id as i64)
            .collect();
        if ids_vec.is_empty() {
            return Ok(String::new());
        }
        let seq_len = ids_vec.len();
        let input_ids = Array2::from_shape_vec((1, seq_len), ids_vec)
            .map_err(|e| TranslatorError::Model(format!("input_ids shape: {e}")))?;
        let attn_mask = Array2::<i64>::ones((1, seq_len));

        // ── Encode (B=1, shared by all beams) ──────────────────────────────
        let enc_hidden: Array3<f32> = {
            let enc_inputs = ort::inputs![
                "input_ids"      => TensorRef::from_array_view(&input_ids)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "attention_mask" => TensorRef::from_array_view(&attn_mask)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
            ];
            let mut guard = self.encoder_session.lock()
                .map_err(|_| TranslatorError::Model("encoder lock poisoned".into()))?;
            let outputs = guard.run(enc_inputs)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| TranslatorError::Model(format!("enc hidden: {e}")))?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TranslatorError::Model(format!("enc hidden shape: {e}")))?
                .to_owned()
        };

        // ── First decoder step — seeds the beam KV caches ───────────────────
        let decoder_start = Array2::<i64>::from_elem((1, 1), self.decoder_start_token_id as i64);
        let (first_logits, seed_kv) = {
            let init_inputs = ort::inputs![
                "input_ids"              => TensorRef::from_array_view(&decoder_start)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "encoder_hidden_states"  => TensorRef::from_array_view(&enc_hidden)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
                "encoder_attention_mask" => TensorRef::from_array_view(&attn_mask)
                    .map_err(|e| TranslatorError::Model(e.to_string()))?,
            ];
            let mut guard = self.decoder_init_session.lock()
                .map_err(|_| TranslatorError::Model("decoder_init lock poisoned".into()))?;
            let outputs = guard.run(init_inputs)
                .map_err(|e| TranslatorError::Model(e.to_string()))?;
            let logits = extract_logits3(&outputs)?;
            let kv = KvCache::from_init_outputs(&outputs, self.num_layers)?;
            (logits, kv)
        };

        // Build initial beams from top-k of the first decode step.
        let row = first_logits.slice(s![0, 0, ..]);
        let first_lps = log_softmax(row.iter().copied());

        struct Beam {
            kv:            KvCache,
            tokens:        Vec<u32>,
            score:         f32,
            current_token: u32,
            finished:      bool,
        }

        let mut top_k: Vec<(usize, f32)> = first_lps.iter().copied().enumerate().collect();
        top_k.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_k.truncate(beam_width);

        let mut beams: Vec<Beam> = top_k
            .into_iter()
            .map(|(token_id, log_p)| {
                let token = token_id as u32;
                let finished = token == self.eos_token_id;
                Beam {
                    kv:            seed_kv.clone(),
                    tokens:        if finished { vec![] } else { vec![token] },
                    score:         log_p,
                    current_token: token,
                    finished,
                }
            })
            .collect();

        let mut completed: Vec<(f32, Vec<u32>)> = vec![];

        for beam in &beams {
            if beam.finished {
                completed.push((beam.score, beam.tokens.clone()));
            }
        }

        // ── Beam search loop ─────────────────────────────────────────────────
        for _ in 1..MAX_NEW_TOKENS {
            beams.retain(|b| !b.finished);
            if beams.is_empty() { break; }

            // Collect top-beam_width candidates from all active beams.
            let mut candidates: Vec<(f32, u32, usize)> = vec![];

            for (bi, beam) in beams.iter_mut().enumerate() {
                let dec_input = Array2::<i64>::from_elem((1, 1), beam.current_token as i64);

                // Build inputs for this beam's KV state.
                // Note: decoder_with_past_model takes encoder_attention_mask but NOT
                // encoder_hidden_states — the hidden states are already in the KV cache.
                let mut step_inputs = ort::inputs![
                    "input_ids"              => TensorRef::from_array_view(&dec_input)
                        .map_err(|e| TranslatorError::Model(e.to_string()))?,
                    "encoder_attention_mask" => TensorRef::from_array_view(&attn_mask)
                        .map_err(|e| TranslatorError::Model(e.to_string()))?,
                ];
                for layer in 0..self.num_layers {
                    step_inputs.push((
                        Cow::Owned(format!("past_key_values.{layer}.decoder.key")),
                        SessionInputValue::from(
                            TensorRef::from_array_view(&beam.kv.dec_keys[layer])
                                .map_err(|e| TranslatorError::Model(e.to_string()))?,
                        ),
                    ));
                    step_inputs.push((
                        Cow::Owned(format!("past_key_values.{layer}.decoder.value")),
                        SessionInputValue::from(
                            TensorRef::from_array_view(&beam.kv.dec_vals[layer])
                                .map_err(|e| TranslatorError::Model(e.to_string()))?,
                        ),
                    ));
                    step_inputs.push((
                        Cow::Owned(format!("past_key_values.{layer}.encoder.key")),
                        SessionInputValue::from(
                            TensorRef::from_array_view(&beam.kv.enc_keys[layer])
                                .map_err(|e| TranslatorError::Model(e.to_string()))?,
                        ),
                    ));
                    step_inputs.push((
                        Cow::Owned(format!("past_key_values.{layer}.encoder.value")),
                        SessionInputValue::from(
                            TensorRef::from_array_view(&beam.kv.enc_vals[layer])
                                .map_err(|e| TranslatorError::Model(e.to_string()))?,
                        ),
                    ));
                }

                // Run one decoder step and update this beam's KV cache.
                let (lps, new_dec_keys, new_dec_vals) = {
                    let mut guard = self.decoder_step_session.lock()
                        .map_err(|_| TranslatorError::Model("decoder_step lock poisoned".into()))?;
                    let outputs = guard.run(step_inputs)
                        .map_err(|e| TranslatorError::Model(e.to_string()))?;
                    let logits = extract_logits3(&outputs)?;
                    let row = logits.slice(s![0, 0, ..]);
                    let lps = log_softmax(row.iter().copied());
                    let mut dk = Vec::with_capacity(self.num_layers);
                    let mut dv = Vec::with_capacity(self.num_layers);
                    for layer in 0..self.num_layers {
                        dk.push(extract_kv4(&outputs, &format!("present.{layer}.decoder.key"))?);
                        dv.push(extract_kv4(&outputs, &format!("present.{layer}.decoder.value"))?);
                    }
                    (lps, dk, dv)
                };
                beam.kv.dec_keys = new_dec_keys;
                beam.kv.dec_vals = new_dec_vals;

                // Collect top-beam_width candidates from this beam's distribution.
                let mut top_k: Vec<(usize, f32)> = lps.iter().copied().enumerate().collect();
                top_k.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (token_id, log_p) in top_k.into_iter().take(beam_width) {
                    candidates.push((beam.score + log_p, token_id as u32, bi));
                }
            }

            // Global top-beam_width selection.
            candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam_width);

            // Snapshot parent KV caches and token histories before rebuilding.
            let parent_kvs:    Vec<KvCache>  = candidates.iter().map(|(_, _, p)| beams[*p].kv.clone()).collect();
            let parent_tokens: Vec<Vec<u32>> = candidates.iter().map(|(_, _, p)| beams[*p].tokens.clone()).collect();

            beams = candidates
                .into_iter()
                .zip(parent_kvs)
                .zip(parent_tokens)
                .map(|(((score, token, _), kv), mut tokens)| {
                    let finished = token == self.eos_token_id;
                    if !finished { tokens.push(token); }
                    Beam { kv, tokens, score, current_token: token, finished }
                })
                .collect();

            for beam in &beams {
                if beam.finished {
                    completed.push((beam.score, beam.tokens.clone()));
                }
            }
        }

        // Add unfinished beams as length-truncated fallbacks.
        for beam in &beams {
            if !beam.finished && !beam.tokens.is_empty() {
                completed.push((beam.score, beam.tokens.clone()));
            }
        }

        let best_tokens = completed
            .into_iter()
            .max_by(|(s_a, t_a), (s_b, t_b)| {
                let norm_a = s_a / (t_a.len().max(1) as f32).powf(BEAM_LENGTH_PENALTY);
                let norm_b = s_b / (t_b.len().max(1) as f32).powf(BEAM_LENGTH_PENALTY);
                norm_a.partial_cmp(&norm_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, tokens)| tokens)
            .unwrap_or_default();

        self.tokenizer
            .decode(&best_tokens, true)
            .map_err(|e| TranslatorError::Model(format!("decode: {e}")))
    }
}
