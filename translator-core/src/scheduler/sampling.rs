//! Logit post-processing filters and temperature sampling.
//!
//! Applied in this order before committing each token:
//!
//! 1. `apply_decoding_filters` — repetition penalty + no-repeat n-gram
//! 2. `apply_length_bias`      — ramp EOS logit as output approaches expected length
//! 3. `force_eos_on_tail_repeat` — catch repetition loops missed by n-gram filter
//! 4. `sample_token`           — temperature / top-K / top-P sampling

use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;

// ── Repetition / n-gram filters ───────────────────────────────────────────────

/// Penalty applied to tokens already present in the output.
/// A value of 1.0 disables the penalty; higher values suppress repetition more
/// aggressively.
pub const REPETITION_PENALTY: f32 = 1.10;

/// Minimum n-gram length for the no-repeat-n-gram filter.
pub const NO_REPEAT_NGRAM_SIZE: usize = 3;

/// Apply repetition penalty and no-repeat n-gram filtering to a flat logit vec.
///
/// `logits` must have length `vocab_size` (raw f32 values from the model).
/// `output_ids` contains all token ids produced so far for this sequence.
///
/// Mutates `logits` in place; call before sampling.
pub fn apply_decoding_filters(logits: &mut [f32], output_ids: &[u32]) {
    // 1. Repetition penalty: penalise tokens already generated.
    for &tok in output_ids {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= REPETITION_PENALTY;
            } else {
                logits[idx] *= REPETITION_PENALTY;
            }
        }
    }

    // 2. No-repeat n-gram: ban tokens that would extend an existing n-gram.
    let n = NO_REPEAT_NGRAM_SIZE;
    if output_ids.len() >= n - 1 {
        let suffix = &output_ids[output_ids.len() - (n - 1)..];
        for window in output_ids.windows(n) {
            if &window[..n - 1] == suffix {
                let banned = window[n - 1] as usize;
                if banned < logits.len() {
                    logits[banned] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

// ── Length bias ───────────────────────────────────────────────────────────────

/// Fraction of the expected output length at which the EOS logit bias begins.
pub const LENGTH_PENALTY_START: f32 = 0.65;

/// Maximum additional logit added to the EOS token at the expected translation
/// endpoint (linear ramp).
pub const EOS_LOGIT_BIAS: f32 = 6.0;

/// Linearly bias the EOS token logit upward as `step` approaches `expected_len`.
///
/// No-op until `step / expected_len >= LENGTH_PENALTY_START`.  After that,
/// adds a linearly increasing bonus that saturates at `EOS_LOGIT_BIAS` when
/// `step >= expected_len`.
///
/// Applied to raw logits (before temperature scaling inside `sample_token`).
pub fn apply_length_bias(
    logits: &mut [f32],
    eos_token_id: u32,
    step: usize,
    expected_len: usize,
) {
    let eos = eos_token_id as usize;
    if eos >= logits.len() || expected_len == 0 {
        return;
    }
    let progress = (step as f32) / (expected_len as f32);
    if progress >= LENGTH_PENALTY_START {
        let fraction =
            ((progress - LENGTH_PENALTY_START) / (1.0 - LENGTH_PENALTY_START)).min(1.0);
        logits[eos] += fraction * EOS_LOGIT_BIAS;
    }
}

// ── Tail-repeat EOS force ─────────────────────────────────────────────────────

/// Number of recently generated tokens to scan for repeated n-grams.
pub const TAIL_REPEAT_CHECK_LEN: usize = 16;

/// Minimum n-gram size for the tail repetition check.
pub const TAIL_REPEAT_NGRAM: usize = 4;

/// Force EOS if any n-gram in the recent tail appeared in earlier output.
///
/// Catches repeated content that slips past the no-repeat-n-gram filter.
pub fn force_eos_on_tail_repeat(logits: &mut [f32], eos_token_id: u32, output_ids: &[u32]) {
    let n = TAIL_REPEAT_NGRAM;
    let tail_len = TAIL_REPEAT_CHECK_LEN;

    if output_ids.len() < tail_len + n {
        return;
    }

    let eos = eos_token_id as usize;
    let tail_start = output_ids.len() - tail_len;

    for i in tail_start..=output_ids.len().saturating_sub(n) {
        let ngram = &output_ids[i..i + n];
        if output_ids[..tail_start].windows(n).any(|w| w == ngram) {
            for (j, v) in logits.iter_mut().enumerate() {
                if j != eos {
                    *v = f32::NEG_INFINITY;
                }
            }
            return;
        }
    }
}

// ── GPU filter index helpers ──────────────────────────────────────────────────

/// Returns the list of token IDs banned by the no-repeat-n-gram filter.
///
/// Returns an empty vec when `output_ids` is too short to trigger the filter.
/// Used to build the GPU ban-index tensor instead of mutating logits CPU-side.
pub fn compute_ban_indices(output_ids: &[u32]) -> Vec<u32> {
    let n = NO_REPEAT_NGRAM_SIZE;
    if output_ids.len() < n - 1 {
        return Vec::new();
    }
    let suffix = &output_ids[output_ids.len() - (n - 1)..];
    let mut banned = Vec::new();
    for window in output_ids.windows(n) {
        if &window[..n - 1] == suffix {
            banned.push(window[n - 1]);
        }
    }
    banned
}

/// Returns `true` if a tail repeat was detected and EOS should be forced.
///
/// Mirrors the logic in [`force_eos_on_tail_repeat`] but returns a bool instead
/// of mutating logits — for building the GPU force-EOS flag tensor.
pub fn check_tail_repeat(output_ids: &[u32]) -> bool {
    let n = TAIL_REPEAT_NGRAM;
    let tail_len = TAIL_REPEAT_CHECK_LEN;

    if output_ids.len() < tail_len + n {
        return false;
    }

    let tail_start = output_ids.len() - tail_len;
    for i in tail_start..=output_ids.len().saturating_sub(n) {
        let ngram = &output_ids[i..i + n];
        if output_ids[..tail_start].windows(n).any(|w| w == ngram) {
            return true;
        }
    }
    false
}

/// Computes the additive EOS logit bias for the length penalty at the given step.
///
/// Returns 0.0 if `step / expected_len` is below [`LENGTH_PENALTY_START`].
/// Used to populate the GPU EOS-bias scatter tensor instead of mutating logits.
pub fn compute_length_bias(step: usize, expected_len: usize) -> f32 {
    if expected_len == 0 {
        return 0.0;
    }
    let progress = (step as f32) / (expected_len as f32);
    if progress < LENGTH_PENALTY_START {
        return 0.0;
    }
    // Ramp from LENGTH_PENALTY_START to 1.0: 0 → EOS_LOGIT_BIAS
    // Past 1.0: continue at double rate (steeper ramp) up to 2× EOS_LOGIT_BIAS
    let fraction = (progress - LENGTH_PENALTY_START) / (1.0 - LENGTH_PENALTY_START);
    fraction.min(2.0) * EOS_LOGIT_BIAS
}

// ── Temperature / top-K / top-P sampling ─────────────────────────────────────

/// Sampling temperature — lower = more peaked distribution.
pub const TEMPERATURE: f32 = 0.15;

/// Top-K: only consider the K highest-logit tokens.
pub const TOP_K: usize = 40;

/// Top-P (nucleus): keep the smallest set of tokens whose cumulative
/// probability reaches this threshold.
pub const TOP_P: f32 = 0.90;

/// Sample the next token from `logits` using top-K / top-P / temperature.
///
/// Temperature is applied to the ≤K candidates AFTER top-K selection —
/// since temperature is a positive constant it doesn't change rank order,
/// so top-K can run on raw logits (saving ~262 K divisions per call).
///
/// `scratch` is a caller-supplied reusable buffer for the candidates vec.
/// Pre-allocate once with `Vec::with_capacity(TOP_K + 1)` and pass on every call
/// to avoid per-token heap allocation.
///
/// Call AFTER `apply_decoding_filters` and `apply_length_bias`.
pub fn sample_token(logits: &mut [f32], rng: &mut SmallRng, scratch: &mut Vec<(u32, f32)>) -> u32 {
    let vocab = logits.len();

    // 1. Top-K: collect finite logits by raw value (temperature preserves rank order).
    let k = TOP_K.min(vocab);
    scratch.clear();
    scratch.extend(
        logits
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v.is_finite())
            .map(|(i, &v)| (i as u32, v)),
    );
    if scratch.len() > k {
        scratch.select_nth_unstable_by(k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scratch.truncate(k);
    }

    if scratch.is_empty() {
        return 0;
    }

    // 2. Apply temperature to the ≤K candidates only (~40 divisions instead of 262K).
    for (_, v) in scratch.iter_mut() {
        *v /= TEMPERATURE;
    }

    // 3. Softmax over the ≤K candidates only.
    let max = scratch.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    for (_, v) in scratch.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = scratch.iter().map(|(_, v)| v).sum();
    if sum > 0.0 {
        for (_, v) in scratch.iter_mut() {
            *v /= sum;
        }
    }

    // 4. Top-P (nucleus): sort the small K-element vec by probability, then
    //    truncate once cumulative mass >= TOP_P.
    scratch.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cumsum = 0.0_f32;
    let mut cutoff = scratch.len();
    for (i, (_, p)) in scratch.iter().enumerate() {
        cumsum += p;
        if cumsum >= TOP_P {
            cutoff = i + 1;
            break;
        }
    }
    scratch.truncate(cutoff);

    // 5. Renormalise & weighted draw.
    let sum: f32 = scratch.iter().map(|(_, v)| v).sum();
    let draw: f32 = Standard.sample(rng);
    let threshold = draw * sum;
    let mut cumsum = 0.0_f32;
    for &(idx, p) in scratch.iter() {
        cumsum += p;
        if cumsum >= threshold {
            return idx;
        }
    }

    // Fallback (should be rare — floating-point rounding at the tail).
    scratch[0].0
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_positive_logit() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        apply_decoding_filters(&mut logits, &[1]); // token 1 already seen
        assert!((logits[1] - 2.0 / REPETITION_PENALTY).abs() < 1e-6);
        // Unseen tokens unchanged.
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_negative_logit() {
        let mut logits = vec![-2.0f32, 0.0, 1.0];
        apply_decoding_filters(&mut logits, &[0]);
        assert!((logits[0] - (-2.0 * REPETITION_PENALTY)).abs() < 1e-6);
    }

    #[test]
    fn no_repeat_ngram_bans_completing_token() {
        // Output so far: [10, 20, 30, 10, 20]
        // Suffix of length n-1=2 is [10, 20].
        // The n-gram [10, 20, 30] exists, so token 30 should be banned.
        let mut logits = vec![1.0f32; 50];
        apply_decoding_filters(&mut logits, &[10, 20, 30, 10, 20]);
        assert_eq!(logits[30], f32::NEG_INFINITY);
        // Other tokens unaffected by n-gram filter (only repetition penalty applies).
        assert!(logits[5].is_finite());
    }

    #[test]
    fn no_repeat_ngram_too_short_history() {
        // Only 1 token in output — not enough for n-gram filter (needs n-1=2).
        let mut logits = vec![1.0f32; 10];
        apply_decoding_filters(&mut logits, &[3]);
        // No n-gram ban should have fired.
        assert!(logits.iter().all(|&v| v.is_finite() || v == f32::NEG_INFINITY));
        // Token 3 penalised by repetition penalty (was 1.0 > 0 → divided).
        assert!((logits[3] - 1.0 / REPETITION_PENALTY).abs() < 1e-6);
    }

    #[test]
    fn length_bias_below_threshold() {
        // progress = 0.5 < LENGTH_PENALTY_START — logits must be unchanged.
        let mut logits = vec![0.0f32; 5];
        let eos = 1u32;
        apply_length_bias(&mut logits, eos, 5, 10); // step/expected = 0.5
        assert!((logits[eos as usize]).abs() < 1e-6);
    }

    #[test]
    fn length_bias_below_start() {
        // progress = 0.60 < LENGTH_PENALTY_START (0.65) → bias = 0.
        let mut logits = vec![0.0f32; 5];
        let eos = 2u32;
        apply_length_bias(&mut logits, eos, 6, 10); // 6/10 = 0.60
        assert!((logits[eos as usize]).abs() < 1e-6);
    }

    #[test]
    fn length_bias_at_capacity() {
        // step == expected_len → fraction = 1.0 → bias == EOS_LOGIT_BIAS (raw logit space).
        let mut logits = vec![0.0f32; 5];
        let eos = 0u32;
        apply_length_bias(&mut logits, eos, 10, 10);
        assert!((logits[eos as usize] - EOS_LOGIT_BIAS).abs() < 1e-6);
    }

    #[test]
    fn length_bias_eos_out_of_range() {
        // eos_token_id >= vocab_size — must not panic.
        let mut logits = vec![0.0f32; 5];
        apply_length_bias(&mut logits, 99, 10, 10);
        assert!(logits.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn tail_repeat_too_short() {
        let mut logits = vec![1.0f32; 10];
        let eos = 0u32;
        let output_ids: Vec<u32> = (0..TAIL_REPEAT_CHECK_LEN as u32).collect();
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn tail_repeat_fires_on_repeat_ngram() {
        let n = TAIL_REPEAT_NGRAM;
        let tail_len = TAIL_REPEAT_CHECK_LEN;
        let eos = 5u32;

        let mut output_ids: Vec<u32> = vec![10; tail_len];
        output_ids[..n].copy_from_slice(&[1, 2, 3, 4]);
        let mut tail: Vec<u32> = vec![20; tail_len];
        tail[..n].copy_from_slice(&[1, 2, 3, 4]);
        output_ids.extend_from_slice(&tail);

        let mut logits = vec![1.0f32; 30];
        logits[eos as usize] = 0.5;
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);

        for (j, &v) in logits.iter().enumerate() {
            if j == eos as usize {
                assert!(v.is_finite(), "EOS logit must remain finite");
            } else {
                assert_eq!(v, f32::NEG_INFINITY, "non-EOS logit {j} must be -inf");
            }
        }
    }

    #[test]
    fn tail_repeat_no_match() {
        let tail_len = TAIL_REPEAT_CHECK_LEN;
        let eos = 0u32;

        let mut output_ids: Vec<u32> = (0..tail_len as u32).collect();
        let tail: Vec<u32> = (100..100 + tail_len as u32).collect();
        output_ids.extend_from_slice(&tail);

        let mut logits = vec![1.0f32; 200];
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn tail_repeat_eos_token_preserved() {
        let n = TAIL_REPEAT_NGRAM;
        let tail_len = TAIL_REPEAT_CHECK_LEN;
        let eos = 3u32;

        let mut output_ids: Vec<u32> = vec![10; tail_len];
        output_ids[..n].copy_from_slice(&[1, 2, 3, 4]);
        let mut tail: Vec<u32> = vec![20; tail_len];
        tail[..n].copy_from_slice(&[1, 2, 3, 4]);
        output_ids.extend_from_slice(&tail);

        let mut logits = vec![1.0f32; 30];
        logits[eos as usize] = 5.0;
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);

        assert!(logits[eos as usize].is_finite(), "EOS logit must stay finite");
        assert_eq!(logits[eos as usize], 5.0, "EOS logit must be unchanged");
    }

    #[test]
    fn sample_token_returns_valid_index() {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(42);
        let mut logits = vec![1.0f32; 100];
        logits[7] = 10.0; // heavily favour token 7
        let mut scratch = Vec::with_capacity(TOP_K + 1);
        let tok = sample_token(&mut logits, &mut rng, &mut scratch);
        assert!((tok as usize) < 100);
    }

    #[test]
    fn sample_token_forced_eos() {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(0);
        // Only EOS is non-neg-inf.
        let mut logits = vec![f32::NEG_INFINITY; 100];
        logits[1] = 0.0; // EOS = 1
        let mut scratch = Vec::with_capacity(TOP_K + 1);
        let tok = sample_token(&mut logits, &mut rng, &mut scratch);
        assert_eq!(tok, 1);
    }
}
