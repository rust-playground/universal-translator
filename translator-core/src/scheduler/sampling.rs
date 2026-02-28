//! Logit post-processing filters applied before argmax / sampling.
//!
//! Two standard techniques prevent T5 autoregressive repetition loops:
//!
//! 1. **Repetition penalty** — tokens already in the output have their logit
//!    divided (if positive) or multiplied (if negative) by `REPETITION_PENALTY`,
//!    making them less likely to be chosen again.
//!
//! 2. **No-repeat n-gram** — if the last `N-1` tokens of the output already
//!    appear as a prefix of some n-gram in the history, the completing token is
//!    banned (logit set to −∞).

/// Penalty applied to tokens already present in the output.
/// A value of 1.0 disables the penalty; higher values suppress repetition more
/// aggressively.  The recommended range for T5-family models is 1.12–1.18.
pub const REPETITION_PENALTY: f32 = 1.15;

/// Minimum n-gram length for the no-repeat-n-gram filter.
/// Setting this to 3 bans any 3-gram that already appears in the output.
pub const NO_REPEAT_NGRAM_SIZE: usize = 3;

/// Apply repetition penalty and no-repeat n-gram filtering to a flat logit vec.
///
/// `logits` must have length `vocab_size` (raw f32 values from the model).
/// `output_ids` contains all token ids produced so far for this sequence.
///
/// Mutates `logits` in place; call before `argmax` / softmax.
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

/// Fraction of the slot capacity at which the EOS logit bias begins ramping up.
pub const LENGTH_PENALTY_START: f32 = 0.60;

/// Maximum additional logit added to the EOS token at the expected translation
/// endpoint (linear ramp).  At LENGTH_PENALTY_START the bias is 0; at
/// `expected_len` (and beyond) it saturates at EOS_LOGIT_BIAS.
pub const EOS_LOGIT_BIAS: f32 = 6.0;

/// Linearly bias the EOS token logit upward as `step` approaches `expected_len`.
///
/// No-op until `step / expected_len >= LENGTH_PENALTY_START`.  After that, adds
/// a linearly increasing bonus that saturates at `EOS_LOGIT_BIAS` when
/// `step >= expected_len`.  `expected_len` is the predicted natural translation
/// endpoint (e.g. 1.35×seq_len + 10), decoupled from the KV-cache capacity
/// ceiling so the bias peaks at the right time rather than at the hard limit.
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

/// Number of recently generated tokens to scan for repeated n-grams.
pub const TAIL_REPEAT_CHECK_LEN: usize = 16;

/// Minimum n-gram size for the tail repetition check.
pub const TAIL_REPEAT_NGRAM: usize = 4;

/// Force EOS if any n-gram in the recent tail appeared in earlier output.
///
/// Scans the last `TAIL_REPEAT_CHECK_LEN` tokens for any n-gram of size
/// `TAIL_REPEAT_NGRAM` that also exists in the preceding history.  When
/// found, all non-EOS logits are set to −∞, forcing the next token to EOS.
///
/// This catches repeated content that slips past the no-repeat-n-gram filter
/// due to BPE tokenization differences (e.g., "▁Hall" vs "▁Hal").
pub fn force_eos_on_tail_repeat(logits: &mut [f32], eos_token_id: u32, output_ids: &[u32]) {
    let n = TAIL_REPEAT_NGRAM;
    let tail_len = TAIL_REPEAT_CHECK_LEN;

    // Need at least tail_len + n tokens before the check can be meaningful.
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
        // progress = 0.5 < LENGTH_PENALTY_START (0.70) — logits must be unchanged.
        let mut logits = vec![0.0f32; 5];
        let eos = 1u32;
        apply_length_bias(&mut logits, eos, 5, 10); // step/capacity = 0.5
        assert!((logits[eos as usize]).abs() < 1e-6);
    }

    #[test]
    fn length_bias_at_threshold() {
        // progress == LENGTH_PENALTY_START exactly → fraction = 0 → bias = 0.
        let mut logits = vec![0.0f32; 5];
        let eos = 2u32;
        apply_length_bias(&mut logits, eos, 6, 10); // 6/10 = 0.60 exactly
        assert!((logits[eos as usize]).abs() < 1e-6);
    }

    #[test]
    fn length_bias_at_capacity() {
        // step == capacity → fraction = 1.0 → bias == EOS_LOGIT_BIAS.
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
        // All logits unchanged.
        assert!(logits.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn tail_repeat_too_short() {
        // Fewer than TAIL_REPEAT_CHECK_LEN + TAIL_REPEAT_NGRAM tokens — no-op.
        let mut logits = vec![1.0f32; 10];
        let eos = 0u32;
        let output_ids: Vec<u32> = (0..TAIL_REPEAT_CHECK_LEN as u32).collect();
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn tail_repeat_fires_on_repeat_ngram() {
        // Build history where the tail repeats a prior n-gram.
        // history (tail_start): [1, 2, 3, 4] repeated at the end.
        let n = TAIL_REPEAT_NGRAM;
        let tail_len = TAIL_REPEAT_CHECK_LEN;
        let eos = 5u32;

        // Prefix: enough history containing [1,2,3,4], then padding, then tail repeating it.
        let mut output_ids: Vec<u32> = vec![10; tail_len]; // padding before tail
        // Place [1,2,3,4] at start of padding (in history before tail).
        output_ids[..n].copy_from_slice(&[1, 2, 3, 4]);
        // Append tail of tail_len tokens that also contains [1,2,3,4].
        let mut tail: Vec<u32> = vec![20; tail_len];
        tail[..n].copy_from_slice(&[1, 2, 3, 4]);
        output_ids.extend_from_slice(&tail);

        let mut logits = vec![1.0f32; 30];
        logits[eos as usize] = 0.5;
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);

        // All non-EOS logits must be -inf.
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
        // Tail has unique n-grams not present in history — logits unchanged.
        let tail_len = TAIL_REPEAT_CHECK_LEN;
        let eos = 0u32;

        // History: sequential tokens 0..tail_len, tail: unique tokens.
        let mut output_ids: Vec<u32> = (0..tail_len as u32).collect();
        let tail: Vec<u32> = (100..100 + tail_len as u32).collect();
        output_ids.extend_from_slice(&tail);

        let mut logits = vec![1.0f32; 200];
        force_eos_on_tail_repeat(&mut logits, eos, &output_ids);
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn tail_repeat_eos_token_preserved() {
        // When the detector fires, the EOS logit must remain finite.
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
}
