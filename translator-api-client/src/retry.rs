use std::time::Duration;

/// Configuration for retry behaviour with exponential backoff and jitter.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(5),
        }
    }
}

impl RetryConfig {
    /// Compute the backoff duration for a given attempt (0-indexed).
    ///
    /// Exponential: `base * 2^attempt`, capped at `max_backoff`, plus 0–50% jitter.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let base_ms = self.base_backoff.as_millis() as u64;
        let exp_ms = base_ms.saturating_mul(1u64 << attempt.min(16));
        let max_ms = self.max_backoff.as_millis() as u64;
        let capped_ms = exp_ms.min(max_ms);

        // Cheap jitter: 0–50% of capped value using nanos as entropy source.
        let jitter_frac = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64)
            % 1000;
        let jitter_ms = capped_ms * jitter_frac / 2000; // 0..50%

        Duration::from_millis(capped_ms + jitter_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_increases() {
        let config = RetryConfig {
            max_retries: 5,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
        };
        // Without jitter the base values would be 100, 200, 400, 800, 1600.
        // With up to 50% jitter they'd be at most 150, 300, 600, 1200, 2400.
        let d0 = config.backoff_duration(0);
        let d2 = config.backoff_duration(2);
        // d2 base (400ms) should be greater than d0 max (150ms).
        assert!(d2.as_millis() >= 400, "d2={:?}", d2);
        assert!(d0.as_millis() <= 200, "d0={:?}", d0);
    }

    #[test]
    fn backoff_capped() {
        let config = RetryConfig {
            max_retries: 10,
            base_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(5),
        };
        let d = config.backoff_duration(20);
        // max 5000ms + 50% jitter = 7500ms max
        assert!(d.as_millis() <= 7500, "d={:?}", d);
    }
}
