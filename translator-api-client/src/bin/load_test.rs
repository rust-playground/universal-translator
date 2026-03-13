use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use translator_api_client::{ClientError, TranslatorClient};
use translator_core::types::TranslationRequest;

// ---------------------------------------------------------------------------
// Test inputs — same as the original Python load test
// ---------------------------------------------------------------------------

const TEST_INPUTS: &[&str] = &[
    // Short
    "Hello, how are you?",
    "The meeting starts at 10 AM tomorrow.",
    // Medium
    "The sun rises in the east and sets in the west. The coffee costs $3.50 and the newspaper costs €2.00.",
    // Long
    "The annual conference will be held in Geneva next month. Participants from over forty countries are expected to attend and registration closes on the fifteenth of March.",
    "Scientists have discovered a new species of deep-sea fish that produces its own bioluminescent light. The creature was found at a depth of three thousand metres during an unmanned submarine survey of the Pacific Ocean floor.",
];

/// Number of non-English languages (en→en is skipped by the engine).
const EN_ALL_LANGUAGE_COUNT: usize = 54;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Load test for the Universal Translator API.
///
/// Measures throughput and latency under concurrent load across all API endpoints.
#[derive(Parser)]
#[command(name = "load-test")]
struct Args {
    /// API base URL.
    #[arg(long, default_value = "http://localhost:3000")]
    url: String,

    /// Max concurrent in-flight requests.
    #[arg(long, default_value_t = 10)]
    concurrency: usize,

    /// Total requests per scenario.
    #[arg(long, default_value_t = 100)]
    requests: usize,

    /// Sequential warmup requests before measurement.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Scenarios to run (default: all).
    #[arg(value_enum)]
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Scenario {
    /// POST /translate — single target (fr)
    TranslateEnFr,
    /// POST /translate — all 54 non-English languages
    TranslateEnAll,
    /// POST /detect-language
    Detect,
    /// GET /languages
    Languages,
    /// GET /health
    Health,
}

impl Scenario {
    fn all() -> &'static [Scenario] {
        &[
            Scenario::TranslateEnFr,
            Scenario::TranslateEnAll,
            Scenario::Detect,
            Scenario::Languages,
            Scenario::Health,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Scenario::TranslateEnFr => "translate-en-fr",
            Scenario::TranslateEnAll => "translate-en-all",
            Scenario::Detect => "detect",
            Scenario::Languages => "languages",
            Scenario::Health => "health",
        }
    }

    /// Number of translations per successful request (for throughput calculation).
    fn translations_per_request(self) -> Option<usize> {
        match self {
            Scenario::TranslateEnFr => Some(1),
            Scenario::TranslateEnAll => Some(EN_ALL_LANGUAGE_COUNT),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Request execution
// ---------------------------------------------------------------------------

async fn execute_request(
    client: &TranslatorClient,
    scenario: Scenario,
    index: usize,
) -> Result<(), ClientError> {
    let input = TEST_INPUTS[index % TEST_INPUTS.len()];

    match scenario {
        Scenario::TranslateEnFr => {
            let req = TranslationRequest {
                texts: vec![input.to_string()],
                target_languages: vec!["fr".to_string()],
                source_language: Some("en".to_string()),
            };
            client.translate(&req).await?;
        }
        Scenario::TranslateEnAll => {
            let req = TranslationRequest {
                texts: vec![input.to_string()],
                target_languages: vec!["all".to_string()],
                source_language: Some("en".to_string()),
            };
            client.translate(&req).await?;
        }
        Scenario::Detect => {
            client.detect_language(input).await?;
        }
        Scenario::Languages => {
            client.languages().await?;
        }
        Scenario::Health => {
            client.health().await?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct ScenarioResults {
    scenario: Scenario,
    concurrency: usize,
    total_requests: usize,
    warmup: usize,
    wall_time: Duration,
    latencies: Vec<Duration>,
    errors: Vec<String>,
}

impl ScenarioResults {
    fn print(&self) {
        let n = self.latencies.len();
        let wall_secs = self.wall_time.as_secs_f64();
        let rps = if wall_secs > 0.0 {
            n as f64 / wall_secs
        } else {
            0.0
        };

        let mut sorted: Vec<u128> = self.latencies.iter().map(|d| d.as_millis()).collect();
        sorted.sort_unstable();

        let pct = |p: f64| -> u128 {
            if sorted.is_empty() {
                return 0;
            }
            let idx = ((p / 100.0) * n as f64) as usize;
            sorted[idx.min(n - 1)]
        };

        let mean_ms: u128 = if n > 0 {
            sorted.iter().sum::<u128>() / n as u128
        } else {
            0
        };

        // Header
        let scenario_label = self.scenario.name();
        println!();
        println!("=== Scenario: {scenario_label} ===");
        println!(
            "Concurrency: {}  |  Requests: {}  |  Warmup: {}",
            self.concurrency, self.total_requests, self.warmup
        );
        println!("Duration:   {wall_secs:.1}s");

        // Throughput
        if let Some(tpr) = self.scenario.translations_per_request() {
            let tps = rps * tpr as f64;
            println!("Throughput: {rps:.2} req/s  |  {tps:.1} translations/s");
        } else {
            println!("Throughput: {rps:.2} req/s");
        }

        // Latency percentiles
        if !sorted.is_empty() {
            println!(
                "Latency (ms): min={}  mean={}  p50={}  p75={}  p95={}  p99={}  max={}",
                sorted[0],
                mean_ms,
                pct(50.0),
                pct(75.0),
                pct(95.0),
                pct(99.0),
                sorted[n - 1],
            );
        }

        // Errors
        println!("Errors: {} / {}", self.errors.len(), self.total_requests);
        if let Some(first) = self.errors.first() {
            println!("  First error: {first}");
        }
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

async fn run_scenario(
    client: &TranslatorClient,
    scenario: Scenario,
    concurrency: usize,
    n_requests: usize,
    warmup: usize,
) -> ScenarioResults {
    let name = scenario.name();

    // --- Warmup ---
    if warmup > 0 {
        println!("  [{name}] Warming up ({warmup} sequential requests)...");
        for i in 0..warmup {
            if let Err(e) = execute_request(client, scenario, i).await {
                println!("  [{name}] WARNING: warmup request {} failed: {e}", i + 1);
            }
        }
    }

    // --- Load phase ---
    println!("  [{name}] Running {n_requests} requests (concurrency={concurrency})...");

    let sem = Arc::new(Semaphore::new(concurrency));
    let client = client.clone();
    let mut join_set = JoinSet::new();

    let start = Instant::now();

    for i in 0..n_requests {
        let client = client.clone();
        let sem = sem.clone();
        join_set.spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let t0 = Instant::now();
            let result = execute_request(&client, scenario, i).await;
            let latency = t0.elapsed();
            (latency, result)
        });
    }

    let mut latencies = Vec::with_capacity(n_requests);
    let mut errors = Vec::new();

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((latency, Ok(()))) => latencies.push(latency),
            Ok((latency, Err(e))) => {
                latencies.push(latency);
                errors.push(e.to_string());
            }
            Err(e) => errors.push(format!("task panic: {e}")),
        }
    }

    let wall_time = start.elapsed();

    ScenarioResults {
        scenario,
        concurrency,
        total_requests: n_requests,
        warmup,
        wall_time,
        latencies,
        errors,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let client = TranslatorClient::builder()
        .base_url(&args.url)
        .max_retries(0)
        .build()
        .expect("failed to build HTTP client");

    let scenarios: &[Scenario] = if args.scenarios.is_empty() {
        Scenario::all()
    } else {
        &args.scenarios
    };

    println!(
        "Load test: {} scenario(s), {} requests each, concurrency={}",
        scenarios.len(),
        args.requests,
        args.concurrency,
    );

    for &scenario in scenarios {
        let results =
            run_scenario(&client, scenario, args.concurrency, args.requests, args.warmup).await;
        results.print();
    }
}
