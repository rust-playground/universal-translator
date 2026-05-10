# Locale evaluation harness

A small Python harness for deciding which BCP 47 locale codes to add to the
translate-side `Language` enum.

The translator engine accepts any BCP 47 code as input — `FromStr` falls
back to the base language for unknown regions — but only ~70 codes are
explicitly enumerated and validated. This harness scores **candidate**
codes (anything outside the current enum) against two signals so we can
promote the ones that consistently produce decent output.

## What it does

For each candidate locale code:

1. **Translate** a fixed set of English source sentences into the target
   locale via the running translator API.
2. **Phase 1 — output-language consistency**: feed each translation back
   through our own `/detect-language` endpoint and check what fraction
   were detected as the target's base language. Catches locales where
   the model echoes the source, returns English, or produces gibberish.
3. **Phase 4 — LLM-as-judge sample**: send a sample of source/translation
   pairs to Claude and score fluency + adequacy on a 1–5 scale.
4. Classify each candidate as **PASS / BORDERLINE / FAIL** based on
   thresholds (defaults: ≥ 0.90 consistency and ≥ 3.5 judge mean → PASS;
   < 0.85 consistency or < 3.0 judge → FAIL).

Output is a results CSV plus a translations CSV (full source/hypothesis
pairs) for manual review.

> **Best-effort, not authoritative.** Reference-free scoring has known
> biases. Treat results as directional — the harness should help us
> prioritize, not replace human review for production decisions.

## Prerequisites

- Translator API running (`cargo run -p translator-api`) on the URL
  passed via `--api-url` (default `http://localhost:3000`).
- Python 3.10+.
- `ANTHROPIC_API_KEY` exported in the environment (skip with
  `--skip-judge` for Phase-1-only runs).

## Setup

```bash
cd eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

1. **Copy the example candidates file** and fill in the codes you want to
   evaluate. The real `candidates.csv` is gitignored — it's per-project
   and may include codes you don't want to publish.

   ```bash
   cp candidates.example.csv candidates.csv
   $EDITOR candidates.csv
   ```

   Columns: `code,name,tier,notes`. `tier` is free-form (we use A/B/C
   internally to indicate priority/risk). Only `code` is required.

2. **Run the harness:**

   ```bash
   # Full run with judge (costs ~$0.01–0.04 per candidate on Haiku)
   python harness.py

   # Phase 1 only — no API costs
   python harness.py --skip-judge

   # Smoke test on the first 3 candidates
   python harness.py --limit 3

   # Custom API URL or judge model
   python harness.py --api-url http://localhost:3000 --judge-model claude-sonnet-4-6
   ```

3. **Review the results CSV** at `results/results-<timestamp>.csv`.
   PASS candidates are good promotion targets; BORDERLINE deserve a
   manual look at the translations CSV; FAIL means leave them off the
   enum (or note them as best-effort).

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates PATH` | `eval/candidates.csv` | Input CSV |
| `--sources PATH` | `eval/sources.txt` | One source sentence per line |
| `--api-url URL` | `http://localhost:3000` (or `$TRANSLATOR_API_URL`) | Running translator API |
| `--judge-model MODEL` | `claude-haiku-4-5` (or `$JUDGE_MODEL`) | Anthropic model for the judge |
| `--sample-judge N` | `10` | Translations per candidate sent to the judge |
| `--skip-judge` | off | Skip Phase 4; Phase 1 only |
| `--limit N` | all | Process at most N candidates |
| `--output PATH` | `results/results-<timestamp>.csv` | Override results path |

## Calibration

Before trusting the thresholds, **run the harness on a known-good set**
(e.g. the existing enum's WMT24++-derived locales). They should all PASS
or BORDERLINE. If they don't, your thresholds are too strict, the source
set is unrepresentative, or the judge model is unreliable for those
languages — adjust before evaluating new candidates.

## Threshold tuning

The constants near the top of `harness.py`:

```python
PASS_CONSISTENCY = 0.90
PASS_JUDGE_MEAN = 3.5
FAIL_CONSISTENCY = 0.85
FAIL_JUDGE_MEAN = 3.0
```

Adjust as the calibration run reveals where validated locales actually
land. A locale should not FAIL the harness if Google's WMT24++ pair for
it scored well in their published metrics.

## Source sentence set

`sources.txt` holds 30 short English sentences chosen to cover variety:
declarative, question, imperative, formal, casual, with names, numbers,
and idioms. The set is small enough to keep API costs down while giving
enough signal for a coarse pass/fail judgement. Swap in something larger
(FLORES-200 dev set is the obvious upgrade) if you want tighter
calibration.

## Cost notes

Cost is dominated by the Phase 4 judge calls. Per candidate with
`--sample-judge 10`:

- **Haiku 4.5**: ~$0.01
- **Sonnet 4.6**: ~$0.04
- **Opus 4.7**: ~$0.20

For a 50-candidate run on Haiku that's around $0.50; on Sonnet, $2;
on Opus, $10. Phase 1 (translation + detection) is free — runs against
the local API.

If you want to be conservative: do a `--skip-judge` pass first to drop
obvious failures, then run the survivors through the judge.

## Cleanup

Results are gitignored under `results/`. Translations CSV is sized to
~`n_candidates * n_sources` rows; on disk this is small (kilobytes per
run) but feel free to prune old runs.
