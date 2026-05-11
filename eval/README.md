# Locale evaluation harness

A small Python harness for deciding which BCP 47 locale codes to add to the
translate-side `Language` enum.

The translator engine accepts any BCP 47 code as input — `FromStr` falls
back to the base language for unknown regions — but only ~70 codes are
explicitly enumerated and validated. This harness scores **candidate**
codes against two signals so we can promote the ones that consistently
produce decent output.

## What it does

For each candidate locale code:

1. **Translate** a fixed set of English source sentences into the target
   locale via the running translator API.
2. **Phase 1 — output-language consistency**: feed each translation back
   through `/detect-language` and check what fraction were detected as the
   target's base language. Catches locales where the model echoes the
   source, returns English, or produces gibberish.
3. **Phase 2 — MetricX-23-QE quality estimation**: reference-free quality
   scoring with [Google's MetricX](https://github.com/google-research/metricx)
   (Apache 2.0). Returns a single quality score per `(source, hypothesis)`
   pair on a roughly 0–25 scale where **lower is better**.
4. Classify each candidate as **PASS / BORDERLINE / FAIL** based on
   thresholds (defaults: ≥ 0.90 consistency and ≤ 2.5 mean MetricX → PASS;
   < 0.85 consistency or ≥ 5.0 MetricX → FAIL).

Output is a results CSV plus a translations CSV (full source/hypothesis
pairs, detected language, MetricX score per row) for manual review.

> **Best-effort, not authoritative.** Reference-free QE has known biases.
> Treat results as directional — the harness should help us prioritize,
> not replace human review for production decisions.

## Why MetricX, not LLM-as-judge

MetricX-23-QE is purpose-built for translation quality estimation: trained
on WMT data, used by Google internally to evaluate TranslateGemma itself.
It returns calibrated scores instead of a 1–5 rubric. It runs locally with
no API costs and an Apache 2.0 license. LLM-as-judge is a coarser proxy
that we'd reach for only if MetricX didn't exist.

## Prerequisites

- Translator API running (`cargo run -p translator-api`) on the URL
  passed via `--api-url` (default `http://localhost:3000`). The API must
  be built from a branch that includes the regional-variants enum.
- **Python 3.10 or 3.11.** MetricX pins `transformers==4.30.2` and
  `sentencepiece==0.1.99`, which have no prebuilt wheels for Python
  3.12+. On macOS: `brew install python@3.11`. The Makefile auto-detects
  3.11 then 3.10; override with `make setup PYTHON_BIN=/path/to/python`.
- ~5 GB free disk for PyTorch + transformers + MetricX-23-QE weights.

## Setup (Makefile)

From the `eval/` directory:

```bash
make setup
```

This:

- creates a `.venv/` virtualenv,
- pip-installs Python deps (torch, transformers, sentencepiece, requests, …),
- clones [`google-research/metricx`](https://github.com/google-research/metricx)
  into `.metricx/`,
- pip-installs MetricX in editable mode.

The MetricX model itself (`google/metricx-23-qe-large-v2p0`, ~1.2 GB)
downloads from HuggingFace on the **first** evaluation run and is cached
in `~/.cache/huggingface/hub/`.

## Workflow

1. **Copy the example candidates file** and edit. The real
   `candidates.csv` is gitignored — it's per-project and may include
   codes you don't want to publish.

   ```bash
   cp candidates.example.csv candidates.csv
   $EDITOR candidates.csv
   ```

   Columns: `code,name,tier,notes`. Only `code` is required.

2. **Calibrate first.** Run the harness on a known-good set (the
   validated locales already in the enum) so you can see what scores
   "good" looks like and adjust thresholds if needed.

   Example `calibration.csv`:

   ```bash
   printf 'code,name,tier,notes\nfr,French,calib,\nde,German,calib,\nja,Japanese,calib,\npt-BR,Brazilian Portuguese,calib,\nzh-CN,Simplified Chinese,calib,\n' > calibration.csv
   make calibrate
   ```

   All five should land at PASS. If any don't, your thresholds are too
   strict — adjust the `PASS_*` / `FAIL_*` constants near the top of
   `harness.py`.

3. **Run the real candidate set:**

   ```bash
   make run                # full pipeline incl. MetricX
   make run-skip-judge     # Phase 1 only — quick smoke check, no MetricX
   ```

4. **Review** `results/results-<timestamp>.csv` for PASS/FAIL flags.
   For BORDERLINE candidates, open `results/translations-<timestamp>.csv`
   and inspect the actual translations and MetricX per-sentence scores.

## Make targets

| Target | What it does |
|---|---|
| `make setup` | Create venv, install deps, clone+install MetricX |
| `make calibrate` | Run harness on `calibration.csv` |
| `make run` | Run harness on `candidates.csv` (Phase 1 + 2) |
| `make run-skip-judge` | Run harness on `candidates.csv` (Phase 1 only) |
| `make clean-results` | Delete `results/results-*.csv` and `translations-*.csv` |
| `make clean` | `clean-results` plus remove `.venv/`, `.metricx/`, caches |

## CLI reference (direct invocation)

The Makefile targets just wrap `python harness.py`. For ad-hoc options:

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates PATH` | `<script_dir>/candidates.csv` | Input CSV |
| `--sources PATH` | `<script_dir>/sources.txt` | One source sentence per line |
| `--api-url URL` | `http://localhost:3000` (or `$TRANSLATOR_API_URL`) | Running translator API |
| `--metricx-model NAME` | `google/metricx-23-qe-large-v2p0` (or `$METRICX_MODEL`) | MetricX model. Use the `xl`/`xxl` variants for higher fidelity (3 GB / 12 GB) |
| `--metricx-tokenizer NAME` | `google/mt5-xl` (or `$METRICX_TOKENIZER`) | Tokenizer model |
| `--metricx-batch-size N` | `4` | Forward-pass batch size |
| `--metricx-max-input N` | `1024` | Max input tokens |
| `--skip-judge` | off | Skip Phase 2; Phase 1 only |
| `--limit N` | all | Process at most N candidates |
| `--output PATH` | `results/results-<timestamp>.csv` | Override results path |

## Score interpretation

Both the per-candidate console line and the results CSV include a
`metricx_quality` label that buckets the raw MetricX-23 score into a
human-readable band (per the MetricX paper):

| MetricX score | Quality label | Interpretation |
|---|---|---|
| < 2.0 | `near-human` | Excellent — comparable to human translation |
| < 5.0 | `good` | Minor errors only |
| < 10.0 | `acceptable` | Noticeable errors but the meaning carries |
| ≥ 10.0 | `poor` | Major errors / often unusable |

Lower is better. The `translations-*.csv` artifact also has a
`metricx_quality` column per row, so you can sort it to find the worst
sentences for a given candidate.

## Threshold tuning

Constants near the top of `harness.py`:

```python
PASS_CONSISTENCY = 0.90       # Phase 1: fraction of outputs in target language
FAIL_CONSISTENCY = 0.85
PASS_METRICX_MAX = 2.5        # Phase 2: mean MetricX score (lower = better)
FAIL_METRICX_MIN = 5.0
```

These defaults are **calibrated against TranslateGemma 4B's WMT24++ set**
(`fr`, `de`, `ja`, `pt-BR`, `zh-CN`), which scored 0.63–1.00 — solidly
in the `near-human` band. The PASS threshold of 2.5 leaves ~3× headroom
over that worst case; FAIL at 5.0 marks the boundary where MetricX
starts indicating noticeable errors.

If a known-good (validated) locale comes back FAIL when you re-calibrate,
the thresholds are too tight — relax them. If many obvious failures
sneak through as PASS, tighten further.

## Source sentence set

`sources.txt` holds 30 short English sentences chosen to cover variety:
declarative, question, imperative, formal, casual, with names, numbers,
and idioms. Small enough to keep run time down, large enough for a
coarse pass/fail. Swap in something larger (FLORES-200 dev set is the
obvious upgrade) if you want tighter calibration.

## Performance notes

MetricX inference is the slow step. On Apple Silicon (M-series), the
Large model scores ~5–10 pairs per second on CPU; XL is ~3× slower.
PyTorch's MPS backend is used automatically when available. For 50
candidates × 30 sources = 1500 pairs, expect roughly 3–5 minutes on Large
CPU. The model loads once per harness invocation (~10–15 s warm-up).

If MetricX is the bottleneck, batch evaluations (one harness invocation
covering many candidates) are far cheaper than per-candidate runs.
