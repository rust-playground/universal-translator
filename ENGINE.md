# Engine Internals

Reference for contributors and advanced users. For HTTP endpoint documentation, see
[API.md](API.md).

---

## Architecture

The workspace is split into three crates:

| Crate | Role |
|-------|------|
| `translator-core` | Library: engine, model loader, scheduler, detector, types |
| `translator-cli` | `ut` binary — Clap CLI (`translate`, `detect`, `languages`) |
| `translator-api` | Axum HTTP server |

`TranslationEngine` is `Clone`-cheap (Arc-backed internals). The model is loaded
once into an `Arc<OnceCell<Arc<LoadedGemmaModel>>>` — async `get_or_try_init` ensures
a single initialisation with a lock-free read path afterwards.

---

## Model

**TranslateGemma 4B** — Gemma 3 4B instruction-tuned, decoder-only, fine-tuned for
translation across 55 languages.

- Weights: GGUF quantised — two formats supported:
  - **Q4_K_M** (`model-q4k.gguf`, ~2.6 GB) — default, ~22% higher decode throughput
  - **Q8_0** (`model-q8_0.gguf`, ~4.1 GB) — opt-in, higher precision, imperceptible quality difference for most use cases
- Selection: `--model-file` flag or `MODEL_FILE` env var. Default auto-detects Q4_K_M → Q8_0 → any `*.gguf`.
- Framework: [Candle](https://github.com/huggingface/candle) (`candle-core`,
  `candle-transformers`, `candle-nn`)
- Tokenizer: HuggingFace fast tokenizer (`tokenizers` crate, `tokenizer.json`)

The inference stack uses a local fork of `quantized_gemma3` (`model_batched.rs`)
with an external KV cache so that weights are read-only and can be safely shared
across concurrent decode slots.

---

## Inference limits

**Output token limit: 4 096 tokens** (SLOT_CAPACITY). Generation truncates silently
at this ceiling — no error is returned.

There is no hard input token limit enforced at the API level. In practice, keep
prompts under approximately 2 000 tokens (~1 500 words). The total context window is
4 096 tokens (prompt + output combined), so very long inputs leave less budget for
the translation output.

**Recommendation:** split long documents into paragraphs or sentences, translate each
piece, then reassemble the results on the client side.

**Expected output length** is estimated as `(input_bytes / 3 + 15).clamp(15, 4096)`
from the original text's byte length. This estimate is used to calibrate EOS bias
(nudging the model toward finishing at an appropriate length) and is not a hard limit.

---

## Concurrency

The `ContinuousScheduler` maintains **24 parallel decode slots** (`N_SLOTS = 24`).
Each slot holds one in-flight translation (one source text × one target language).

Requests that arrive when all 24 slots are occupied queue internally until a slot
frees. There is no explicit queue size limit at the API level.

Multiple concurrent HTTP requests are coalesced into a single batched forward pass
each decode step, keeping GPU/CPU utilisation high even under mixed load.

---

## Sampling parameters

Parameters are defined in `translator-core/src/scheduler/sampling.rs`.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.15 | Low = near-deterministic output |
| Top-K | 40 | Vocabulary candidates per step |
| Top-P | 0.90 | Nucleus sampling cutoff |
| Repetition penalty | 1.10 | Applied to already-generated token IDs |
| No-repeat n-gram | 3 | Blocks any 3-gram that has already appeared |

---

## Language detection

Source language detection uses two layers:

1. **Lingua** (`lingua-rs`) — statistical n-gram model covering 75+ languages.
   Detection runs in parallel for multi-text requests.

2. **Unicode script fallback** — Malayalam (`ml`) is detected via Unicode block
   analysis (U+0D00–U+0D7F) when Lingua returns no result. This handles script-
   distinctive text that Lingua may under-represent.

Detection is skipped entirely when the caller supplies a `source_language` field in
the request; the supplied code is normalised and used directly.

### Confidence score

The `confidence` value returned by `POST /detect-language` (and the CLI `detect`
subcommand) is a **relative** score:

```
confidence = top / (top + second)
```

where `top` and `second` are the raw Lingua probability scores for the first- and
second-ranked candidate languages. See [API.md — confidence score semantics](API.md#confidence-score-semantics)
for the full interpretation table.
