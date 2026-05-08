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
synchronously at startup into an `Arc<LoadedGemmaModel>` — shared read-only across
the engine and scheduler.

---

## Model

**TranslateGemma 4B** — Gemma 3 4B instruction-tuned, decoder-only, fine-tuned for
translation. The translate-side `Language` enum exposes 70 entries: 55 base
ISO 639-1 codes (the original set) + 4 base codes added for WMT24++ coverage
(`he`, `is`, `fil`, `zu`) + 11 regional pairs from WMT24++ (`ar-EG`,
`ar-SA`, `es-MX`, `fr-CA`, `fr-FR`, `pt-BR`, `pt-PT`, `sw-KE`, `sw-TZ`,
`zh-CN`, `zh-TW`). 8 of the original codes (`af`, `am`, `ha`, `ms`, `mt`,
`ne`, `si`, `yi`) sit outside the WMT24++ training distribution and are
best-effort.

- Weights: GGUF quantised — two formats supported:
  - **Q8_0** (`model-q8_0.gguf`, ~4.1 GB) — default, higher precision
  - **Q4_K_M** (`model-q4k.gguf`, ~2.6 GB) — opt-in, smaller footprint, comparable throughput under llama.cpp
- Selection: `--model-path` flag or `MODEL_PATH` env var. Default: `<cache>/ut/models/translategemma-4b/model-q8_0.gguf`.
- Framework: [llama.cpp](https://github.com/ggerganov/llama.cpp) via the `llama-cpp-2` Rust crate
- Tokenizer: embedded in the GGUF file (no separate `tokenizer.json` needed)

The inference stack uses `LlamaModel` for weight loading and `LlamaContext` for
batched decode, with weights shared read-only across concurrent decode slots.

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

---

## Text chunking

The engine automatically splits long inputs at `\n\n` paragraph boundaries, falling
back to Unicode sentence boundaries (via `unicode-segmentation`) for oversized
paragraphs. Chunked translations are reassembled before returning results.

### Configurable limits

| Env var / constant | Default | Purpose |
|--------------------|---------|---------|
| `MAX_CHUNK_CHARS` | 1 500 | Hard ceiling per chunk (sentence fallback limit) |
| `PARAGRAPH_TARGET_CHARS` | 800 | Soft target — flush paragraph accumulator when exceeded |

### Line ending normalization

`\r\n` (Windows) and `\r` (old Mac) line endings are normalized to `\n` before
chunking. Translated output always uses Unix line endings (`\n`), regardless of the
input encoding.

### Per-text error handling

If a chunk is still too large after splitting (e.g. no sentence boundaries found),
the scheduler may return an `InputTooLong` error. This error is routed to the
individual text's `errors` field in the response — other texts in the same batch are
unaffected and return their translations normally. Only `ServiceUnavailable`
(backpressure) fails the entire batch.

---

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

`Detector::detect` returns a BCP 47 `String`. The pipeline runs four layers,
each non-destructive — every step either refines the previous result or
passes it through unchanged:

1. **Lingua** (`lingua-rs`) — statistical n-gram model covering 75 base
   languages. Returns the raw ISO 639-1 (or 639-3 for `fil` / `tl`) code in
   lowercase. Detection runs in parallel for multi-text requests.
2. **Script disambiguation** (deterministic, no false positives) — Unicode
   block / character-set membership refines specific base codes:
   - `zh` → `zh-CN` / `zh-TW` via Simplified-only vs Traditional-only
     character set lookup. Region form (matches WMT24++); FromStr accepts
     the script form (`zh-Hans` / `zh-Hant`) as input alias.
   - `sr` → `sr-Cyrl` / `sr-Latn`
   - `az` → `az-Cyrl` / `az-Latn` / `az-Arab`
   - `pa` → `pa-Guru` / `pa-Arab`
   - `mn` → `mn-Cyrl` / `mn-Mong`
   When the script doesn't match any rule, the base code passes through.
3. **Heuristic dialect refinement** (`translator-core/src/dialect.rs`) —
   marker word/phrase scoring for same-script regional pairs:
   - `pt` → `pt-BR` / `pt-PT`
   - `en` → `en-US` / `en-GB`
   - `fr` → `fr-CA` / `fr-FR`
   Aho-Corasick matcher with word-boundary check; commits when one side
   has ≥ 2 hits and beats the other by a margin of ≥ 2. **Streaming with
   early return** — scanning aborts as soon as the threshold is met, so
   cost is bounded on large inputs. If neither side commits, the base
   code passes through. Best-effort: short or neutral text typically
   yields no commit.
4. **Malayalam script fallback** — Malayalam (`ml`) is detected via the
   U+0D00–U+0D7F block when Lingua returns no result.

**Boundary contract.** The detect universe is broader than the translate-side
`Language` enum. Codes returned by step 1 may be lingua-only (`cy`, `ka`,
`eu`, `eo`, `la`); codes from step 2 may be script tags not in the enum
(`sr-Cyrl`, `pa-Guru`, `mn-Mong`). When the engine's auto-detect path can't
parse the detected code into a `Language`, it returns `UnsupportedLanguage`
with a clear message rather than silently falling back. Callers wanting the
broader detect surface for non-translation use cases can call the detector
directly and parse the string themselves.

`POST /detect-language` returns both the raw detect code (`language`) and
its translate-side equivalent (`translate_language: Option<Language>`) —
the latter applies `Language::FromStr` and surfaces the standard alias
mapping (`nb`/`nn` → `no`, `tl` → `fil`, `iw` → `he`, `zh-Hans` → `zh-CN`,
unknown region tags → base, etc.). One mapping table, two consumers
(input parsing on `/translate` and output mapping on `/detect-language`).

Detection is skipped entirely when the caller supplies a `source_language`
field; the supplied code is parsed via `Language::FromStr` (BCP 47, dash or
underscore, case-insensitive) and used directly. Unknown region tags fall
back to the base language (e.g. `pt-AO` → `Pt`) with a debug log.

### Confidence score

The `confidence` value returned by `POST /detect-language` (and the CLI `detect`
subcommand) is a **relative** score:

```
confidence = top / (top + second)
```

where `top` and `second` are the raw Lingua probability scores for the first- and
second-ranked candidate languages. See [API.md — confidence score semantics](API.md#confidence-score-semantics)
for the full interpretation table.
