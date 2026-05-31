# universal-translator


[![CI](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-playground/universal-translator/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Platform: Linux | macOS](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

A universal text translator built in Rust. Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) (via the `llama-cpp-2` Rust crate) for fast, fully local inference of [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it) (Gemma 3 4B instruction-tuned) — 98 translate-side languages and locales (WMT24++ validated set plus harness-validated additions) — and [Lingua](https://github.com/pemistahl/lingua-rs) for automatic source-language detection (75 base languages, plus script and heuristic refinements that surface regional variants like `zh-Hant` and `pt-BR`).

No API keys required. No network calls at runtime. Everything runs on your machine.

**License:** MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
**Model attributions:** see [ATTRIBUTIONS.md](ATTRIBUTIONS.md) — covers TranslateGemma 4B and the runtime libraries used for inference.
**Model license:** TranslateGemma 4B weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — see [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE).


---

## Supported languages

The translate side and detect side have different ceilings. Translate is built
on [WMT24++](https://huggingface.co/datasets/google/wmt24pp), the canonical
TranslateGemma training set. Detect is layered on
[Lingua](https://github.com/pemistahl/lingua-rs) (75 base languages) with two
post-processing passes for regional granularity.

**How to read the table:**

- **Trans** ✓ — code is accepted as a `source_language` / `target_languages`
  value by the engine. Inputs are BCP 47, dash or underscore, case-insensitive
  (`pt-BR`, `pt_BR`, `PT-br` all work). Unknown region tags fall back to the
  base language (e.g. `pt-AO` → `pt`).
- **Detect** ✓ — the detector can emit this exact code as a result.
- **Both** ✓ — code round-trips: detect can produce it AND translate accepts it.
- **(heur.)** — produced by the heuristic dialect classifier (marker-word
  scoring). Best-effort; falls back to the base language when text is too
  short or neutral to commit. May false-positive on adversarial input.
- **[Brackets]** — script-tag refinement using Unicode-block tests
  (`zh-Hans`, `sr-Cyrl`, etc.). Deterministic, no false positives.
- Codes only checked under **Detect** (e.g. `cy`, `ka`, `eu`) are languages
  Lingua identifies but the engine cannot translate. The auto-detect translate
  flow returns `UnsupportedLanguage` for these — pass an explicit
  `source_language` to translate from a different language.
- 8 codes (`af`, `am`, `ha`, `ms`, `mt`, `ne`, `si`, `yi`) are translate-supported
  but not in the WMT24++ training distribution. They work because Gemma 3
  instruct generalizes, but quality is best-effort.

Run `cargo run -p translator-cli -- languages --for translate` (or `--for
detect`) for the live list.

| Code     | Language                | Trans | Detect | Both |
|----------|-------------------------|-------|--------|------|
| af       | Afrikaans               |   ✓   |   ✓    |  ✓   |
| am       | Amharic                 |   ✓   |   ✓    |  ✓   |
| ar       | Arabic                  |   ✓   |   ✓    |  ✓   |
| ar-EG    | Egyptian Arabic         |   ✓   |        |      |
| ar-SA    | Saudi Arabic            |   ✓   |        |      |
| az       | Azerbaijani             |       |   ✓    |      |
| az-Arab  | Azerbaijani [Arabic]    |       |   ✓    |      |
| az-Cyrl  | Azerbaijani [Cyrl]      |       |   ✓    |      |
| az-Latn  | Azerbaijani [Latn]      |       |   ✓    |      |
| be       | Belarusian              |       |   ✓    |      |
| bg       | Bulgarian               |   ✓   |   ✓    |  ✓   |
| bn       | Bengali                 |   ✓   |   ✓    |  ✓   |
| bs       | Bosnian                 |       |   ✓    |      |
| ca       | Catalan                 |   ✓   |   ✓    |  ✓   |
| cs       | Czech                   |   ✓   |   ✓    |  ✓   |
| cy       | Welsh                   |       |   ✓    |      |
| da       | Danish                  |   ✓   |   ✓    |  ✓   |
| de       | German                  |   ✓   |   ✓    |  ✓   |
| el       | Greek                   |   ✓   |   ✓    |  ✓   |
| en       | English                 |   ✓   |   ✓    |  ✓   |
| en-GB    | English [UK] (heur.)    |       |   ✓    |      |
| en-US    | English [US] (heur.)    |       |   ✓    |      |
| eo       | Esperanto               |       |   ✓    |      |
| es       | Spanish                 |   ✓   |   ✓    |  ✓   |
| es-MX    | Mexican Spanish         |   ✓   |        |      |
| et       | Estonian                |   ✓   |   ✓    |  ✓   |
| eu       | Basque                  |       |   ✓    |      |
| fa       | Persian                 |   ✓   |   ✓    |  ✓   |
| fi       | Finnish                 |   ✓   |   ✓    |  ✓   |
| fil      | Filipino                |   ✓   |        |      |
| fr       | French                  |   ✓   |   ✓    |  ✓   |
| fr-CA    | Canadian French (heur.) |   ✓   |   ✓    |  ✓   |
| fr-FR    | European French (heur.) |   ✓   |   ✓    |  ✓   |
| ga       | Irish                   |       |   ✓    |      |
| gu       | Gujarati                |   ✓   |   ✓    |  ✓   |
| ha       | Hausa                   |       |   ✓    |      |
| he       | Hebrew                  |   ✓   |   ✓    |  ✓   |
| hi       | Hindi                   |   ✓   |   ✓    |  ✓   |
| hr       | Croatian                |   ✓   |   ✓    |  ✓   |
| hu       | Hungarian               |   ✓   |   ✓    |  ✓   |
| hy       | Armenian                |       |   ✓    |      |
| id       | Indonesian              |   ✓   |   ✓    |  ✓   |
| is       | Icelandic               |   ✓   |   ✓    |  ✓   |
| it       | Italian                 |   ✓   |   ✓    |  ✓   |
| ja       | Japanese                |   ✓   |   ✓    |  ✓   |
| ka       | Georgian                |       |   ✓    |      |
| kk       | Kazakh                  |       |   ✓    |      |
| kn       | Kannada                 |   ✓   |   ✓    |  ✓   |
| ko       | Korean                  |   ✓   |   ✓    |  ✓   |
| la       | Latin                   |       |   ✓    |      |
| lg       | Ganda                   |       |   ✓    |      |
| lt       | Lithuanian              |   ✓   |   ✓    |  ✓   |
| lv       | Latvian                 |   ✓   |   ✓    |  ✓   |
| mi       | Maori                   |       |   ✓    |      |
| mk       | Macedonian              |       |   ✓    |      |
| ml       | Malayalam               |   ✓   |   ✓    |  ✓   |
| mn       | Mongolian               |       |   ✓    |      |
| mn-Cyrl  | Mongolian [Cyrl]        |       |   ✓    |      |
| mn-Mong  | Mongolian [Mong]        |       |   ✓    |      |
| mr       | Marathi                 |   ✓   |   ✓    |  ✓   |
| ms       | Malay                   |   ✓   |   ✓    |  ✓   |
| nb       | Norwegian Bokmål        |       |   ✓    |      |
| ne       | Nepali                  |   ✓   |   ✓    |  ✓   |
| nl       | Dutch                   |   ✓   |   ✓    |  ✓   |
| nn       | Norwegian Nynorsk       |       |   ✓    |      |
| no       | Norwegian               |   ✓   |        |      |
| pa       | Punjabi                 |       |   ✓    |      |
| pa-Arab  | Punjabi [Shahmukhi]     |       |   ✓    |      |
| pa-Guru  | Punjabi [Gurmukhi]      |       |   ✓    |      |
| pl       | Polish                  |   ✓   |   ✓    |  ✓   |
| pt       | Portuguese              |   ✓   |   ✓    |  ✓   |
| pt-BR    | Brazilian Pt (heur.)    |   ✓   |   ✓    |  ✓   |
| pt-PT    | European Pt (heur.)     |   ✓   |   ✓    |  ✓   |
| ro       | Romanian                |   ✓   |   ✓    |  ✓   |
| ru       | Russian                 |   ✓   |   ✓    |  ✓   |
| si       | Sinhala                 |   ✓   |   ✓    |  ✓   |
| sk       | Slovak                  |   ✓   |   ✓    |  ✓   |
| sl       | Slovenian               |   ✓   |   ✓    |  ✓   |
| sn       | Shona                   |       |   ✓    |      |
| so       | Somali                  |       |   ✓    |      |
| sq       | Albanian                |       |   ✓    |      |
| sr       | Serbian                 |   ✓   |   ✓    |  ✓   |
| sr-Cyrl  | Serbian [Cyrl]          |       |   ✓    |      |
| sr-Latn  | Serbian [Latn]          |       |   ✓    |      |
| st       | Southern Sotho          |       |   ✓    |      |
| sv       | Swedish                 |   ✓   |   ✓    |  ✓   |
| sw       | Swahili                 |   ✓   |   ✓    |  ✓   |
| sw-KE    | Kenyan Swahili          |   ✓   |        |      |
| sw-TZ    | Tanzanian Swahili       |   ✓   |        |      |
| ta       | Tamil                   |   ✓   |   ✓    |  ✓   |
| te       | Telugu                  |   ✓   |   ✓    |  ✓   |
| th       | Thai                    |   ✓   |   ✓    |  ✓   |
| tl       | Tagalog                 |       |   ✓    |      |
| tn       | Tswana                  |       |   ✓    |      |
| tr       | Turkish                 |   ✓   |   ✓    |  ✓   |
| ts       | Tsonga                  |       |   ✓    |      |
| uk       | Ukrainian               |   ✓   |   ✓    |  ✓   |
| ur       | Urdu                    |   ✓   |   ✓    |  ✓   |
| vi       | Vietnamese              |   ✓   |   ✓    |  ✓   |
| xh       | Xhosa                   |       |   ✓    |      |
| yi       | Yiddish                 |   ✓   |   ✓    |  ✓   |
| yo       | Yoruba                  |       |   ✓    |      |
| zh       | Chinese                 |   ✓   |   ✓    |  ✓   |
| zh-CN    | Simplified Chinese      |   ✓   |   ✓    |  ✓   |
| zh-TW    | Traditional Chinese     |   ✓   |   ✓    |  ✓   |
| zu       | Zulu                    |       |   ✓    |      |

**Counts:** Translate enum has **98 variants** (`Language::all()`); detector
emits **116 distinct codes** (lingua's 75 base + script and dialect refinements).
The table above lists the canonical subset; call `GET /languages?for=translate`
or `for=detect` for the authoritative live list.

**Aliases.** When detect emits a code that isn't a translate variant directly,
`POST /detect-language` includes a `translate_language` field that maps it
to the right `Language` enum member. `FromStr` accepts any of these forms as
input on `/translate`, so callers can pass either side:

- `zh-Hans` ⇔ `zh-CN`, `zh-Hant` ⇔ `zh-TW` — script vs region BCP 47 subtag
  style for the **same locale**. Detect emits the region form (matches
  WMT24++); FromStr accepts both.
- `nb` (Bokmål), `nn` (Nynorsk) → `no` — distinct ISO 639-1 codes for
  distinct written standards. The model wasn't trained per-variant, so
  translate has only the macrolanguage `no`. Detect emits `nb`/`nn`
  (preserves the linguistic distinction); `translate_language` reports
  `"no"`.
- `tl` (Tagalog) → `fil` (Filipino) — distinct ISO 639-1 codes. Lingua
  emits `tl`; WMT24++ uses `fil`. Detect emits `tl`; `translate_language`
  reports `"fil"`.
- `iw` → `he` — deprecated ISO 639-1 form for Hebrew, accepted as input.

Example detect responses:
```json
{ "language": "nb",   "translate_language": "no",  "confidence": 0.92 }
{ "language": "tl",   "translate_language": "fil", "confidence": 0.94 }
{ "language": "cy",   "translate_language": null,  "confidence": 0.88 }  // Welsh — not translate-supported
{ "language": "pt-BR","translate_language": "pt-BR","confidence": 0.96 } // already aligned
```

Source language is detected automatically — no configuration required.

---

## Workspace layout

```
universal-translator/
├── translator-core/   # Core library: engine, language detector, types
├── translator-api/    # Axum HTTP API server
├── translator-cli/    # Command-line interface
└── docs/models.md     # Model management guide
```

## Prerequisites

- Rust toolchain (stable) — [rustup.rs](https://rustup.rs)
- CMake ≥ 3.14 and a C++17 compiler (required to build vendored llama.cpp)
  - macOS: `xcode-select --install` (provides both)
  - Ubuntu/Debian: `sudo apt install cmake g++`
  - Fedora: `sudo dnf install cmake gcc-c++`
- Tested on Linux (x86_64, arm64) and macOS (Apple Silicon)

## Quick start

### Build

```bash
cargo build --release
```

### Get the model

```bash
cargo run -p translator-cli -- setup
```

This downloads TranslateGemma 4B Q8_0 (~4.1 GB) directly from HuggingFace into the
default model directory. No Python or HuggingFace CLI required.

For the smaller Q4_K_M variant (~2.6 GB):

```bash
cargo run -p translator-cli -- setup --url https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf
```

Use `--model-path <path>` to select a specific model file at runtime. Q8_0 is the default.
See [docs/models.md](docs/models.md) for details and alternative hosting options.

### Run the API server

```bash
cargo run -p translator-api
```

The server listens on `http://localhost:3000`. See [API.md](API.md) for endpoint
reference and request/response schemas.

To run with full observability (traces, metrics, logs pushed to a local monitoring stack):

```bash
docker compose -f docker/docker-compose.yml up -d
OTLP_ENDPOINT=http://localhost:4317 \
  cargo run -p translator-api --features opentelemetry
```

Grafana opens at http://localhost:3001 with a pre-built dashboard. See [METRICS.md](METRICS.md)
for the full setup guide and metrics catalogue.

### Run the CLI

```bash
cargo run -p translator-cli -- translate -t "Hello world" -l fr
```

See [CLI.md](CLI.md) for the full command reference (`translate`, `detect-language`,
`detect`, `languages`) including all flags and output formats.

## Documentation

- [CLI Reference](CLI.md) — all subcommands, flags, and output formats
- [API Reference](API.md) — HTTP endpoints, request/response schemas, examples
- [Engine Internals](ENGINE.md) — inference limits, concurrency, sampling, language detection
- [Observability](METRICS.md) — OTel traces/metrics/logs, Grafana dashboard, metrics catalogue

## Licensing

The source code in this repository is dual-licensed under the
[Apache License, Version 2.0](LICENSE-APACHE) and the [MIT License](LICENSE-MIT).
You may choose either license for your use of the source code.

### Model license

The **TranslateGemma** model weights and any model derivatives are **not** covered
by the Apache or MIT licenses. They are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
By downloading or using the model, you agree to abide by those terms.
See [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE).
