# Model Management

This guide explains how to download and manage the TranslateGemma 4B model for use
with universal-translator.

---

## Default model path

| Platform | Default path |
|----------|-------------|
| Linux    | `~/.cache/ut/models/translategemma-4b/model-q8_0.gguf` (respects `$XDG_CACHE_HOME`) |
| macOS    | `~/Library/Caches/ut/models/translategemma-4b/model-q8_0.gguf` |

Override with the `MODEL_PATH` environment variable or `--model-path` flag.

---

## Download

```bash
cargo run -p translator-cli -- setup
```

`ut setup` downloads the Q8_0 GGUF weights (~4.1 GB) directly from HuggingFace.
No Python, `hf` CLI, or authentication required — the GGUF files are public.

For the smaller Q4_K_M variant (~2.6 GB):

```bash
cargo run -p translator-cli -- setup \
  --url https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf \
  --output <cache>/ut/models/translategemma-4b/model-q4k.gguf
```

Use `--force` to re-download if the file already exists.

### Available quantisations

| Format | File | Size | Description |
|--------|------|------|-------------|
| Q8_0 | `model-q8_0.gguf` | ~4.1 GB | Default — higher precision |
| Q4_K_M | `model-q4k.gguf` | ~2.6 GB | Opt-in — smaller footprint, comparable throughput under llama.cpp |

---

## Expected directory layout

After running `ut setup`, the model directory contains:

```
<cache>/ut/models/translategemma-4b/
└── model-q8_0.gguf         # ~4.1 GB — Q8_0 quantised GGUF weights (default)
```

Only the GGUF file is needed. The tokenizer is embedded in the GGUF — no separate
`tokenizer.json` or config files required.

---

## Runtime model selection

Use `--model-path` or the `MODEL_PATH` environment variable to select a specific model file:

```bash
# Use Q4_K_M via CLI flag
ut --model-path ~/models/model-q4k.gguf translate -t "Hello" -l fr

# Use Q4_K_M via env var
MODEL_PATH=~/models/model-q4k.gguf cargo run -p translator-cli -- translate -t "Hello" -l fr

# API server with Q4_K_M
MODEL_PATH=~/models/model-q4k.gguf cargo run -p translator-api
```

If no `--model-path` is specified, the default path is used. If the model file is not
found, the error message directs you to run `ut setup`.

---

## Build targets

```bash
cargo build                   # CPU (default)
cargo build --features metal  # macOS GPU (Metal)
cargo build --features cuda   # Linux GPU (CUDA/NVIDIA)
```

---

## Hosting the model files

The GGUF weights (~4.1 GB) are too large to check into git. Three options for
distributing them to teammates or CI:

### Option A — Direct download (recommended)

`ut setup` downloads directly from HuggingFace. No additional tooling needed.

### Option B — GitHub Releases

Zip the model file and attach it as a release asset. Useful if you want the
model version-locked to a specific code release.

### Option C — Cloud object storage (S3 / GCS / R2)

Upload the model file to a bucket. Use `ut setup --url <presigned-url>` to download
from your own hosting:

```bash
# Upload
aws s3 cp model-q8_0.gguf s3://your-bucket/translategemma-4b/model-q8_0.gguf

# Download (on a new machine)
cargo run -p translator-cli -- setup --url https://your-bucket.s3.amazonaws.com/translategemma-4b/model-q8_0.gguf
```

Cloudflare R2 has no egress fees, which makes it cost-effective for frequent downloads.

---

## Language detection: fully local

universal-translator uses [Lingua](https://github.com/pemistahl/lingua-rs) to
automatically detect the source language of incoming text. Lingua is entirely
self-contained:

- All language model data is compiled directly into the binary as Rust crates —
  there are no external data files to manage.
- Detection covers 75 base languages.
- Zero network calls are made at runtime. Lingua works completely offline.

On top of lingua's base output, the detector layers two passes for regional
granularity (see [ENGINE.md](../ENGINE.md#language-detection) for the full
pipeline):

- **Script disambiguation** — deterministic Unicode-block tests for
  `zh-CN`/`zh-TW` (Han Simplified vs Traditional character-set membership),
  `sr-Cyrl`/`sr-Latn`, `az-Cyrl`/`az-Latn`/`az-Arab`,
  `pa-Guru`/`pa-Arab`, `mn-Cyrl`/`mn-Mong`. No false positives.
- **Heuristic dialect markers** — best-effort word/phrase scoring for
  `pt-BR`/`pt-PT`, `en-US`/`en-GB`, `fr-CA`/`fr-FR`. Returns the base code
  when text is too short or neutral to commit.

Pure-dialect pairs without a script difference (e.g. `ar-EG` vs `ar-SA`,
`sw-KE` vs `sw-TZ`) return the base code — no off-the-shelf detector
distinguishes them reliably.

No configuration is required for language detection.

## Translate-side language set

The `Language` enum exposes 70 entries: 55 base ISO 639-1 codes + 4 added
base codes (`he`, `is`, `fil`, `zu`) + 11 regional pairs from the WMT24++
training set. The
[WMT24++ dataset](https://huggingface.co/datasets/google/wmt24pp) is the
canonical training set for TranslateGemma's regional variants. Inputs accept
BCP 47 in dash or underscore form (`pt-BR`, `pt_BR`); the `code()` returned
by the API/CLI is always the dash form. Unknown region tags fall back to
the base language (e.g. `pt-AO` → `pt`).

The TranslateGemma model card documents that `source_lang_code` and
`target_lang_code` accept either ISO 639-1 alpha-2 codes (`en`) or
regionalized variants (`en_US`, `en-GB`). The engine forwards regional
codes verbatim.
