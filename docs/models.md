# Model Management

This guide explains how to download and manage the TranslateGemma 4B model for use
with universal-translator.

---

## Default model directory

| Platform | Default path |
|----------|-------------|
| Linux    | `~/.cache/ut/models` (respects `$XDG_CACHE_HOME`) |
| macOS    | `~/Library/Caches/ut/models` |

Override the default by setting the `MODELS_DIR` environment variable or passing
`--models-dir` to the CLI or API server.

---

## Prerequisites

```bash
pip install huggingface_hub[cli]
```

This installs the `hf` CLI (canonical name since huggingface_hub 0.26+; `huggingface-cli`
remains as an alias), used by `download.sh` to fetch model files.

**Phase 2 (gated tokenizer) also requires authentication:**

```bash
hf auth login
```

You must also accept the Gemma Terms of Use at
https://huggingface.co/google/translategemma-4b-it before downloading.

---

## Download

```bash
bash models/download.sh
```

The script downloads files in two phases:

**Phase 1 — GGUF weights (public, no auth required)**
From [`mradermacher/translategemma-4b-it-GGUF`](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF):

| File | Size | Description |
|------|------|-------------|
| `model-q8_0.gguf` | ~4.1 GB | Q8_0 quantised GGUF weights (default) |
| `model-q4k.gguf` | ~2.6 GB | Q4_K_M quantised GGUF weights (optional, `--q4` flag) |

**Phase 2 — Tokenizer + config (gated — requires HF login and Gemma license acceptance)**
From [`google/translategemma-4b-it`](https://huggingface.co/google/translategemma-4b-it):

> **Note:** These files are downloaded for reference and external tooling compatibility.
> The GGUF file embeds the tokenizer, so these are **not required** at runtime.

| File | Size | Description |
|------|------|-------------|
| `tokenizer.json` | <1 MB | HuggingFace fast tokenizer (reference only) |
| `tokenizer_config.json` | <1 MB | Tokenizer metadata (reference only) |
| `config.json` | <1 MB | Model configuration (reference only) |
| `special_tokens_map.json` | <1 MB | Special token definitions (reference only) |

> **Gated repo:** You must accept the Gemma Terms of Use at
> https://huggingface.co/google/translategemma-4b-it and run `hf auth login`
> before Phase 2 will succeed.

---

## Expected directory layout

After running `download.sh`, the model directory should look like this:

```
${MODELS_DIR}/translategemma-4b/
├── model-q8_0.gguf         # ~4.1 GB — Q8_0 quantised GGUF weights (default)
├── model-q4k.gguf          # ~2.6 GB — Q4_K_M quantised GGUF weights (optional)
├── config.json              # optional — for reference/external tooling
├── tokenizer.json           # optional — tokenizer is embedded in GGUF
├── tokenizer_config.json    # optional — for reference/external tooling
└── special_tokens_map.json  # optional — for reference/external tooling
```

Only `model-q8_0.gguf` is required. `model-q4k.gguf` is optional (download with `--q4` flag).

---

## Runtime model selection

The engine auto-detects the GGUF file with this priority:

1. `--model-file` CLI/API flag (highest priority)
2. `MODEL_FILE` environment variable
3. `model-q8_0.gguf` (preferred default — higher precision)
4. `model-q4k.gguf` (fallback if Q8_0 not present)
5. Any `*.gguf` in the directory (last resort)

```bash
# Use Q4_K_M via CLI flag
ut --model-file model-q4k.gguf translate -t "Hello" -l fr

# Use Q4_K_M via env var
MODEL_FILE=model-q4k.gguf cargo run -p translator-cli -- translate -t "Hello" -l fr

# API server with Q4_K_M
cargo run -p translator-api -- --model-file model-q4k.gguf
MODEL_FILE=model-q4k.gguf cargo run -p translator-api
```

| Format | File | Size | Throughput | Quality |
|--------|------|------|------------|---------|
| Q8_0 | `model-q8_0.gguf` | ~4.1 GB | Comparable to Q4 under llama.cpp | Highest precision among quantised formats (default) |
| Q4_K_M | `model-q4k.gguf` | ~2.6 GB | Comparable to Q8 under llama.cpp | Excellent — imperceptible difference from Q8_0 |

---

## Build targets

```bash
cargo build                   # CPU (default)
cargo build --features metal  # macOS GPU (Metal)
cargo build --features cuda   # Linux GPU (CUDA/NVIDIA)
```

---

## Hosting the model files

The GGUF weights (~2.6 GB) are too large to check into git. Three options for
distributing them to teammates or CI:

### Option A — Hugging Face Hub (recommended)

Create a private HuggingFace repository of type **model** and push the model
directory there. Anyone with access can pull it with `hf download` and
no conversion tooling is needed on their machine.

```bash
# Upload (once, after running download.sh)
hf upload your-org/universal-translator-models \
  ${MODELS_DIR}/translategemma-4b/ \
  --repo-type model

# Download (on a new machine)
hf download your-org/universal-translator-models \
  --local-dir ${MODELS_DIR}/translategemma-4b/ \
  --repo-type model
```

### Option B — GitHub Releases

Zip the model directory and attach it as a release asset. Useful if you want the
model version-locked to a specific code release.

### Option C — Cloud object storage (S3 / GCS / R2)

Upload the model directory to a bucket and sync it down in CI. Cloudflare R2 has
no egress fees, which makes it cost-effective for frequent downloads.

```bash
# Upload
aws s3 sync ${MODELS_DIR}/translategemma-4b/ s3://your-bucket/translategemma-4b/

# Download (e.g. in CI)
aws s3 sync s3://your-bucket/translategemma-4b/ ${MODELS_DIR}/translategemma-4b/
```

---

## Lingua: fully local language detection

universal-translator uses [Lingua](https://github.com/pemistahl/lingua-rs) to
automatically detect the source language of incoming text. Lingua is entirely
self-contained:

- All language model data is compiled directly into the binary as Rust crates —
  there are no external data files to manage.
- Detection covers 75+ languages.
- Zero network calls are made at runtime. Lingua works completely offline.

No configuration is required for language detection.
