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
| `model-q4k.gguf` | ~2.6 GB | Q4_K_M quantised GGUF weights |

**Phase 2 — Tokenizer + config (gated — requires HF login and Gemma license acceptance)**
From [`google/translategemma-4b-it`](https://huggingface.co/google/translategemma-4b-it):

| File | Size | Description |
|------|------|-------------|
| `tokenizer.json` | <1 MB | HuggingFace fast tokenizer |
| `tokenizer_config.json` | <1 MB | Tokenizer metadata |
| `config.json` | <1 MB | Model configuration |
| `special_tokens_map.json` | <1 MB | Special token definitions |

> **Gated repo:** You must accept the Gemma Terms of Use at
> https://huggingface.co/google/translategemma-4b-it and run `hf auth login`
> before Phase 2 will succeed.

---

## Expected directory layout

After running `download.sh`, the model directory should look like this:

```
${MODELS_DIR}/translategemma-4b/
├── model-q4k.gguf          # ~2.6 GB — Q4_K_M quantised GGUF weights
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

If any of these files are missing the engine will fail to load the model at startup.

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
