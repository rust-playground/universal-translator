# Model Management

This guide explains how to download and manage the MADLAD-400-3B-MT model for use
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

This installs `huggingface-cli`, used by `download.sh` to fetch model files.
`curl` is used as a fallback if `huggingface-cli` is unavailable.

---

## Download

```bash
bash models/download.sh
```

The script downloads the following files from
[`jbochi/madlad400-3b-mt`](https://huggingface.co/jbochi/madlad400-3b-mt) on
Hugging Face into `${MODELS_DIR}/madlad400-3b-mt/`:

| File | Size | Description |
|------|------|-------------|
| `model-q4k.gguf` | ~1.65 GB | int4 quantised GGUF weights |
| `config.json` | <1 MB | Model configuration |
| `tokenizer.json` | <1 MB | HuggingFace fast tokenizer |
| `tokenizer_config.json` | <1 MB | Tokenizer metadata |

---

## Expected directory layout

After running `download.sh`, the model directory should look like this:

```
${MODELS_DIR}/madlad400-3b-mt/
├── model-q4k.gguf          # ~1.65 GB — int4 quantised GGUF weights
├── config.json
├── tokenizer.json
└── tokenizer_config.json
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

The GGUF weights (~1.65 GB) are too large to check into git. Three options for
distributing them to teammates or CI:

### Option A — Hugging Face Hub (recommended)

Create a private HuggingFace repository of type **model** and push the model
directory there. Anyone with access can pull it with `huggingface-cli download` and
no conversion tooling is needed on their machine.

```bash
# Upload (once, after running download.sh)
huggingface-cli upload your-org/universal-translator-models \
  ${MODELS_DIR}/madlad400-3b-mt/ \
  --repo-type model

# Download (on a new machine)
huggingface-cli download your-org/universal-translator-models \
  --local-dir ${MODELS_DIR}/madlad400-3b-mt/ \
  --repo-type model
```

### Option B — GitHub Releases

Zip the model directory and attach it as a release asset. Useful if you want the
model version-locked to a specific code release. The single directory (~1.65 GB)
fits within GitHub's 2 GB release asset limit.

### Option C — Cloud object storage (S3 / GCS / R2)

Upload the model directory to a bucket and sync it down in CI. Cloudflare R2 has
no egress fees, which makes it cost-effective for frequent downloads.

```bash
# Upload
aws s3 sync ${MODELS_DIR}/madlad400-3b-mt/ s3://your-bucket/madlad400-3b-mt/

# Download (e.g. in CI)
aws s3 sync s3://your-bucket/madlad400-3b-mt/ ${MODELS_DIR}/madlad400-3b-mt/
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
