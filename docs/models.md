# Model Management

This guide explains how to convert and install Helsinki-NLP OPUS-MT models for use with universal-translator.

---

## Prerequisites

- Python 3.x
- cmake (required to build CTranslate2 native extensions)
- Rust toolchain (already needed to build the project)

## Install conversion tooling

```bash
pip install ctranslate2 transformers sentencepiece torch
```

## Convert a model

Use the `ct2-transformers-converter` command-line tool that ships with the `ctranslate2` Python package.

Example: English to French

```bash
ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-en-fr \
  --output_dir models/en-fr \
  --quantization int8 \
  --copy_files source.spm target.spm \
  --force
```

### Flag reference

| Flag | Purpose |
|------|---------|
| `--model` | HuggingFace model ID to download and convert |
| `--output_dir` | Directory to write the converted model into |
| `--quantization int8` | Quantize weights to 8-bit integers, roughly halving model size with minimal quality loss |
| `--copy_files source.spm target.spm` | Copy the SentencePiece tokenizer files into the output directory — the converter does not do this automatically |
| `--force` | Overwrite the output directory if it already exists |

## Verify the output

After conversion, `models/en-fr/` should contain:

```
models/en-fr/
├── model.bin              # Converted CTranslate2 weights (~77 MB with int8)
├── source.spm             # Source-language SentencePiece model
├── target.spm             # Target-language SentencePiece model
├── config.json            # Model configuration
└── shared_vocabulary.json # Vocabulary used by source and target tokenizers
```

If any of these files are missing the engine will fail to load the model at startup.

## Available OPUS-MT pairs

The table below lists common Helsinki-NLP model pairs. Any pair available on HuggingFace under the `Helsinki-NLP` organisation can be converted with the same command — just substitute the model name and output directory.

| Language Pair | HuggingFace Model | Output Dir |
|---------------|-------------------|------------|
| English -> French | `Helsinki-NLP/opus-mt-en-fr` | `models/en-fr` |
| English -> German | `Helsinki-NLP/opus-mt-en-de` | `models/en-de` |
| English -> Spanish | `Helsinki-NLP/opus-mt-en-es` | `models/en-es` |
| English -> Italian | `Helsinki-NLP/opus-mt-en-it` | `models/en-it` |
| English -> Portuguese | `Helsinki-NLP/opus-mt-en-pt` | `models/en-pt` |
| English -> Chinese | `Helsinki-NLP/opus-mt-en-zh` | `models/en-zh` |
| French -> English | `Helsinki-NLP/opus-mt-fr-en` | `models/fr-en` |
| German -> English | `Helsinki-NLP/opus-mt-de-en` | `models/de-en` |
| Spanish -> English | `Helsinki-NLP/opus-mt-es-en` | `models/es-en` |

The engine scans the `models/` directory at startup and loads every valid model it finds. Adding a new language pair requires no Rust code changes — drop the converted directory in and restart.

## Hosting pre-converted models

The converted model directories (model.bin + tokenizer files) are too large to
check into git (~50–200 MB each, ~4 GB total). Three options for distributing
them to teammates or CI:

### Option A — Hugging Face Hub (recommended)

Create a private HuggingFace repository of type **model** and push the
converted directories there with `huggingface-cli upload`. Anyone with access
can then pull them with `huggingface-cli download` and no Python conversion
tooling is needed on their machine. This also keeps the models discoverable and
versioned alongside the rest of the ML ecosystem.

```bash
# Upload (once, after running download.sh)
huggingface-cli upload your-org/universal-translator-models models/ --repo-type model

# Download (on a new machine)
huggingface-cli download your-org/universal-translator-models --local-dir models/ --repo-type model
```

### Option B — GitHub Releases

Zip the `models/` directory and attach it as a release asset. Useful if you
want the models version-locked to a specific code release. Free up to 2 GB per
release asset; for the full set you would need to split into multiple archives.

### Option C — Cloud object storage (S3 / GCS / R2)

Upload the `models/` directory to a bucket and sync it down in CI. Cloudflare
R2 has no egress fees, which makes it cost-effective for frequent downloads.

```bash
# Upload
aws s3 sync models/ s3://your-bucket/models/

# Download (e.g. in CI)
aws s3 sync s3://your-bucket/models/ models/
```

---

## Lingua: fully local language detection

universal-translator uses [Lingua](https://github.com/pemistahl/lingua-rs) to automatically detect the source language of incoming text. Lingua is entirely self-contained:

- All language model data is compiled directly into the binary as Rust crates — there are no external data files to manage.
- Detection covers 75+ languages.
- Zero network calls are made at runtime. Lingua works completely offline.

No configuration is required for language detection.
