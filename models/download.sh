#!/usr/bin/env bash
# Download TranslateGemma 4B GGUF weights and tokenizer from HuggingFace.
#
# Usage:
#   bash models/download.sh                        # Q8_0 only (default)
#   bash models/download.sh --q4                   # also download Q4_K_M
#   MODELS_DIR=/custom/path bash models/download.sh
#
# Prerequisites: hf CLI  (pip install huggingface_hub[cli])
#   hf auth login  (required for the gated tokenizer repo)
#
# Model: TranslateGemma 4B — instruction-tuned Gemma 3 4B for translation.
#
# Phase 1 — GGUF weights (public, no auth):
#   mradermacher/translategemma-4b-it-GGUF
#   Q8_0:   translategemma-4b-it.Q8_0.gguf   → model-q8_0.gguf  (~4.1 GB)
#   Q4_K_M: translategemma-4b-it.Q4_K_M.gguf → model-q4k.gguf  (~2.6 GB)
#
# Phase 2 — tokenizer + config (GATED — requires HF login + license acceptance):
#   google/translategemma-4b-it
#   Accept the license at: https://huggingface.co/google/translategemma-4b-it
#   NOTE: These files are optional — the GGUF embeds the tokenizer. Kept for
#   reference and compatibility with external tooling (e.g. HF Transformers).
#
# To use Q4_K_M at runtime: MODEL_FILE=model-q4k.gguf cargo run ...

set -euo pipefail

DOWNLOAD_Q4=false
for arg in "$@"; do
  case "$arg" in
    --q4) DOWNLOAD_Q4=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# Resolve platform-appropriate cache dir (mirrors dirs::cache_dir() in Rust)
case "$(uname -s)" in
  Darwin) _cache_base="${HOME}/Library/Caches" ;;
  *)      _cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}" ;;
esac
DEFAULT_MODELS_DIR="${_cache_base}/ut/models"
MODELS_DIR="${MODELS_DIR:-${DEFAULT_MODELS_DIR}}"
GEMMA_DIR="${MODELS_DIR}/translategemma-4b"

mkdir -p "${GEMMA_DIR}"

echo "━━━  translategemma-4b"
echo "     Output: ${GEMMA_DIR}"
echo ""

if command -v hf &>/dev/null; then
  HF_CLI="hf"
elif command -v huggingface-cli &>/dev/null; then
  HF_CLI="huggingface-cli"
else
  echo "ERROR: hf CLI not found."
  echo "Install with: pip install huggingface_hub[cli]"
  exit 1
fi

GGUF_REPO="mradermacher/translategemma-4b-it-GGUF"

# ── Phase 1a: Q8_0 weights (public, no auth) ─────────────────────────────────
if [[ -f "${GEMMA_DIR}/model-q8_0.gguf" ]]; then
  echo "SKIP  model-q8_0.gguf (already exists)"
else
  echo "Phase 1a — Downloading Q8_0 GGUF weights (~4.1 GB)…"
  GGUF_FILE="translategemma-4b-it.Q8_0.gguf"

  ${HF_CLI} download "${GGUF_REPO}" \
    "${GGUF_FILE}" \
    --local-dir "${GEMMA_DIR}"

  mv "${GEMMA_DIR}/${GGUF_FILE}" "${GEMMA_DIR}/model-q8_0.gguf"
  echo "OK    model-q8_0.gguf"
fi
echo ""

# ── Phase 1b: Q4_K_M weights (optional, public) ─────────────────────────────
if [[ "${DOWNLOAD_Q4}" == "true" ]]; then
  if [[ -f "${GEMMA_DIR}/model-q4k.gguf" ]]; then
    echo "SKIP  model-q4k.gguf (already exists)"
  else
    echo "Phase 1b — Downloading Q4_K_M GGUF weights (~2.6 GB)…"
    GGUF_FILE_Q4="translategemma-4b-it.Q4_K_M.gguf"

    ${HF_CLI} download "${GGUF_REPO}" \
      "${GGUF_FILE_Q4}" \
      --local-dir "${GEMMA_DIR}"

    mv "${GEMMA_DIR}/${GGUF_FILE_Q4}" "${GEMMA_DIR}/model-q4k.gguf"
    echo "OK    model-q4k.gguf"
  fi
  echo ""
fi

# ── Phase 2: tokenizer + config (gated, optional) ──────────────────────────
# NOTE: The GGUF file embeds the tokenizer, so these files are NOT required at
# runtime. They are downloaded for reference and compatibility with external
# tooling (e.g. HF Transformers, tokenizer inspection scripts).
if [[ -f "${GEMMA_DIR}/tokenizer.json" ]]; then
  echo "SKIP  tokenizer + config (already exists)"
else
  echo "Phase 2 — Downloading tokenizer + config (gated repo)…"
  echo "  Requires: hf auth login AND license acceptance at"
  echo "  https://huggingface.co/google/translategemma-4b-it"
  echo ""

  GATED_REPO="google/translategemma-4b-it"

  if ! ${HF_CLI} download "${GATED_REPO}" \
    tokenizer.json tokenizer_config.json config.json special_tokens_map.json \
    --local-dir "${GEMMA_DIR}"; then
    echo ""
    echo "ERROR: Failed to download from ${GATED_REPO}."
    echo ""
    echo "This repo is gated. To fix:"
    echo "  1. Accept the license at https://huggingface.co/google/translategemma-4b-it"
    echo "  2. Log in: hf auth login"
    echo "  3. Re-run: bash models/download.sh"
    exit 1
  fi
fi

echo ""
echo "OK    translategemma-4b"
echo ""
echo "Build (CPU):"
echo "  cargo build -r"
echo "Build (Metal — macOS):"
echo "  cargo build -r --features metal"
echo "Build (CUDA — Linux NVIDIA):"
echo "  cargo build -r --features cuda"
