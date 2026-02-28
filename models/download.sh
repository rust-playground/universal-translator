#!/usr/bin/env bash
# Download TranslateGemma 4B GGUF weights and tokenizer from HuggingFace.
#
# Usage:
#   bash models/download.sh                        # → platform cache dir
#   MODELS_DIR=/custom/path bash models/download.sh
#
# Prerequisites: huggingface-cli  (pip install huggingface_hub[cli])
#   huggingface-cli login  (required for the gated tokenizer repo)
#
# Model: TranslateGemma 4B — instruction-tuned Gemma 3 4B for translation.
#
# Phase 1 — GGUF weights (public, no auth):
#   mradermacher/translategemma-4b-it-GGUF
#   File: translategemma-4b-it.Q4_K_M.gguf → renamed to model-q4k.gguf
#
# Phase 2 — tokenizer + config (GATED — requires HF login + license acceptance):
#   google/translategemma-4b-it
#   Accept the license at: https://huggingface.co/google/translategemma-4b-it
#
# Download size: ~2.6 GB (model-q4k.gguf, Q4_K_M quantisation)

set -euo pipefail

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

# Check if already downloaded (key file present)
if [[ -f "${GEMMA_DIR}/model-q4k.gguf" ]]; then
  echo "SKIP  translategemma-4b  (model-q4k.gguf already exists)"
  exit 0
fi

if ! command -v huggingface-cli &>/dev/null; then
  echo "ERROR: huggingface-cli not found."
  echo "Install with: pip install huggingface_hub[cli]"
  exit 1
fi

# ── Phase 1: GGUF weights (public, no auth) ──────────────────────────────────
echo "Phase 1/2 — Downloading GGUF weights (public)…"
GGUF_REPO="mradermacher/translategemma-4b-it-GGUF"
GGUF_FILE="translategemma-4b-it.Q4_K_M.gguf"

huggingface-cli download "${GGUF_REPO}" \
  "${GGUF_FILE}" \
  --local-dir "${GEMMA_DIR}"

mv "${GEMMA_DIR}/${GGUF_FILE}" "${GEMMA_DIR}/model-q4k.gguf"
echo "OK    model-q4k.gguf"
echo ""

# ── Phase 2: tokenizer + config (gated) ──────────────────────────────────────
echo "Phase 2/2 — Downloading tokenizer + config (gated repo)…"
echo "  Requires: huggingface-cli login AND license acceptance at"
echo "  https://huggingface.co/google/translategemma-4b-it"
echo ""

GATED_REPO="google/translategemma-4b-it"

if ! huggingface-cli download "${GATED_REPO}" \
  tokenizer.json tokenizer_config.json config.json special_tokens_map.json \
  --local-dir "${GEMMA_DIR}"; then
  echo ""
  echo "ERROR: Failed to download from ${GATED_REPO}."
  echo ""
  echo "This repo is gated. To fix:"
  echo "  1. Accept the license at https://huggingface.co/google/translategemma-4b-it"
  echo "  2. Log in: huggingface-cli login"
  echo "  3. Re-run: bash models/download.sh"
  exit 1
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
