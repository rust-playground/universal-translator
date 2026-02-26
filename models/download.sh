#!/usr/bin/env bash
# Download and convert the MADLAD-400-3B-MT model to CTranslate2 format.
#
# Usage:
#   bash models/download.sh                         # → ~/Library/Caches/ut/models/ (macOS)
#                                                   # → ~/.cache/ut/models/ (Linux)
#   MODELS_DIR=/custom/path bash models/download.sh # override output directory
#
# Prerequisites: pip install ctranslate2 transformers sentencepiece torch
#
# License: google/madlad400-3b-mt → Apache-2.0 (permits commercial use)
# Disk: ~12 GB download → ~3 GB output (int8 quantization)

set -euo pipefail

# Resolve platform-appropriate cache dir (mirrors dirs::cache_dir() in Rust)
case "$(uname -s)" in
  Darwin) _cache_base="${HOME}/Library/Caches" ;;
  *)      _cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}" ;;
esac
DEFAULT_MODELS_DIR="${_cache_base}/ut/models"
MODELS_DIR="${MODELS_DIR:-${DEFAULT_MODELS_DIR}}"

mkdir -p "${MODELS_DIR}"

# ── MADLAD-400-3B-MT ─────────────────────────────────────────────────────────
# Single model covering 62 verified languages.
MADLAD_DIR="${MODELS_DIR}/madlad400-3b-mt"
if [[ ! -f "${MADLAD_DIR}/model.bin" ]]; then
  echo "━━━  madlad400-3b-mt  (google/madlad400-3b-mt)"
  ct2-transformers-converter \
      --model google/madlad400-3b-mt \
      --output_dir "${MADLAD_DIR}" \
      --quantization int8 \
      --copy_files spiece.model \
      --force
  # SpmTokenizer expects source.spm / target.spm; MADLAD uses one shared spiece model
  cp "${MADLAD_DIR}/spiece.model" "${MADLAD_DIR}/source.spm"
  cp "${MADLAD_DIR}/spiece.model" "${MADLAD_DIR}/target.spm"
  echo "OK    madlad400-3b-mt"
else
  echo "SKIP  madlad400-3b-mt  (already exists)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model ready in: ${MADLAD_DIR}"
