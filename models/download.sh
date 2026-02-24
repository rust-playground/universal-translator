#!/usr/bin/env bash
# Download MADLAD-400-3B-MT GGUF weights and tokenizer from HuggingFace.
#
# Usage:
#   bash models/download.sh                        # → platform cache dir
#   MODELS_DIR=/custom/path bash models/download.sh
#
# Prerequisites: huggingface-cli  (pip install huggingface_hub[cli])
#   — or — curl (fallback, no auth / rate-limit handling)
#
# License: jbochi/madlad400-3b-mt is a community GGUF conversion of
#   google/madlad400-3b-mt.  Both repositories are Apache-2.0 licensed.
#   Commercial use permitted.
#
# Download size: ~1.65 GB (model-q4k.gguf, Q4_K mixed-precision)
# Disk after download: ~1.65 GB  (no conversion step needed)

set -euo pipefail

# Resolve platform-appropriate cache dir (mirrors dirs::cache_dir() in Rust)
case "$(uname -s)" in
  Darwin) _cache_base="${HOME}/Library/Caches" ;;
  *)      _cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}" ;;
esac
DEFAULT_MODELS_DIR="${_cache_base}/ut/models"
MODELS_DIR="${MODELS_DIR:-${DEFAULT_MODELS_DIR}}"
MADLAD_DIR="${MODELS_DIR}/madlad400-3b-mt"

mkdir -p "${MADLAD_DIR}"

HF_REPO="jbochi/madlad400-3b-mt"
FILES=(
  "model-q4k.gguf"
  "tokenizer.json"
  "tokenizer_config.json"
  "config.json"
)

echo "━━━  madlad400-3b-mt  (${HF_REPO})"
echo "     Output: ${MADLAD_DIR}"
echo ""

# Check if already downloaded (key file present)
if [[ -f "${MADLAD_DIR}/model-q4k.gguf" ]]; then
  echo "SKIP  madlad400-3b-mt  (model-q4k.gguf already exists)"
  exit 0
fi

# Prefer huggingface-cli; fall back to plain curl
if command -v huggingface-cli &>/dev/null; then
  echo "Downloading via huggingface-cli …"
  huggingface-cli download "${HF_REPO}" \
    "${FILES[@]}" \
    --local-dir "${MADLAD_DIR}"
else
  echo "huggingface-cli not found — falling back to curl"
  echo "(install with: pip install huggingface_hub[cli])"
  echo ""
  HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"
  for file in "${FILES[@]}"; do
    dest="${MADLAD_DIR}/${file}"
    if [[ -f "${dest}" ]]; then
      echo "SKIP  ${file}  (already exists)"
      continue
    fi
    echo "GET   ${file}"
    curl -L --progress-bar --output "${dest}" "${HF_BASE}/${file}"
  done
fi

echo ""
echo "OK    madlad400-3b-mt"
echo ""
echo "Build (CPU):"
echo "  cargo build -r"
echo "Build (Metal — macOS):"
echo "  cargo build -r --features metal"
echo "Build (CUDA — Linux NVIDIA):"
echo "  cargo build -r --features cuda"
