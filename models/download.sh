#!/usr/bin/env bash
# Export MADLAD-400-3B-MT to ONNX format via HuggingFace Optimum.
#
# Usage:
#   bash models/download.sh                        # → platform cache dir
#   MODELS_DIR=/custom/path bash models/download.sh
#
# Prerequisites:
#   python3 (Python dependencies are installed automatically via models/requirements.txt)
#   (huggingface-cli login  — only needed for gated models; madlad400 is public)
#
# Output directory: ${MODELS_DIR}/madlad400-3b-mt-onnx/
#   encoder_model.onnx              (~4 GB, fp32)
#   decoder_model.onnx              (~4 GB, fp32)
#   decoder_with_past_model.onnx    (~4 GB, fp32)
#   config.json, tokenizer.json, ...
#
# Total disk: ~12 GB (fp32).  Add --quantize for int8 CPU-optimized export (~7 GB).
# RAM required during export: ~8 GB.
#
# License: google/madlad400-3b-mt is Apache-2.0 licensed.
#   Commercial use permitted.

set -euo pipefail

# Resolve the script's own directory so paths are repo-relative regardless
# of where the caller invokes this script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve platform-appropriate cache dir (mirrors dirs::cache_dir() in Rust)
case "$(uname -s)" in
  Darwin) _cache_base="${HOME}/Library/Caches" ;;
  *)      _cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}" ;;
esac
DEFAULT_MODELS_DIR="${_cache_base}/ut/models"
MODELS_DIR="${MODELS_DIR:-${DEFAULT_MODELS_DIR}}"
ONNX_DIR="${MODELS_DIR}/madlad400-3b-mt-onnx"

echo "━━━  madlad400-3b-mt ONNX export"
echo "     Output: ${ONNX_DIR}"
echo ""

# ── Skip if already exported ──────────────────────────────────────────────
if [[ -f "${ONNX_DIR}/encoder_model.onnx" ]] && \
   [[ -f "${ONNX_DIR}/decoder_model.onnx" ]] && \
   [[ -f "${ONNX_DIR}/decoder_with_past_model.onnx" ]]; then
  echo "SKIP  madlad400-3b-mt-onnx  (all 3 ONNX files already exist)"
  exit 0
fi

# ── Check Python ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found." >&2
  exit 1
fi

# ── Install Python dependencies ───────────────────────────────────────────
echo "Checking Python dependencies..."
if ! python3 -c "from optimum.exporters.onnx import main_export; import transformers, sentencepiece, onnxruntime" &>/dev/null 2>&1; then
  echo "Installing Python dependencies (this may take a minute)..."
  python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

# ── Run the export script ─────────────────────────────────────────────────
python3 "${SCRIPT_DIR}/export_onnx.py" --output "${ONNX_DIR}"

echo ""
echo "OK    madlad400-3b-mt ONNX export complete"
echo ""
echo "Build (CPU):"
echo "  cargo build -r"
echo "Build (CoreML — macOS):"
echo "  cargo build -r --features coreml"
echo "Build (CUDA — Linux NVIDIA):"
echo "  cargo build -r --features cuda"
